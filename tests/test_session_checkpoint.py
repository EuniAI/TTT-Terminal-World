import copy
import pytest
import torch
from terminal_world.checkpoint import load_checkpoint, save_checkpoint, save_session, resume_session
from terminal_world.model.loop import TrainingState, train_window
from terminal_world.model.data import Event, WindowBatch
from terminal_world.session import Session
from terminal_world.evaluate import evaluate


def window(encoder, start, commands):
    events = [Event(k, c, o, start + i) for i, (k, c, o) in enumerate(commands)]
    return WindowBatch([encoder.encode(e) for e in events], start + len(events), events[-1].command != "close")


def test_training_checkpoint_matches_uninterrupted(tmp_path, tiny):
    model, fast, encoder, config = tiny
    state = TrainingState(fast)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.outer_lr, weight_decay=0)
    first = window(encoder, 0, [("control", "open", ""), ("shell", "x", "a")])
    second = window(encoder, 2, [("shell", "y", ""), ("control", "close", "")])
    train_window(model, optimizer, state, first, config.training)
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, state, optimizer, config, encoder.tokenizer, {}, "digest")
    loaded_model, loaded_state, loaded_opt, _, _, _ = load_checkpoint(path, expected_manifest_digest="digest")
    assert loaded_state.is_open and loaded_state.next_cursor == 2
    a, _ = train_window(model, optimizer, state, second, config.training)
    b, _ = train_window(loaded_model, loaded_opt, loaded_state, second, config.training)
    assert a == b
    for x, y in zip(model.parameters(), loaded_model.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for key in state.fast_weights:
        torch.testing.assert_close(state.fast_weights[key], loaded_state.fast_weights[key], rtol=0, atol=0)
    assert state.next_cursor == loaded_state.next_cursor == 4
    with pytest.raises(ValueError, match="manifest"):
        load_checkpoint(path, expected_manifest_digest="wrong")
    payload = torch.load(path, weights_only=False)
    del payload["fast_weights"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="incomplete"):
        load_checkpoint(path)


def test_session_resume_predict_and_evaluation_copy(tmp_path, tiny):
    model, fast, encoder, config = tiny
    session = Session(model, fast, encoder, alpha=config.training.inner_lr)
    session.control("open")
    session.observe("x", "a")
    path = tmp_path / "session.pt"
    save_session(path, session, config)
    restored = resume_session(path)
    assert restored.observations == session.observations == 2
    before = {k: v.clone() for k, v in restored.fast_weights.items()}
    assert restored.predict("y") == session.predict("y")
    for key in before:
        assert torch.equal(before[key], restored.fast_weights[key])
    with pytest.raises(ValueError, match="nested"):
        restored.control("open")
    assert restored.observe("y", "b") == session.observe("y", "b")
    restored.control("close")
    memory = {k: v.clone() for k, v in restored.fast_weights.items()}
    events = [Event("control", "open"), Event("shell", "x", "c"), Event("control", "close")]
    _, frozen = evaluate(restored.model, restored.fast_weights, encoder, events, alpha=0.2, adapt=False)
    _, adapted = evaluate(restored.model, restored.fast_weights, encoder, events, alpha=0.2)
    for key in memory:
        assert torch.equal(memory[key], restored.fast_weights[key])
        assert torch.equal(memory[key], frozen.fast_weights[key])
    assert any(not torch.equal(memory[k], adapted.fast_weights[k]) for k in memory)


def test_nonfinite_window_does_not_commit(tiny):
    model, fast, encoder, config = tiny
    state = TrainingState(fast)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    with torch.no_grad():
        model.lm_head.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        train_window(model, optimizer, state, window(encoder, 0, [("control", "open", "")]), config.training)
    assert state.next_cursor == 0 and state.windows_completed == 0


def test_generation_metrics_and_frozen_training_copy(tiny):
    model, fast, encoder, config = tiny
    original = {k: v.clone() for k, v in fast.items()}
    events = [Event("control", "open"), Event("shell", "x", ""),
              Event("shell", "y", "a" * 8), Event("shell", "z", "b" * 9), Event("control", "close")]
    result, _ = evaluate(model, fast, encoder, events, alpha=config.training.inner_lr, adapt=False, generate=True)
    assert result["complete"]["generated_events"] == 1
    assert result["body_complete_without_end"]["generated_events"] == 1
    assert result["body_truncated"]["generated_events"] == 1
    assert result["body_truncated"]["exact_with_end_rate"] is None
    assert all(p.requires_grad for p in model.parameters())
    assert all(torch.equal(fast[k], original[k]) for k in fast)
