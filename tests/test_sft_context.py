from argparse import Namespace
import copy
import json

import torch
import yaml
import pytest

from terminal_world.checkpoint import load_checkpoint, resume_session, save_session
from terminal_world.dataloader.encoding import ByteTokenizer, Encoder
from terminal_world.dataloader.events import windows
from terminal_world.dataloader.trajectory import EventStream, build_manifest
from terminal_world.model.data import Event
from terminal_world.model.transformer import create_model
from terminal_world.session import Session
from terminal_world.train import run


def test_context_mask_boundaries_and_budget():
    encoder = Encoder(ByteTokenizer(), 8, 40, trajectory_context=True)
    history = []
    first = Event("shell", "x", "answer")
    current = Event("shell", "y", "secret")
    encoder.observe(history, first)
    prompt = encoder.prompt(current, history)
    assert encoder.tokenizer.encode_text("answer") == prompt[3:9]
    assert encoder.prompt(Event("shell", "y", "different"), history) == prompt
    step = encoder.encode(current, history)
    assert not step.loss_masks[:len(prompt)-1].any()
    assert step.target_tokens[step.loss_masks].tolist() == list(b"secret") + [261]
    encoder.observe(history, Event("shell", "z", "b" * 16))
    retained = encoder.prompt(current, history)
    assert len(retained) + 8 <= 40
    assert list(b"answer") != retained[3:9]
    assert retained[1] == ord("z")  # Most recent complete turn remains.
    for boundary in ("close", "open"):
        encoder.observe(history, Event("control", boundary))
        assert history == []
    assert encoder.prompt(current, history) == encoder.prompt(current)


def test_train_and_session_context_agree_and_session_resumes(tmp_path, tiny):
    _, _, _, config = tiny
    config.model.adaptive_layers = 0
    config.training.trajectory_context = True
    encoder = Encoder(ByteTokenizer(), 8, 128, trajectory_context=True)
    model, fast = create_model(config.model)
    session = Session(model, fast, encoder, alpha=0.1)
    events = [Event("control", "open", cursor=0), Event("shell", "x", "a", 1),
              Event("shell", "y", "b", 2), Event("control", "close", cursor=3),
              Event("control", "open", cursor=4), Event("shell", "z", "c", 5)]
    steps = [s for w in windows(events, encoder, 2) for s in w.steps]
    for event, expected in zip(events, steps):
        actual = encoder.encode(event, session.history)
        assert torch.equal(actual.input_ids, expected.input_ids)
        before = copy.deepcopy(session.history)
        if event.kind == "shell":
            session.generate_tokens(event.command)
            assert session.history == before  # Predictions do not enter history.
        session.apply_event(event)
        if event.cursor == 1:
            path = tmp_path / "session.pt"
            save_session(path, session, config)
            session = resume_session(path)
            assert session.history
        if event.command == "close":
            assert not session.history
    assert steps[-1].input_ids.tolist() == encoder.encode(events[-1]).input_ids.tolist()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_context_training_resume_mid_trajectory(tmp_path, tiny, batch_size):
    _, _, _, config = tiny
    config.model.adaptive_layers = 0
    config.training.trajectory_context = True
    config.training.unroll_events = 2
    config.training.sft_batch_size = batch_size
    config.training.outer_lr_schedule = "warmup_cosine"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config.to_dict()))
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"trajectory": [
        {"command": str(i), "output": "out" + str(i)} for i in range(4)
    ]}) + "\n")
    manifest = build_manifest([data], 0)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    full, resumed = tmp_path / "full.pt", tmp_path / "resumed.pt"
    def args(output, resume=None, limit=None):
        return Namespace(config=str(cfg) if resume is None else None, manifest=str(path),
                         output=str(output), resume=str(resume) if resume else None,
                         device="cpu", epochs=2, max_windows=limit)
    run(args(full))
    run(args(resumed, limit=2 // batch_size))
    _, state, _, _, encoder, _ = load_checkpoint(resumed)
    assert state.next_cursor == 2 and state.is_open
    history = EventStream(manifest).history_before(state.next_cursor, encoder)
    assert history
    assert resume_session(resumed).history == history
    run(args(resumed, resume=resumed))
    a, sa, _, _, _, _ = load_checkpoint(full)
    b, sb, _, _, _, _ = load_checkpoint(resumed)
    assert sa.epoch == sb.epoch == 2
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
