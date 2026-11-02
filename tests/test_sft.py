import copy

import pytest
import torch

from terminal_world.config import RunConfig
from terminal_world.evaluate import evaluate
from terminal_world.model.data import Event
from terminal_world.model.meta_model import loss_for_window, step_loss
from terminal_world.model.transformer import create_model


def test_sft_matches_supervised_loss_and_gradients(tiny):
    _, _, encoder, config = tiny
    config.model.adaptive_layers = 0
    model, fast = create_model(config.model)
    assert fast == {}
    assert all(layer.prime_norm is None for layer in model.layers)
    steps = [encoder.encode(Event("control", "open")),
             encoder.encode(Event("shell", "x", "abc")),
             encoder.encode(Event("shell", "longer", "d"))]
    reference = copy.deepcopy(model)
    losses = [step_loss(reference, {}, step) for step in steps]
    expected = sum(loss for loss, _ in losses) / sum(n for _, n in losses)
    actual, next_fast, records = loss_for_window(model, fast, steps, alpha=999, prefix_batch_size=2)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    assert next_fast == {}
    assert all(r["update_norm"] == 0 for r in records)


def test_sft_evaluation_is_frozen_even_when_adapt_requested(tiny):
    _, _, encoder, config = tiny
    config.model.adaptive_layers = 0
    model, fast = create_model(config.model)
    before = copy.deepcopy(model.state_dict())
    events = [Event("control", "open"), Event("shell", "x", "a"), Event("control", "close")]
    frozen, _ = evaluate(model, fast, encoder, events, alpha=1, adapt=False, generate=True)
    requested, session = evaluate(model, fast, encoder, events, alpha=1, adapt=True, generate=True)
    assert frozen == requested
    assert session.fast_weights == {}
    for name, value in session.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_sft_config_and_missing_ttt_memory(tiny):
    sft = RunConfig.load("configs/experiment/terminal_sft.yaml")
    ttt = RunConfig.load("configs/experiment/terminal_ttt.yaml")
    assert sft.model.adaptive_layers == 0
    assert sft.training.sft_batch_size == 4
    assert sft.model.max_seq_len == 32768
    assert sft.model.model_id == ttt.model.model_id
    assert sft.model.revision == ttt.model.revision
    assert sft.training.max_output_tokens == ttt.training.max_output_tokens
    # Permitting an empty SFT state must never permit lost TTT memory.
    model, _, encoder, _ = tiny
    with pytest.raises(ValueError, match="fast-weight names mismatch"):
        loss_for_window(model, {}, [encoder.encode(Event("control", "open"))], alpha=0.1)


def test_sft_optimizer_matches_ordinary_supervised_update(tiny):
    from terminal_world.model.data import WindowBatch
    from terminal_world.model.loop import TrainingState, train_window
    from terminal_world.schedule import make_schedule
    from terminal_world.train import events_per_step

    _, _, encoder, config = tiny
    config.model.adaptive_layers = 0
    config.training.sft_batch_size = 1
    config.training.outer_lr_schedule = "warmup_cosine"
    assert events_per_step(config) == 1
    assert make_schedule(config.training, 6, 2, events_per_step=1)["total_windows"] == 12
    model, fast = create_model(config.model)
    reference = copy.deepcopy(model)
    steps = [encoder.encode(Event("shell", "x", "abc")),
             encoder.encode(Event("shell", "y", "d"))]
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=0.001)
    losses = [step_loss(reference, {}, step) for step in steps]
    expected = sum(loss for loss, _ in losses) / sum(n for _, n in losses)
    expected.backward()
    ref_opt.step()
    train_window(model, opt, TrainingState(fast), WindowBatch(steps, 2, True), config.training)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q)
