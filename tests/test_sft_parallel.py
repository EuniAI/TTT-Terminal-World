import copy
import json
from argparse import Namespace

import pytest
import torch
import yaml

from terminal_world.checkpoint import load_checkpoint
from terminal_world.dataloader.sft import shell_batches
from terminal_world.dataloader.trajectory import build_manifest
from terminal_world.model.data import Event
from terminal_world.model.meta_model import step_loss
from terminal_world.model.sft import batch_loss
from terminal_world.model.transformer import create_model
from terminal_world.sft_train import run


def test_parallel_batch_matches_serial_loss_and_gradients(tiny):
    _, _, encoder, config = tiny
    config.model.adaptive_layers = 0
    encoder.trajectory_context = True
    model, _ = create_model(config.model)
    reference = copy.deepcopy(model)
    events = [Event("control", "open"), Event("shell", "a", "hello"),
              Event("shell", "longer command", "x"), Event("control", "close"),
              Event("control", "open"), Event("shell", "z", ""),
              Event("shell", "q", "abc"),
              *[Event("shell", str(i), "a") for i in range(4)], Event("control", "close")]
    steps = next(shell_batches(events, encoder, 8)).steps
    calls = []
    hook = model.layers[0].register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[0]))
    actual, records = batch_loss(model, steps, pad_id=encoder.tokenizer.ids["pad"])
    hook.remove()
    assert calls == [8]  # One batched transformer call, not eight serial calls.
    losses = [step_loss(reference, {}, step) for step in steps]
    expected = sum(loss for loss, _ in losses) / sum(n for _, n in losses)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    for record, (loss, count) in zip(records, losses):
        assert record["tokens"] == count
        assert record["nll_sum"] == pytest.approx(loss.item())
    # Changing padding IDs cannot affect real-token losses under causal attention.
    changed, _ = batch_loss(model, steps, pad_id=0)
    torch.testing.assert_close(changed, actual)


@pytest.mark.parametrize("legacy_resume", [False, True])
def test_parallel_entry_resume_and_tail_batch(tmp_path, tiny, capsys, legacy_resume):
    _, _, _, config = tiny
    config.model.adaptive_layers = 0
    config.training.trajectory_context = True
    config.training.sft_batch_size = 8
    config.training.outer_lr_schedule = "warmup_cosine"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config.to_dict()))
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"trajectory": [
        {"command": str(i), "output": "a" * (i % 4)} for i in range(18)
    ]}) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest([data], 0)))
    def args(output, resume=None, limit=None):
        return Namespace(config=str(cfg) if resume is None else None, manifest=str(manifest),
                         output=str(output), resume=str(resume) if resume else None,
                         device="cpu", epochs=2, max_steps=limit, save_every_steps=50)
    full, resumed = tmp_path / "full.pt", tmp_path / "resume.pt"
    run(args(full))
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["batch_size"] for r in records if "loss" in r] == [8, 8, 2, 8, 8, 2]
    if legacy_resume:
        from terminal_world.train import run as legacy_run
        legacy_args = args(resumed, limit=2)
        legacy_args.max_windows = legacy_args.max_steps
        legacy_args.save_every_windows = legacy_args.save_every_steps
        legacy_run(legacy_args)
    else:
        run(args(resumed, limit=2))
    _, state, _, _, _, _ = load_checkpoint(resumed)
    assert state.next_cursor == (16 if legacy_resume else 17) and state.is_open
    run(args(resumed, resume=resumed))
    a, sa, _, _, _, _ = load_checkpoint(full)
    b, sb, _, _, _, _ = load_checkpoint(resumed)
    assert sa.epoch == sb.epoch == 2
    for p, q in zip(a.parameters(), b.parameters()):
        if legacy_resume:
            assert torch.isfinite(q).all()  # Earlier control training cannot be undone.
        else:
            torch.testing.assert_close(p, q, atol=0, rtol=0)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    # The resumed shell-only loop never emits a control training target.
    resumed_records = [r for r in records if "learning_rate" in r]
    assert all(r["control"]["events"] == 0 for r in resumed_records)


def test_sft_entry_rejects_ttt_config(tmp_path, tiny):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(tiny[3].to_dict()))
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"trajectory": [{"command": "x", "output": "a"}]}) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest([data], 0)))
    with pytest.raises(ValueError, match="adaptive_layers=0"):
        run(Namespace(config=str(cfg), manifest=str(manifest), output=str(tmp_path / "out.pt"),
                      resume=None, device="cpu", epochs=1, max_steps=None, save_every_steps=50))


@pytest.mark.parametrize("stop", ["SIGTERM", "error", "log_error"])
def test_parallel_stop_and_failure_checkpoint(tmp_path, tiny, monkeypatch, stop):
    import signal
    import terminal_world.sft_train as training
    from terminal_world.tracking import Tracker

    config = tiny[3]
    config.model.adaptive_layers = 0
    config.training.sft_batch_size = 8
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config.to_dict()))
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"trajectory": [
        {"command": str(i), "output": "a"} for i in range(9)
    ]}) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest([data], 0)))
    output = tmp_path / "out.pt"
    original = training.train_batch
    def step(*args, **kwargs):
        result = original(*args, **kwargs)
        if stop == "SIGTERM":
            signal.raise_signal(signal.SIGTERM)
        elif stop == "error":
            raise RuntimeError("unsafe training failure")
        return result
    def fail_log(*args, **kwargs):
        raise RuntimeError("logging failed")
    monkeypatch.setattr(training, "train_batch", step)
    if stop == "log_error":
        monkeypatch.setattr(Tracker, "log", fail_log)
    args = Namespace(config=str(cfg), manifest=str(manifest), output=str(output),
                     resume=None, device="cpu", epochs=1, max_steps=None, save_every_steps=50)
    previous = signal.getsignal(signal.SIGTERM)
    if stop == "SIGTERM":
        run(args)
    else:
        with pytest.raises(RuntimeError):
            run(args)
    payload = torch.load(output, weights_only=False)
    assert payload["windows_completed"] == (0 if stop == "error" else 1)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_parallel_bfloat16_backward(tiny):
    config = tiny[3]
    config.model.adaptive_layers = 0
    config.model.dtype = "bfloat16"
    model, _ = create_model(config.model)
    steps = [tiny[2].encode(Event("shell", "x" * (i + 1), "y" * (i % 4))) for i in range(8)]
    loss, records = batch_loss(model, steps, pad_id=256)
    assert loss.dtype == torch.float32
    loss.backward()
    assert len(records) == 8
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
