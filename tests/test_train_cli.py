from argparse import Namespace
import json
import torch
import yaml
import pytest
from terminal_world.train import run
from terminal_world.checkpoint import load_checkpoint
from terminal_world.dataloader.trajectory import build_manifest


@pytest.mark.parametrize("schedule", ["constant", "warmup_cosine"])
@pytest.mark.parametrize("pause_after", [1, 3])
def test_cli_resume_across_epochs(tmp_path, tiny, capsys, schedule, pause_after):
    tiny[3].training.unroll_events = 5
    tiny[3].training.outer_lr_schedule = schedule
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(tiny[3].to_dict()))
    data = tmp_path / "data.jsonl"
    data.write_text('\n'.join(json.dumps({"trajectory": [{"command": x, "output": "a"}]}) for x in ("x", "y")) + '\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest([data], 0)))
    full, resumed = tmp_path / "full.pt", tmp_path / "resumed.pt"
    def args(output, resume=None, max_windows=None):
        return Namespace(config=str(config) if not resume else None, manifest=str(manifest), output=str(output),
                         resume=str(resume) if resume else None, device="cpu", epochs=2, max_windows=max_windows)
    run(args(full))
    full_logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    run(args(resumed, max_windows=pause_after))
    run(args(resumed, resume=resumed))
    resumed_logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    rates = lambda logs: [item["outer_lr"] for item in logs if "outer_lr" in item]
    assert rates(full_logs) == rates(resumed_logs)
    if schedule == "warmup_cosine":
        assert rates(full_logs)[0] == 0
        assert rates(full_logs)[-1] == tiny[3].training.outer_end_lr
    m1, s1, _, _, _, d1 = load_checkpoint(full)
    m2, s2, _, _, _, d2 = load_checkpoint(resumed)
    assert s1.epoch == s2.epoch == 2
    assert s1.next_cursor == s2.next_cursor == 0
    for a, b in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for k in s1.fast_weights:
        torch.testing.assert_close(s1.fast_weights[k], s2.fast_weights[k], atol=0, rtol=0)
    assert d1["extra"] == d2["extra"]
    assert d1["extra"]["boundary_coverage"]["covered"] > 0
    if schedule == "warmup_cosine":
        assert d1["extra"]["outer_lr_plan"] == {"total_windows": 4, "warmup_windows": 1}
        # Extending epochs preserves the original horizon and stays at the floor.
        extended = args(resumed, resume=resumed, max_windows=1)
        extended.epochs = 3
        run(extended)
        logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert rates(logs) == [tiny[3].training.outer_end_lr]


@pytest.mark.parametrize("limit", [None, 51])
def test_checkpoint_every_50_and_final_save(tmp_path, tiny, monkeypatch, limit):
    import terminal_world.train as training
    tiny[3].training.unroll_events = 2
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(tiny[3].to_dict()))
    data = tmp_path / "data.jsonl"
    data.write_text('\n'.join(json.dumps({"trajectory": [{"command": str(i), "output": "a"}]})
                              for i in range(34)) + '\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest([data], 0)))
    output = tmp_path / "run.pt"
    saves = []
    original = training.save_checkpoint
    def save(path, model, state, *args, **kwargs):
        saves.append((state.windows_completed, state.epoch))
        return original(path, model, state, *args, **kwargs)
    monkeypatch.setattr(training, "save_checkpoint", save)
    run(Namespace(config=str(cfg), manifest=str(manifest), output=str(output), resume=None,
                  device="cpu", epochs=1, max_windows=limit))
    assert saves == [(0, 0), (50, 0), (51, 1 if limit is None else 0)]
    saved = torch.load(output, weights_only=False)
    assert saved["windows_completed"] == 51


@pytest.mark.parametrize("stop", ["SIGTERM", "SIGINT", "error"])
def test_stop_saves_only_completed_safe_windows(tmp_path, tiny, monkeypatch, stop):
    import signal
    import terminal_world.train as training
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(tiny[3].to_dict()))
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"trajectory": [{"command": "x", "output": "a"}]}) + '\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest([data], 0)))
    output = tmp_path / "run.pt"
    original = training.train_window
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def train(*args, **kwargs):
        result = original(*args, **kwargs)
        if stop == "error":
            raise RuntimeError("unsafe update")
        signal.raise_signal(getattr(signal, stop))
        return result
    monkeypatch.setattr(training, "train_window", train)
    args = Namespace(config=str(cfg), manifest=str(manifest), output=str(output), resume=None,
                     device="cpu", epochs=1, max_windows=None)
    if stop == "error":
        with pytest.raises(RuntimeError, match="unsafe update"):
            run(args)
    else:
        run(args)
    saved = torch.load(output, weights_only=False)
    assert saved["windows_completed"] == (0 if stop == "error" else 1)
    assert all(signal.getsignal(s) == handler for s, handler in handlers.items())
