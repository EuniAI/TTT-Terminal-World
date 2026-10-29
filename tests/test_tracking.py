from argparse import Namespace
import copy
import json
import random
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from terminal_world.dataloader.trajectory import build_manifest
from terminal_world.train import run
from terminal_world.tracking import scalar_metrics, start_tracker
from terminal_world.model.loop import TrainingState


@pytest.fixture
def sdk(monkeypatch):
    calls = []
    def init(**kwargs):
        random.random()
        torch.rand(1)
        item = SimpleNamespace(id=kwargs["id"], project=kwargs["project"],
                               entity=kwargs["entity"] or "test-user", summary={},
                               logs=[], definitions=[], finishes=[], kwargs=kwargs)
        item.define_metric = lambda *a, **k: item.definitions.append((a, k))
        item.log = lambda data: item.logs.append(copy.deepcopy(data))
        item.finish = lambda **k: item.finishes.append(k)
        calls.append(item)
        return item
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **k: k))
    return calls


def test_scalar_metrics_exclude_text_lists_and_nonfinite():
    assert scalar_metrics({"loss": 2., "shell": {"nll": None, "events": 2},
                           "raw": "secret output", "tail": [["close", 1]], "bad": float("nan")}) == {
        "train/loss": 2., "train/shell/events": 2,
    }


def test_wandb_training_resume_preserves_model_and_committed_steps(tmp_path, tiny, sdk):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(tiny[3].to_dict()))
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"trajectory": [{"command": "x", "output": "a"},
                                               {"command": "y", "output": "b"}]}) + '\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest([data], 0)))
    def args(path, resume=None, mode="online", limit=None):
        return Namespace(config=None if resume else str(cfg), manifest=str(manifest), output=str(path),
                         resume=str(resume) if resume else None, device="cpu", epochs=2,
                         max_windows=limit, wandb_mode=mode, wandb_project="terminal-test")
    baseline, tracked = tmp_path / "baseline.pt", tmp_path / "tracked.pt"
    run(args(baseline, mode="disabled"))
    run(args(tracked, limit=1))
    saved = torch.load(tracked, weights_only=False)
    identity = saved["extra"]["wandb"]
    assert saved["windows_completed"] == 1
    run(args(tracked, resume=tracked))
    assert sdk[0].id == sdk[1].id == identity["id"]
    assert sdk[1].kwargs["resume"] == "must"
    assert sdk[1].kwargs["entity"] == "test-user"
    assert [r["train/windows"] for call in sdk for r in call.logs] == [1, 2]
    assert all(call.finishes == [{"exit_code": 0}] for call in sdk)
    assert "manifest" not in sdk[0].kwargs["config"]
    assert "manifest_digest" in sdk[0].kwargs["config"]
    logged = sdk[0].logs[0]
    assert "train/shell/nll" in logged
    assert "train/boundary_coverage/tail" not in logged
    assert logged["train/performance/window_compute_seconds"] > 0
    a, b = (torch.load(p, weights_only=False) for p in (baseline, tracked))
    for field in ("theta", "fast_weights"):
        for key in a[field]:
            torch.testing.assert_close(a[field][key], b[field][key], atol=0, rtol=0)
    assert a["rng"]["python"] == b["rng"]["python"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])


def test_tracking_modes_and_failure_cleanup(tmp_path, tiny, sdk):
    model, fast, _, config = tiny
    args = Namespace(output=str(tmp_path / "run.pt"), wandb_mode="offline", wandb_project="p")
    with start_tracker(args, config, TrainingState(fast), {}, "digest") as tracker:
        first = tracker.identity
    with pytest.raises(RuntimeError, match="training failed"):
        with start_tracker(args, config, TrainingState(fast), {}, "digest", first) as tracker:
            assert tracker.identity["id"] != first["id"]
            assert sdk[-1].kwargs["resume"] is None
            assert sdk[-1].kwargs["config"]["parent_run_id"] == first["id"]
            raise RuntimeError("training failed")
    assert sdk[-1].finishes == [{"exit_code": 1}]
    args.wandb_project = "other"
    with pytest.raises(ValueError, match="project/entity"):
        start_tracker(args, config, TrainingState(fast), {}, "digest", first)
    args.wandb_mode = "disabled"
    with start_tracker(args, config, TrainingState(fast), {}, "digest", first) as tracker:
        assert not tracker.enabled and tracker.identity == first


def test_log_failure_leaves_committed_checkpoint(tmp_path, tiny, sdk, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(tiny[3].to_dict()))
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"trajectory": [{"command": "x", "output": "a"}]}) + '\n')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest([data], 0)))
    output = tmp_path / "run.pt"
    from terminal_world.tracking import Tracker
    def fail(self, metrics):
        saved = torch.load(output, weights_only=False)
        assert saved["windows_completed"] == 0
        assert metrics["windows"] == 1
        raise RuntimeError("logging failed")
    monkeypatch.setattr(Tracker, "log", fail)
    with pytest.raises(RuntimeError, match="logging failed"):
        run(Namespace(config=str(cfg), manifest=str(manifest), output=str(output), resume=None,
                      device="cpu", epochs=1, max_windows=None, wandb_project="p"))
    assert sdk[-1].finishes == [{"exit_code": 1}]
    assert torch.load(output, weights_only=False)["windows_completed"] == 1
