from argparse import Namespace
import json

import pytest
import torch
import yaml

from terminal_world.checkpoint import load_checkpoint
from terminal_world.dataloader.trajectory import build_file_split_manifest
from terminal_world.train import run


def stage_manifest(tmp_path, name):
    paths = [tmp_path / f"{name}_{split}.jsonl" for split in ("training", "test")]
    for path, prefix in zip(paths, (name, name + "test")):
        path.write_text("\n".join(json.dumps({"trajectory": [
            {"command": prefix + str(i), "output": "a"}]}) for i in range(4)) + "\n")
    manifest = tmp_path / f"{name}.json"
    manifest.write_text(json.dumps(build_file_split_manifest(paths[:1], paths[1:])))
    return manifest


def arguments(output, manifest, *, config=None, resume=None, source=None, limit=None):
    return Namespace(output=str(output), manifest=str(manifest), config=str(config) if config else None,
                     resume=str(resume) if resume else None, start_stage_from=str(source) if source else None,
                     epochs=1, max_windows=limit, device="cpu")


def test_stage_preserves_state_restarts_schedule_and_resumes(tmp_path, tiny, monkeypatch, capsys):
    import terminal_world.train as trainer
    config = tiny[3]
    config.training.outer_lr_schedule = "warmup_cosine"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config.to_dict()))
    a, b = stage_manifest(tmp_path, "a"), stage_manifest(tmp_path, "b")
    first = tmp_path / "first.pt"
    run(arguments(first, a, config=cfg))
    source_bytes = first.read_bytes()
    source = torch.load(first, weights_only=False)
    assert source["windows_completed"] == 3
    capsys.readouterr()

    # Inspect the initialized second stage before any learning takes place.
    second = tmp_path / "second.pt"
    with monkeypatch.context() as m:
        m.setattr(trainer, "_train_windows", lambda *args, **kwargs: None)
        run(arguments(second, b, source=first))
    initialized = torch.load(second, weights_only=False)
    for field in ("theta", "fast_weights"):
        for key in source[field]:
            torch.testing.assert_close(initialized[field][key], source[field][key], atol=0, rtol=0)
    for index, old_state in source["optimizer"]["state"].items():
        for key, value in old_state.items():
            torch.testing.assert_close(initialized["optimizer"]["state"][index][key], value, atol=0, rtol=0)
    assert initialized["epoch"] == initialized["cursor"] == 0
    assert initialized["windows_completed"] == 3
    assert initialized["extra"]["outer_lr_plan"] == {
        "total_windows": 3, "warmup_windows": 1, "start_window": 3}
    assert initialized["extra"]["stage_history"][-1]["manifest_digest"] == source["manifest_digest"]
    capsys.readouterr()

    run(arguments(second, b, resume=second, limit=1))
    run(arguments(second, b, resume=second))
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["outer_lr"] for row in logs if "outer_lr" in row] == [
        0, config.training.outer_lr, config.training.outer_end_lr]
    full = tmp_path / "full-second.pt"
    run(arguments(full, b, source=first))
    full_data, resumed = (torch.load(p, weights_only=False) for p in (full, second))
    for field in ("theta", "fast_weights"):
        for key in full_data[field]:
            torch.testing.assert_close(full_data[field][key], resumed[field][key], atol=0, rtol=0)
    assert resumed["epoch"] == 1 and resumed["windows_completed"] == 6
    assert source_bytes == first.read_bytes()
    load_checkpoint(second)


def test_stage_transition_rejects_incomplete_or_accidental_data_switch(tmp_path, tiny):
    cfg = tmp_path / "config.yaml"
    tiny[3].training.outer_lr_schedule = "warmup_cosine"
    cfg.write_text(yaml.safe_dump(tiny[3].to_dict()))
    a, b = stage_manifest(tmp_path, "a"), stage_manifest(tmp_path, "b")
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    run(arguments(first, a, config=cfg, limit=1))
    with pytest.raises(ValueError, match="finish the source stage"):
        run(arguments(second, b, source=first))
    assert not second.exists()
    with pytest.raises(ValueError, match="manifest mismatch"):
        run(arguments(second, b, resume=first))
    run(arguments(first, a, resume=first))
    with pytest.raises(ValueError, match="different manifest"):
        run(arguments(second, a, source=first))
    with pytest.raises(FileExistsError):
        run(arguments(first, b, source=first))
    with pytest.raises(ValueError, match="mutually exclusive"):
        run(arguments(second, b, source=first, resume=first))
    manifest = json.loads(b.read_text())
    old = json.loads(a.read_text())
    # Use an actual source test file as the new training set.
    training = old["files"][1]["path"]
    test = manifest["files"][1]["path"]
    b.write_text(json.dumps(build_file_split_manifest([training], [test])))
    with pytest.raises(ValueError, match="overlap across stages"):
        run(arguments(second, b, source=first))
