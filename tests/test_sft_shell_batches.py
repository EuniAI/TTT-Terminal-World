import json
from argparse import Namespace

import pytest
import torch
import yaml

from terminal_world.dataloader.encoding import ByteTokenizer, Encoder
from terminal_world.dataloader.sft import shell_batches
from terminal_world.dataloader.trajectory import EventStream, build_manifest
from terminal_world.sft_train import run


@pytest.mark.parametrize("counts", [[0, 4, 0, 4, 0], [3, 6], [0, 0]])
def test_controls_only_clear_history_and_never_enter_batch(tmp_path, counts):
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"trajectory": [
        {"command": f"{traj}-{i}", "output": "a"} for i in range(count)
    ]}) for traj, count in enumerate(counts)) + "\n")
    stream = EventStream(build_manifest([data], 0))
    encoder = Encoder(ByteTokenizer(), 8, 128, trajectory_context=True)
    original = encoder.encode
    calls = []
    def encode(event, history=()):
        assert event.kind == "shell"
        if event.command.endswith("-0"):
            assert not history
        calls.append(event.cursor)
        return original(event, history)
    encoder.encode = encode
    batches = list(shell_batches(stream.from_cursor(), encoder, 8))
    assert len(calls) == sum(counts)
    assert sum(len(batch.steps) for batch in batches) == sum(counts)
    assert all(step.kind == "shell" for batch in batches for step in batch.steps)
    if batches:
        first = batches[0]
        tail = list(shell_batches(stream.from_cursor(first.next_cursor), encoder, 8,
                                  start_open=first.is_open,
                                  history=stream.history_before(first.next_cursor, encoder)))
        assert len(tail) == len(batches) - 1


def test_exact_full_batch_trailing_controls_finish_epoch(tmp_path, tiny, capsys):
    config = tiny[3]
    config.model.adaptive_layers = 0
    config.training.trajectory_context = True
    config.training.sft_batch_size = 8
    config.training.outer_lr_schedule = "warmup_cosine"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config.to_dict()))
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"trajectory": [
        {"command": str(i), "output": "a"} for i in range(n)
    ]}) for n in [4, 4, 0]) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest([data], 0)))
    output = tmp_path / "out.pt"
    run(Namespace(config=str(cfg), manifest=str(manifest), output=str(output),
                  resume=None, device="cpu", epochs=3, max_steps=None, save_every_steps=50))
    payload = torch.load(output, weights_only=False)
    assert payload["epoch"] == 3 and not payload["is_open"]
    assert payload["windows_completed"] == 3
    assert payload["extra"]["outer_lr_plan"]["total_windows"] == 3
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    for log in (item for item in logs if "loss" in item):
        assert log["batch_size"] == log["shell"]["events"] == 8
        assert log["control"]["events"] == 0
    # A preserved old control-inclusive schedule must not block a completed
    # shell-only stage from advancing to the next dataset.
    payload["extra"]["outer_lr_plan"]["total_windows"] = 6
    torch.save(payload, output)
    next_data = tmp_path / "next.jsonl"
    next_data.write_text(json.dumps({"trajectory": [
        {"command": "next-" + str(i), "output": "b"} for i in range(8)
    ]}) + "\n")
    next_manifest = tmp_path / "next-manifest.json"
    next_manifest.write_text(json.dumps(build_manifest([next_data], 0)))
    next_output = tmp_path / "next.pt"
    run(Namespace(config=None, manifest=str(next_manifest), output=str(next_output),
                  resume=None, start_stage_from=str(output), device="cpu", epochs=3,
                  max_steps=1, save_every_steps=50))
    saved = torch.load(next_output, weights_only=False)
    assert saved["extra"]["stage_history"][-1]["outer_lr_plan"]["total_windows"] == 6
    assert not saved["extra"]["sft_stage_complete"]
