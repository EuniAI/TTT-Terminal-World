"""Real two-process CPU DDP regression; CUDA/NCCL still needs GPU acceptance."""
from argparse import Namespace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch
import yaml

from terminal_world.config import ModelConfig, RunConfig, TrainConfig
from terminal_world.dataloader.trajectory import build_manifest
from terminal_world.sft_train import run


@pytest.mark.parametrize("origin", ["ddp", "single"])
def test_two_rank_training_matches_single_rank_and_resumes(tmp_path, origin):
    # Restricted test sandboxes may disallow even loopback sockets.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("local sockets forbidden; run DDP test with loopback permission")
    config = RunConfig(
        ModelConfig(hidden_size=8, intermediate_size=12, num_heads=2, num_kv_heads=1,
                    adaptive_layers=0, max_seq_len=128, dtype="float64", qkv_bias=True),
        TrainConfig(.1, .001, 4, 1, max_output_tokens=8,
                    trajectory_context=True, sft_batch_size=4, outer_lr_schedule="warmup_cosine"))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config.to_dict()))
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"trajectory": [
        {"command": str(i), "output": "x" * (i + 1)} for i in range(5)
    ]}) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(build_manifest([data], 0)))
    single, parallel = tmp_path / "single.pt", tmp_path / "parallel.pt"
    run(Namespace(config=str(cfg), manifest=str(manifest), output=str(single),
                  resume=None, device="cpu", epochs=2, max_steps=None, save_every_steps=50))
    # A local fake SDK checks rank ownership and resume identity without network.
    sdk_dir = tmp_path / "sdk"
    sdk_dir.mkdir()
    sdk_log = tmp_path / "wandb-calls.jsonl"
    (sdk_dir / "wandb.py").write_text('''
import json, os
from types import SimpleNamespace
def Settings(**kwargs): return kwargs
def init(**kwargs):
    with open(os.environ["SFT_TEST_SDK_LOG"], "a") as f:
        f.write(json.dumps({"rank": os.environ["RANK"], "id": kwargs["id"],
                            "resume": kwargs["resume"]}) + "\\n")
    return SimpleNamespace(id=kwargs["id"], project=kwargs["project"], entity="test",
                           summary={}, define_metric=lambda *a, **k: None,
                           log=lambda *a, **k: None, finish=lambda **k: None)
''')

    def launch(*source):
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   "-m", "terminal_world.sft_train", "--manifest", str(manifest),
                   "--output", str(parallel), "--device", "cpu", "--epochs", "2",
                   "--wandb-project", "test-ddp", *source]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "SFT_TEST_SDK_LOG": str(sdk_log),
               "PYTHONPATH": str(sdk_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env,
                                text=True, capture_output=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    if origin == "ddp":
        first = launch("--config", str(cfg), "--max-steps", "1")
    else:
        run(Namespace(config=str(cfg), manifest=str(manifest), output=str(parallel),
                      resume=None, device="cpu", epochs=2, max_steps=3, save_every_steps=50))
        first = []
    saved = torch.load(parallel, weights_only=False)
    assert saved["cursor"] == 5 and saved["is_open"]
    assert len(saved["extra"]["sft_distributed_rng"]) == (2 if origin == "ddp" else 1)
    rest = launch("--resume", str(parallel))
    updates = [r for r in first + rest if "loss" in r]
    assert [r["steps"] for r in updates] == ([1, 2, 3, 4] if origin == "ddp" else [4])
    assert [r["shell"]["events"] for r in updates] == ([4, 1, 4, 1] if origin == "ddp" else [1])
    a, b = (torch.load(path, weights_only=False) for path in (single, parallel))
    assert a["epoch"] == b["epoch"] == 2
    assert a["windows_completed"] == b["windows_completed"] == 4
    calls = [json.loads(line) for line in sdk_log.read_text().splitlines()]
    assert [call["rank"] for call in calls] == (["0", "0"] if origin == "ddp" else ["0"])
    if origin == "ddp":
        assert calls[0]["id"] == calls[1]["id"]
        assert calls[1]["resume"] == "must"
    for name in a["theta"]:
        torch.testing.assert_close(a["theta"][name], b["theta"][name], atol=1e-10, rtol=1e-10)
    for key in a["optimizer"]["state"]:
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(a["optimizer"]["state"][key][name],
                                       b["optimizer"]["state"][key][name], atol=1e-10, rtol=1e-10)
