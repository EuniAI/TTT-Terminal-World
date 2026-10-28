from argparse import Namespace
import json
import torch
import yaml
from terminal_world.train import run
from terminal_world.checkpoint import load_checkpoint
from terminal_world.dataloader.trajectory import build_manifest


def test_cli_resume_across_epochs(tmp_path, tiny, capsys):
    tiny[3].training.unroll_events = 5
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
    run(args(resumed, max_windows=1))
    run(args(resumed, resume=resumed))
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
