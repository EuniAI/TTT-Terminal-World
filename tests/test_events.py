import json
import pytest
from terminal_world.dataloader.trajectory import build_manifest, build_file_split_manifest, EventStream
from terminal_world.dataloader.events import transition, windows
from terminal_world.model.data import Event


def test_order_resume_and_split(tmp_path, tiny):
    path = tmp_path / "data.jsonl"
    a = {"trajectory": [{"command": "x", "output": ""}, {"command": "y", "output": "z\n"}]}
    b = {"trajectory": [{"command": "q", "output": "err"}]}
    path.write_text("\n".join(json.dumps(x) for x in [a, b, a]) + "\n")
    original = path.read_bytes()
    manifest = build_manifest([path], test_fraction=0)
    stream = EventStream(manifest)
    events = list(stream.from_cursor())
    assert [e.command for e in events] == ["open", "x", "y", "close", "open", "q", "close", "open", "x", "y", "close"]
    for cursor in range(len(events) + 1):
        assert list(stream.from_cursor(cursor)) == events[cursor:]
        state = False
        for event in events[:cursor]:
            state = transition(state, event)
        assert stream.state_at(cursor) == state
        resumed = list(windows(stream.from_cursor(cursor), tiny[2], 2, start_open=state))
        assert sum(len(w.steps) for w in resumed) == len(events) - cursor
    split = build_manifest([path], test_fraction=0.5)
    assert split["entries"][0]["split"] == split["entries"][2]["split"]
    assert path.read_bytes() == original
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        EventStream(manifest)


def test_protocol_errors():
    with pytest.raises(ValueError):
        transition(False, Event("shell", "ls"))
    with pytest.raises(ValueError):
        transition(True, Event("control", "open"))
    with pytest.raises(ValueError):
        transition(False, Event("control", "close"))


def test_explicit_splits_and_legacy_digest(tmp_path):
    import copy
    import hashlib
    paths = [tmp_path / name for name in ("training.jsonl", "test.jsonl")]
    for path, command in zip(paths, ("x", "y")):
        path.write_text(json.dumps({"trajectory": [{"command": command, "output": "a"}]}) + "\n")
    manifest = build_file_split_manifest(paths[:1], paths[1:])
    assert [e["split"] for e in manifest["entries"]] == ["training", "test"]
    assert [e.command for e in EventStream(manifest).from_cursor()] == ["open", "x", "close"]
    assert [e.command for e in EventStream(manifest, "test").from_cursor()] == ["open", "y", "close"]
    legacy = copy.deepcopy(manifest)
    for entry in legacy["entries"]:
        entry["split"] = {"training": "train", "test": "validation"}[entry["split"]]
    legacy["validation_fraction"] = legacy.pop("test_fraction")
    before = json.dumps(legacy, sort_keys=True)
    for canonical, old in (("training", "train"), ("test", "validation")):
        a, b = EventStream(legacy, canonical), EventStream(legacy, old)
        assert list(a.from_cursor()) == list(b.from_cursor())
        assert a.digest == hashlib.sha256(before.encode()).hexdigest()
    assert json.dumps(legacy, sort_keys=True) == before
    with pytest.raises(ValueError, match="unsupported split"):
        EventStream(manifest, "typo")
    with pytest.raises(ValueError, match="only once"):
        build_file_split_manifest(paths[:1], paths[:1])
    paths[1].write_bytes(paths[0].read_bytes())
    with pytest.raises(ValueError, match="identical trajectories"):
        build_file_split_manifest(paths[:1], paths[1:])
    paths[1].write_text("")
    with pytest.raises(ValueError, match="both contain"):
        build_file_split_manifest(paths[:1], paths[1:])


def test_manifest_cli_explicit_files(tmp_path, monkeypatch):
    from terminal_world.dataloader.trajectory import main
    paths = [tmp_path / name for name in ("a.jsonl", "b.jsonl")]
    for i, path in enumerate(paths):
        path.write_text(json.dumps({"trajectory": [{"command": str(i), "output": ""}]}) + "\n")
    output = tmp_path / "manifest.json"
    argv = ["manifest", "--training-files", str(paths[0]), "--test-files", str(paths[1]),
            "--output", str(output)]
    monkeypatch.setattr("sys.argv", argv)
    main()
    manifest = json.loads(output.read_text())
    assert manifest["split_strategy"] == "preassigned_files"
    assert len(EventStream(manifest, "training").entries) == 1
    assert len(EventStream(manifest, "test").entries) == 1
    with pytest.raises(FileExistsError):
        main()
    monkeypatch.setattr("sys.argv", argv + ["--test-fraction", "0.1"])
    with pytest.raises(SystemExit):
        main()
