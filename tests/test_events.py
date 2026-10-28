import json
import pytest
from terminal_world.dataloader.trajectory import build_manifest, EventStream
from terminal_world.dataloader.events import transition, windows
from terminal_world.model.data import Event


def test_order_resume_and_split(tmp_path, tiny):
    path = tmp_path / "data.jsonl"
    a = {"trajectory": [{"command": "x", "output": ""}, {"command": "y", "output": "z\n"}]}
    b = {"trajectory": [{"command": "q", "output": "err"}]}
    path.write_text("\n".join(json.dumps(x) for x in [a, b, a]) + "\n")
    original = path.read_bytes()
    manifest = build_manifest([path], validation_fraction=0)
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
    split = build_manifest([path], validation_fraction=0.5)
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
