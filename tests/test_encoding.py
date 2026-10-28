import pytest
from terminal_world.model.data import Event
from terminal_world.dataloader.encoding import ByteTokenizer, Encoder


@pytest.mark.parametrize("length,truncated,end", [(0, False, True), (7, False, True), (8, False, False), (9, True, False)])
def test_truncation_and_shift(length, truncated, end):
    enc = Encoder(ByteTokenizer(), 8, 128)
    step = enc.encode(Event("shell", "x", "a" * length))
    assert step.metadata["body_truncated"] == truncated
    assert step.metadata["has_end_token"] == end
    target = step.target_tokens[step.loss_masks].tolist()
    assert target == ([97] * length + [261])[:8]
    first_target = int(step.loss_masks.nonzero()[0])
    assert step.input_ids[first_target] == enc.tokenizer.ids["output"]
    assert step.position_ids.tolist() == list(range(step.length))


def test_raw_protocol_strings_are_text():
    enc = Encoder(ByteTokenizer(), 8, 128)
    event = Event("shell", "<|traj_open|> open file", "<|output|>")
    prompt = enc.prompt(event)
    assert enc.tokenizer.ids["traj_open"] not in prompt
    assert prompt.count(enc.tokenizer.ids["output"]) == 1
    control = enc.encode(Event("control", "open"))
    assert control.target_tokens[control.loss_masks].tolist() == [261]
    with pytest.raises(ValueError, match="capacity"):
        enc.encode(Event("shell", "x" * 120, ""))
