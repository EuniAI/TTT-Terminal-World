"""SFT samples are shell observations; controls only maintain boundaries."""
from terminal_world.dataloader.events import transition
from terminal_world.model.data import WindowBatch


def shell_batches(events, encoder, batch_size, *, start_open=False, history=()):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    steps, history, is_open = [], list(history), start_open
    next_cursor = None
    for event in events:
        is_open = transition(is_open, event)
        next_cursor = event.cursor + 1
        if event.kind == "shell":
            steps.append(encoder.encode(event, history))
        encoder.observe(history, event)
        if len(steps) == batch_size:
            yield WindowBatch(steps, next_cursor, is_open)
            steps = []
    if is_open:
        raise ValueError("SFT event stream ended inside a trajectory")
    if steps:
        yield WindowBatch(steps, next_cursor, is_open)
