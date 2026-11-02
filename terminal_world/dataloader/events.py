from terminal_world.model.data import Event, WindowBatch


def transition(is_open: bool, event: Event) -> bool:
    if event.kind == "shell":
        if not is_open:
            raise ValueError("shell event requires an open trajectory")
        return True
    if event.command == "open":
        if is_open:
            raise ValueError("nested open")
        return True
    if not is_open:
        raise ValueError("close without open")
    return False


def windows(events, encoder, unroll_events, *, start_open=False, history=()):
    if unroll_events < 2:
        raise ValueError("unroll_events must be >= 2")
    yield from supervised_batches(events, encoder, unroll_events, start_open=start_open, history=history)


def supervised_batches(events, encoder, batch_size=1, *, start_open=False, history=()):
    """Independent supervised samples; history survives optimizer steps."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    steps, is_open = [], start_open
    history = list(history)
    for event in events:
        is_open = transition(is_open, event)
        steps.append(encoder.encode(event, history))
        encoder.observe(history, event)
        if len(steps) == batch_size:
            yield WindowBatch(steps, event.cursor + 1, is_open)
            steps = []
    if steps:
        yield WindowBatch(steps, event.cursor + 1, is_open)
