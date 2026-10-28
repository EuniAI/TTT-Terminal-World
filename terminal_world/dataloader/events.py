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


def windows(events, encoder, unroll_events, *, start_open=False):
    if unroll_events < 2:
        raise ValueError("unroll_events must be >= 2")
    steps, is_open = [], start_open
    for event in events:
        is_open = transition(is_open, event)
        steps.append(encoder.encode(event))
        if len(steps) == unroll_events:
            yield WindowBatch(steps, event.cursor + 1, is_open)
            steps = []
    if steps:
        yield WindowBatch(steps, event.cursor + 1, is_open)
