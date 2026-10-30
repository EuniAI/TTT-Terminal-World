"""Read-only checkpoint evaluation. Adaptation is confined to an in-memory copy."""
import argparse
import copy
import json
import sys
from pathlib import Path
from tqdm import tqdm
from terminal_world.checkpoint import load_checkpoint
from terminal_world.dataloader.trajectory import EventStream
from terminal_world.metrics import summarize
from terminal_world.model.data import Event
from terminal_world.session import Session


def evaluate(model, fast, encoder, events, *, alpha, clip_norm=None, adapt=True, is_open=False, generate=False):
    # Never freeze or mutate the caller's training model/fast weights.
    session = Session(copy.deepcopy(model), {k: v.detach().clone().requires_grad_(True) for k, v in fast.items()},
                      encoder, alpha=alpha, clip_norm=clip_norm, is_open=is_open)
    records = []
    for event in events:
        generated = session.generate_tokens(event.command) if generate and event.kind == "shell" else None
        record = session.apply_event(event, adapt=adapt)
        if generated is not None:
            # Real target is consulted only AFTER generation, never to choose generation length.
            body = encoder.tokenizer.encode_text(event.output)
            end = encoder.tokenizer.ids["obs_end"]
            kept = body[:encoder.max_output_tokens]
            generated_body = generated[:-1] if generated and generated[-1] == end else generated
            if record["has_end_token"]:
                record["body_match"] = generated_body == kept
                record["end_correct"] = generated == body + [end]
            else:
                record["body_match"] = generated[:len(kept)] == kept
                record["end_correct"] = None  # No end label in this target.
            record["generated_tokens"] = len(generated)
        records.append(record)
    return summarize(records), session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", help="defaults to the checkpoint manifest")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--generate", action="store_true", help="also score greedy output matching; slower")
    parser.add_argument("--no-progress", action="store_true", help="disable progress and status messages on stderr")
    parser.add_argument("--exposure-label", required=True, help="record whether evaluation data has been seen before")
    args = parser.parse_args()
    def status(message):
        if not args.no_progress:
            print(message, file=sys.stderr, flush=True)

    status(f"Loading checkpoint: {args.checkpoint}")
    model, state, _, config, encoder, data = load_checkpoint(args.checkpoint, device=args.device,
                                                            training=False, restore_rng=False)
    status("Verifying test data files...")
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else data["manifest"]
    stream = EventStream(manifest, "test")
    if not stream.total_events:
        raise ValueError("test split is empty")
    # The checkpoint may be mid-training-trajectory. End that context explicitly in the copy.
    events = stream.from_cursor()
    if state.is_open:
        from itertools import chain
        events = chain([Event("control", "close")], events)
    total = stream.total_events + int(state.is_open)
    status(f"Test data: {len(stream.entries)} trajectories, {total} events (including control events).")
    status("Preparing evaluation model copy...")
    with tqdm(events, total=total, desc="Eval frozen" if args.frozen else "Eval adapt",
              unit="event", file=sys.stderr, dynamic_ncols=True, disable=args.no_progress) as progress:
        result, _ = evaluate(model, state.fast_weights, encoder, progress, alpha=config.training.inner_lr,
                             clip_norm=config.training.inner_clip_norm, adapt=not args.frozen,
                             is_open=state.is_open, generate=args.generate)
    result.update(exposure_label=args.exposure_label, adaptation=not args.frozen,
                  source_cursor=state.next_cursor, test_manifest=stream.digest)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
