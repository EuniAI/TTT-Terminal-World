"""python -m terminal_world.train --config ... --manifest ... --output ..."""
import argparse
import json
from pathlib import Path
import random
import time
import torch
from terminal_world.checkpoint import load_checkpoint, save_checkpoint
from terminal_world.config import RunConfig
from terminal_world.dataloader.encoding import ByteTokenizer, Encoder
from terminal_world.dataloader.events import windows
from terminal_world.dataloader.trajectory import EventStream
from terminal_world.metrics import summarize
from terminal_world.model.loop import TrainingState, train_window
from terminal_world.model.transformer import create_model
from terminal_world.tracking import start_tracker


def run(args):
    if args.epochs < 1 or (args.max_windows is not None and args.max_windows < 1):
        raise ValueError("epochs and max-windows must be positive")
    output = Path(args.output)
    if not args.resume and output.exists():
        raise FileExistsError("output checkpoint already exists; use --resume or a new output")
    manifest = json.loads(Path(args.manifest).read_text())
    stream = EventStream(manifest)
    if not stream.total_events:
        raise ValueError("training split is empty")
    previous = {}
    if args.resume:
        model, state, optimizer, config, encoder, previous = load_checkpoint(
            args.resume, device=args.device, expected_manifest_digest=stream.digest)
        if args.config and RunConfig.load(args.config).to_dict() != config.to_dict():
            raise ValueError("resume config mismatch; use the checkpoint's configuration")
        coverage = previous.get("extra", {}).get("boundary_coverage", {"triples": 0, "covered": 0, "tail": []})
    else:
        if not args.config:
            raise ValueError("fresh training requires --config")
        config = RunConfig.load(args.config)
        torch.manual_seed(config.training.seed)
        random.seed(config.training.seed)
        if config.model.model_id == "tiny-random":
            tokenizer = ByteTokenizer()
            if config.model.vocab_size != tokenizer.vocab_size:
                raise ValueError("tiny model requires byte tokenizer vocabulary")
            model, fast = create_model(config.model, args.device)
        else:
            from terminal_world.model.pretrained import load_pretrained
            model, fast, tokenizer, config.model = load_pretrained(config.model, device=args.device)
        state = TrainingState(fast)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.outer_lr,
                                     weight_decay=config.training.outer_weight_decay)
        encoder = Encoder(tokenizer, config.training.max_output_tokens, config.model.max_seq_len)
        coverage = {"triples": 0, "covered": 0, "tail": []}
    if stream.state_at(state.next_cursor) != state.is_open:
        raise ValueError("checkpoint cursor/protocol state mismatch")
    model_info = {"theta_parameters": sum(p.numel() for p in model.parameters()),
                  "fast_parameters": sum(p.numel() for p in state.fast_weights.values()),
                  "backend": config.model.attention_backend, "device": args.device}
    print(json.dumps(model_info), flush=True)
    extra = dict(previous.get("extra", {}))
    with start_tracker(args, config, state, model_info, stream.digest, extra.get("wandb")) as tracker:
        if tracker.identity is not None:
            extra["wandb"] = tracker.identity
        def commit():
            extra["boundary_coverage"] = coverage
            save_checkpoint(output, model, state, optimizer, config, encoder.tokenizer,
                            manifest, stream.digest, extra=extra)
        # Save identity before the first window, including when enabling logging on an old run.
        commit()
        _train_windows(args, model, state, optimizer, config, encoder, stream, coverage, commit, tracker)


def _train_windows(args, model, state, optimizer, config, encoder, stream, coverage, commit, tracker):
    completed = 0
    while state.epoch < args.epochs:
        for window in windows(stream.from_cursor(state.next_cursor), encoder, config.training.unroll_events,
                              start_open=state.is_open):
            cuda_timing = tracker.enabled and next(model.parameters()).device.type == "cuda"
            if cuda_timing:
                torch.cuda.synchronize(next(model.parameters()).device)
                torch.cuda.reset_peak_memory_stats(next(model.parameters()).device)
            started = time.perf_counter()
            loss, records = train_window(model, optimizer, state, window, config.training, pad_id=encoder.tokenizer.ids["pad"])
            if cuda_timing:
                torch.cuda.synchronize(next(model.parameters()).device)
            compute_seconds = time.perf_counter() - started
            tags = [(r.get("command") if r["kind"] == "control" else "shell", state.windows_completed) for r in records]
            joined = coverage["tail"] + tags
            for i in range(len(joined) - 2):
                if [t[0] for t in joined[i:i + 3]] == ["close", "open", "shell"]:
                    coverage["triples"] += 1
                    coverage["covered"] += int(len({t[1] for t in joined[i:i + 3]}) == 1)
            coverage["tail"] = joined[-2:]
            # Checkpoint every committed window; no shell command is ever executed here.
            checkpoint_started = time.perf_counter()
            commit()
            checkpoint_seconds = time.perf_counter() - checkpoint_started
            metrics = summarize(records)
            metrics.update(loss=loss, epoch=state.epoch, cursor=state.next_cursor,
                           windows=state.windows_completed, boundary_coverage=coverage)
            if tracker.enabled:
                metrics["performance"] = {
                    "window_compute_seconds": compute_seconds,
                    "checkpoint_seconds": checkpoint_seconds,
                    "target_tokens_per_second": sum(r["tokens"] for r in records) / compute_seconds,
                }
                metrics["inner_lr"] = config.training.inner_lr
                metrics["outer_lr"] = optimizer.param_groups[0]["lr"]
                triples = coverage["triples"]
                metrics["boundary_coverage"] = dict(coverage)
                metrics["boundary_coverage"]["rate"] = coverage["covered"] / triples if triples else None
                if cuda_timing:
                    metrics["performance"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(next(model.parameters()).device)
                # Log only committed windows; an SDK failure leaves the checkpoint recoverable.
                tracker.log(metrics)
            print(json.dumps(metrics), flush=True)
            completed += 1
            if args.max_windows is not None and completed >= args.max_windows:
                return
        if state.is_open:
            raise ValueError("epoch ended inside a trajectory")
        state.epoch += 1
        state.next_cursor = 0
        commit()
    if coverage["covered"] == 0:
        print(json.dumps({"notice": "No close/open/shell triple fell in a single window; cross-trajectory meta-gradient coverage is unverified."}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Persistent-fast-weight terminal meta-training")
    parser.add_argument("--config")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1, help="total target epochs, including completed epochs")
    parser.add_argument("--max-windows", type=int, help="stop after this many additional windows")
    parser.add_argument("--wandb-project", help="enable online W&B tracking in this project")
    parser.add_argument("--wandb-entity", help="W&B user or team")
    parser.add_argument("--wandb-name", help="display name of the W&B run")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        help="default: online with --wandb-project, otherwise disabled")
    parser.add_argument("--wandb-dir", help="local SDK files; defaults to checkpoint directory")
    parser.add_argument("--wandb-new-run", action="store_true", help="fork logging from a saved run identity")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
