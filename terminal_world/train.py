"""python -m terminal_world.train --config ... --manifest ... --output ..."""
import argparse
from contextlib import contextmanager
import signal
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
from terminal_world.schedule import make_schedule, outer_learning_rate


def start_new_stage(state, previous, stream):
    """Explicit data transition at a completed stage boundary; preserve weights/AdamW."""
    old_stream = EventStream(previous["manifest"])
    if old_stream.digest != previous["manifest_digest"]:
        raise ValueError("source checkpoint data manifest mismatch")
    if old_stream.digest == stream.digest:
        raise ValueError("new stage requires a different manifest; use --resume for the same stage")
    at_epoch_end = (state.next_cursor == old_stream.total_events or
                    (state.epoch > 0 and state.next_cursor == 0))
    if not old_stream.total_events or state.is_open or not at_epoch_end or state.windows_completed == 0:
        raise ValueError("finish the source stage before starting a new stage")
    old_plan = previous.get("extra", {}).get("outer_lr_plan")
    if old_plan and state.windows_completed < old_plan.get("start_window", 0) + old_plan["total_windows"]:
        raise ValueError("finish the source stage learning-rate plan before starting a new stage")
    old_test = EventStream(previous["manifest"], "test", verify=False)
    new_test = EventStream(stream.manifest, "test", verify=False)
    hashes = lambda entries: {entry["content_hash"] for entry in entries}
    if (hashes(stream.entries) & hashes(old_test.entries) or
            hashes(old_stream.entries) & hashes(new_test.entries)):
        raise ValueError("training/test trajectory overlap across stages")
    source = {"manifest_digest": old_stream.digest, "epoch": state.epoch,
              "cursor": state.next_cursor, "windows_completed": state.windows_completed,
              "outer_lr_plan": old_plan, "wandb": previous.get("extra", {}).get("wandb")}
    state.epoch = 0
    state.next_cursor = 0
    return source


@contextmanager
def _stop_requests():
    # Finish the current window before saving: never persist a partial optimizer update.
    requested = []
    def request_stop(signum, frame):
        requested.append(signum)
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield requested
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run(args):
    if args.epochs < 1 or (args.max_windows is not None and args.max_windows < 1):
        raise ValueError("epochs and max-windows must be positive")
    if getattr(args, "save_every_windows", 50) < 1:
        raise ValueError("save-every-windows must be positive")
    output = Path(args.output)
    stage_from = getattr(args, "start_stage_from", None)
    if stage_from and args.resume:
        raise ValueError("--resume and --start-stage-from are mutually exclusive")
    if not args.resume and output.exists():
        raise FileExistsError("output checkpoint already exists; use --resume or a new output")
    manifest = json.loads(Path(args.manifest).read_text())
    stream = EventStream(manifest)
    if not stream.total_events:
        raise ValueError("training split is empty")
    previous = {}
    stage_source = None
    if args.resume or stage_from:
        model, state, optimizer, config, encoder, previous = load_checkpoint(
            args.resume or stage_from, device=args.device,
            expected_manifest_digest=stream.digest if args.resume else None)
        if args.config and RunConfig.load(args.config).to_dict() != config.to_dict():
            raise ValueError("resume config mismatch; use the checkpoint's configuration")
        coverage = previous.get("extra", {}).get("boundary_coverage", {"triples": 0, "covered": 0, "tail": []})
        if stage_from:
            stage_source = start_new_stage(state, previous, stream)
            stage_source["checkpoint"] = str(Path(stage_from).resolve())
            coverage = {"triples": 0, "covered": 0, "tail": []}
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
                  "backend": config.model.attention_backend, "device": args.device,
                  "compute_dtype": str(config.model.torch_dtype),
                  "parameter_dtype": str(next(model.parameters()).dtype),
                  "fast_weight_dtype": str(next(iter(state.fast_weights.values())).dtype)}
    print(json.dumps(model_info), flush=True)
    extra = dict(previous.get("extra", {}))
    if stage_source is not None:
        extra["stage_history"] = [*extra.get("stage_history", []), stage_source]
        extra.pop("wandb", None)  # A new data stage owns a new logging run.
        extra.pop("outer_lr_plan", None)
    if not args.resume:
        plan = make_schedule(config.training, stream.total_events, args.epochs)
        if plan is not None:
            if stage_from:
                plan["start_window"] = state.windows_completed
            extra["outer_lr_plan"] = plan
    plan = extra.get("outer_lr_plan")
    # Fail before writing a checkpoint if a scheduled resume lost its horizon.
    outer_learning_rate(config.training, state.windows_completed, plan)
    with start_tracker(args, config, state, model_info, stream.digest, extra.get("wandb")) as tracker:
        if tracker.identity is not None:
            extra["wandb"] = tracker.identity
        def commit():
            extra["boundary_coverage"] = coverage
            save_checkpoint(output, model, state, optimizer, config, encoder.tokenizer,
                            manifest, stream.digest, extra=extra, keep=3)
        # Save identity before the first window, including when enabling logging on an old run.
        commit()
        with _stop_requests() as stop_requested:
            _train_windows(args, model, state, optimizer, config, encoder, stream, coverage,
                           commit, tracker, plan, stop_requested)


def _train_windows(args, model, state, optimizer, config, encoder, stream, coverage, commit, tracker, plan=None, stop_requested=None):
    completed = 0
    interval = getattr(args, "save_every_windows", 50)
    while state.epoch < args.epochs:
        for window in windows(stream.from_cursor(state.next_cursor), encoder, config.training.unroll_events,
                              start_open=state.is_open):
            if stop_requested:
                commit()
                return
            cuda_timing = tracker.enabled and next(model.parameters()).device.type == "cuda"
            if cuda_timing:
                torch.cuda.synchronize(next(model.parameters()).device)
                torch.cuda.reset_peak_memory_stats(next(model.parameters()).device)
            started = time.perf_counter()
            lr = outer_learning_rate(config.training, state.windows_completed, plan)
            for group in optimizer.param_groups:
                group["lr"] = lr
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
            completed += 1
            limit_reached = args.max_windows is not None and completed >= args.max_windows
            checkpoint_seconds = 0.0
            saved = state.windows_completed % interval == 0 or limit_reached or bool(stop_requested)
            if saved:
                checkpoint_started = time.perf_counter()
                commit()
                checkpoint_seconds = time.perf_counter() - checkpoint_started
            metrics = summarize(records)
            metrics.update(loss=loss, epoch=state.epoch, cursor=state.next_cursor,
                           windows=state.windows_completed, boundary_coverage=coverage,
                           inner_lr=config.training.inner_lr, outer_lr=lr)
            if tracker.enabled:
                metrics["performance"] = {
                    "window_compute_seconds": compute_seconds,
                    "checkpoint_seconds": checkpoint_seconds,
                    "checkpoint_saved": int(saved),
                    "target_tokens_per_second": sum(r["tokens"] for r in records) / compute_seconds,
                }
                triples = coverage["triples"]
                metrics["boundary_coverage"] = dict(coverage)
                metrics["boundary_coverage"]["rate"] = coverage["covered"] / triples if triples else None
                if cuda_timing:
                    metrics["performance"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(next(model.parameters()).device)
                # Log every completed window; disk recovery follows the checkpoint cadence.
                try:
                    tracker.log(metrics)
                except Exception:
                    # Training completed safely; preserve it if the reporting service fails.
                    if not saved:
                        commit()
                    raise
            print(json.dumps(metrics), flush=True)
            if limit_reached or stop_requested:
                if not saved:
                    commit()
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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", help="resume the same data stage and schedule")
    source.add_argument("--start-stage-from", help="start a new data stage from a completed training checkpoint; preserve weights and AdamW, restart the LR schedule")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1, help="total target epochs within this stage, including completed epochs")
    parser.add_argument("--max-windows", type=int, help="stop after this many additional windows")
    parser.add_argument("--save-every-windows", type=int, default=50,
                        help="checkpoint every N total windows (default: 50); also save at epoch end and orderly stop")
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
