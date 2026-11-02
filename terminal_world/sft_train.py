"""python -m terminal_world.sft_train: independent parallel SFT training loop."""
import argparse
import json
from pathlib import Path
import random
import time
import torch
from terminal_world.checkpoint import load_checkpoint, save_checkpoint
from terminal_world.config import RunConfig
from terminal_world.dataloader.encoding import ByteTokenizer, Encoder
from terminal_world.dataloader.events import supervised_batches
from terminal_world.dataloader.trajectory import EventStream
from terminal_world.metrics import summarize
from terminal_world.model.loop import TrainingState
from terminal_world.model.sft import train_batch
from terminal_world.model.transformer import create_model
from terminal_world.schedule import make_schedule, outer_learning_rate
from terminal_world.tracking import start_tracker
# Reuse lifecycle utilities, not the TTT training loop.
from terminal_world.train import start_new_stage, _stop_requests


def run(args):
    if args.epochs < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("epochs and max-steps must be positive")
    if getattr(args, "save_every_steps", 50) < 1:
        raise ValueError("save-every-steps must be positive")
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
        if config.model.adaptive_layers:
            raise ValueError("sft_train requires adaptive_layers=0")
        if args.config and RunConfig.load(args.config).to_dict() != config.to_dict():
            raise ValueError("resume config mismatch; use the checkpoint's configuration")
        if stage_from:
            stage_source = start_new_stage(state, previous, stream)
            stage_source["checkpoint"] = str(Path(stage_from).resolve())
    else:
        if not args.config:
            raise ValueError("fresh training requires --config")
        config = RunConfig.load(args.config)
        if config.model.adaptive_layers:
            raise ValueError("sft_train requires adaptive_layers=0")
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
        encoder = Encoder(tokenizer, config.training.max_output_tokens, config.model.max_seq_len,
                          trajectory_context=config.training.trajectory_context)
    if stream.state_at(state.next_cursor) != state.is_open:
        raise ValueError("checkpoint cursor/protocol state mismatch")
    model_info = {"theta_parameters": sum(p.numel() for p in model.parameters()),
                  "fast_parameters": sum(p.numel() for p in state.fast_weights.values()),
                  "backend": config.model.attention_backend, "device": args.device,
                  "compute_dtype": str(config.model.torch_dtype),
                  "parameter_dtype": str(next(model.parameters()).dtype),
                  "training_method": "sft", "batch_execution": "parallel",
                  "batch_size": config.training.sft_batch_size or config.training.unroll_events,
                  "fast_weight_dtype": str(next(iter(state.fast_weights.values())).dtype) if state.fast_weights else None}
    print(json.dumps(model_info), flush=True)
    extra = dict(previous.get("extra", {}))
    extra["sft_execution"] = "parallel"
    if stage_source is not None:
        extra["stage_history"] = [*extra.get("stage_history", []), stage_source]
        extra.pop("wandb", None)  # A new data stage owns a new logging run.
        extra.pop("outer_lr_plan", None)
    if not args.resume:
        plan = make_schedule(config.training, stream.total_events, args.epochs,
                             events_per_step=(config.training.sft_batch_size or config.training.unroll_events))
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
            save_checkpoint(output, model, state, optimizer, config, encoder.tokenizer,
                            manifest, stream.digest, extra=extra, keep=3)
        # Save identity before the first step, including when enabling logging on an old run.
        commit()
        with _stop_requests() as stop_requested:
            _train_sft(args, model, state, optimizer, config, encoder, stream,
                       commit, tracker, plan, stop_requested)


def _train_sft(args, model, state, optimizer, config, encoder, stream,
               commit, tracker, plan, stop_requested):
    completed = 0
    batch_size = config.training.sft_batch_size or config.training.unroll_events
    device = next(model.parameters()).device
    while state.epoch < args.epochs:
        batches = supervised_batches(
            stream.from_cursor(state.next_cursor), encoder, batch_size,
            start_open=state.is_open, history=stream.history_before(state.next_cursor, encoder))
        for batch in batches:
            if stop_requested:
                commit()
                return
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            lr = outer_learning_rate(config.training, state.windows_completed, plan)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss, records = train_batch(model, optimizer, state, batch, pad_id=encoder.tokenizer.ids["pad"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            completed += 1
            done = args.max_steps is not None and completed >= args.max_steps
            saved = state.windows_completed % args.save_every_steps == 0 or done or bool(stop_requested)
            checkpoint_seconds = 0.0
            if saved:
                checkpoint_started = time.perf_counter()
                commit()
                checkpoint_seconds = time.perf_counter() - checkpoint_started
            metrics = summarize(records)
            metrics.update(loss=loss, epoch=state.epoch, cursor=state.next_cursor,
                           steps=state.windows_completed,
                           windows=state.windows_completed,  # Existing W&B x-axis compatibility.
                           learning_rate=lr, batch_size=len(batch.steps))
            metrics["performance"] = {
                "step_compute_seconds": seconds,
                "target_tokens_per_second": sum(r["tokens"] for r in records) / seconds,
                "checkpoint_seconds": checkpoint_seconds,
            }
            if device.type == "cuda":
                metrics["performance"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            try:
                tracker.log(metrics)
            except Exception:
                if not saved:
                    commit()
                raise
            print(json.dumps(metrics), flush=True)
            if done or stop_requested:
                if not saved:
                    commit()
                return
        if state.is_open:
            raise ValueError("epoch ended inside a trajectory")
        state.epoch += 1
        state.next_cursor = 0
        commit()


def main():
    parser = argparse.ArgumentParser(description="Parallel minibatch SFT with trajectory context")
    parser.add_argument("--config")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", help="resume the same data stage and schedule")
    source.add_argument("--start-stage-from", help="start a new data stage from a completed training checkpoint; preserve weights and AdamW, restart the LR schedule")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1, help="total target epochs within this stage, including completed epochs")
    parser.add_argument("--max-steps", type=int, metavar="N",
                        help="stop after this many additional optimizer steps; omit to complete --epochs")
    parser.add_argument("--save-every-steps", type=int, default=50, metavar="N",
                        help="checkpoint every N total optimizer steps (default: 50); also save at epoch end and orderly stop")
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
