"""python -m terminal_world.train --config ... --manifest ... --output ..."""
import argparse
import json
from pathlib import Path
import random
import torch
from terminal_world.checkpoint import load_checkpoint, save_checkpoint
from terminal_world.config import RunConfig
from terminal_world.dataloader.encoding import ByteTokenizer, Encoder
from terminal_world.dataloader.events import windows
from terminal_world.dataloader.trajectory import EventStream
from terminal_world.metrics import summarize
from terminal_world.model.loop import TrainingState, train_window
from terminal_world.model.transformer import create_model


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
        save_checkpoint(output, model, state, optimizer, config, tokenizer, manifest, stream.digest,
                        extra={"boundary_coverage": coverage})
    if stream.state_at(state.next_cursor) != state.is_open:
        raise ValueError("checkpoint cursor/protocol state mismatch")
    print(json.dumps({"theta_parameters": sum(p.numel() for p in model.parameters()),
                      "fast_parameters": sum(p.numel() for p in state.fast_weights.values()),
                      "backend": config.model.attention_backend, "device": args.device}), flush=True)
    completed = 0
    while state.epoch < args.epochs:
        for window in windows(stream.from_cursor(state.next_cursor), encoder, config.training.unroll_events,
                              start_open=state.is_open):
            loss, records = train_window(model, optimizer, state, window, config.training, pad_id=encoder.tokenizer.ids["pad"])
            tags = [(r.get("command") if r["kind"] == "control" else "shell", state.windows_completed) for r in records]
            joined = coverage["tail"] + tags
            for i in range(len(joined) - 2):
                if [t[0] for t in joined[i:i + 3]] == ["close", "open", "shell"]:
                    coverage["triples"] += 1
                    coverage["covered"] += int(len({t[1] for t in joined[i:i + 3]}) == 1)
            coverage["tail"] = joined[-2:]
            # Checkpoint every committed window; no shell command is ever executed here.
            save_checkpoint(output, model, state, optimizer, config, encoder.tokenizer, manifest, stream.digest,
                            extra={"boundary_coverage": coverage})
            metrics = summarize(records)
            metrics.update(loss=loss, epoch=state.epoch, cursor=state.next_cursor,
                           windows=state.windows_completed, boundary_coverage=coverage)
            print(json.dumps(metrics), flush=True)
            completed += 1
            if args.max_windows is not None and completed >= args.max_windows:
                return
        if state.is_open:
            raise ValueError("epoch ended inside a trajectory")
        state.epoch += 1
        state.next_cursor = 0
        save_checkpoint(output, model, state, optimizer, config, encoder.tokenizer, manifest, stream.digest,
                        extra={"boundary_coverage": coverage})
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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
