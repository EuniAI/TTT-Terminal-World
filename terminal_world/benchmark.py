"""Full-window (inner+outer) CUDA timings with identical initial parameters."""
import argparse
from dataclasses import replace
import json
import time
import torch
from terminal_world.checkpoint import runtime_info
from terminal_world.config import ModelConfig, TrainConfig
from terminal_world.dataloader.encoding import Encoder, ByteTokenizer
from terminal_world.model.data import Event, WindowBatch
from terminal_world.model.loop import TrainingState, train_window
from terminal_world.model.transformer import create_model


def benchmark(config, training, steps, backend, *, repeats, warmup):
    config = replace(config, attention_backend=backend)
    torch.manual_seed(0)
    model, fast = create_model(config, "cuda")
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_fast = {k: v.detach().clone() for k, v in fast.items()}
    timings, peaks = [], []
    for iteration in range(warmup + repeats + 1):
        model.load_state_dict(original)
        state = TrainingState({k: v.clone().requires_grad_(True) for k, v in initial_fast.items()})
        optimizer = torch.optim.AdamW(model.parameters(), lr=training.outer_lr, weight_decay=0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        train_window(model, optimizer, state, WindowBatch(steps, len(steps), False), training)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        peak = torch.cuda.max_memory_allocated()
        if iteration == 0:
            first = elapsed
        elif iteration > warmup:
            timings.append(elapsed)
            peaks.append(peak)
        del state, optimizer
    return {"backend": backend, "first_window_seconds": first,
            "mean_window_seconds": sum(timings) / len(timings),
            "peak_allocated_bytes": max(peaks), "samples_seconds": timings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--prefix-batch-size", type=int, default=3)
    parser.add_argument("--output-length", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA; CPU timings cannot validate FlashBoB")
    if args.repeats < 1 or args.warmup < 0 or args.output_length < 1:
        raise ValueError("invalid benchmark lengths")
    config = ModelConfig(hidden_size=128, intermediate_size=256, num_layers=4, adaptive_layers=2,
                         num_heads=4, num_kv_heads=2, dtype="bfloat16", max_seq_len=args.output_length + 128)
    training = TrainConfig(0.01, 0.0001, 6, args.prefix_batch_size, max_output_tokens=args.output_length)
    training.validate()
    encoder = Encoder(ByteTokenizer(), training.max_output_tokens, config.max_seq_len)
    events = [Event("control", "open"), Event("shell", "cat x", "a" * args.output_length),
              Event("control", "close"), Event("control", "open"),
              Event("shell", "ls", ""), Event("control", "close")]
    steps = [encoder.encode(e).to("cuda") for e in events]
    print(json.dumps({"runtime": runtime_info(), "configuration": vars(args),
                      "results": [benchmark(config, training, steps, backend, repeats=args.repeats, warmup=args.warmup)
                                  for backend in ("math", "flashbob")]}))


if __name__ == "__main__":
    main()
