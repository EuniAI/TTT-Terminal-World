#!/usr/bin/env python3
"""Collect terminal trajectories using mini-swe-agent's native Docker environment."""

import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--workers", type=int, default=15,
                        help="Parallel LLM trajectories/containers (default: 15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120,
                        help="Per-command execution timeout in seconds (default: 120)")
    parser.add_argument("--output", type=Path, default=Path("terminal_trajs_llm.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="z-ai/glm-5.2")
    parser.add_argument("--image", default="ubuntu:24.04", help="Pre-pulled Docker image for LLM collection")
    parser.add_argument("--min-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--api-timeout", type=float, default=60)
    parser.add_argument("--task-temperature", type=float, default=1.0,
                        help="Task generation temperature (default: 1.0); actions stay at 0.3")
    args = parser.parse_args()
    if not 0 <= args.task_temperature <= 2:
        parser.error("--task-temperature must be between 0 and 2")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.num_trajectories <= 0 or not 0 < args.timeout < float("inf"):
        parser.error("--num-trajectories and --timeout must be positive and finite")
    if not 1 <= args.min_steps <= args.max_steps or not 0 < args.api_timeout < float("inf"):
        parser.error("Require 1 <= min-steps <= max-steps and a positive finite API timeout")
    try:
        from collect_llm_terminal_data import collect_llm
        result = collect_llm(args)
    except (OSError, RuntimeError, UnicodeError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {args.output}")
    print(json.dumps(result, ensure_ascii=False))
    if result.get("stopped_on_error"):
        parser.exit(1, "Collection stopped on an error; only observed prefixes were saved.\n")


if __name__ == "__main__":
    main()
