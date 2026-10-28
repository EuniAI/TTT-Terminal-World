#!/usr/bin/env python3
"""Convert Nemotron skill-based easy or medium records to terminal trajectories.

Install: python -m pip install datasets
Run:     python scripts/prepare_terminal_data.py
Medium:  python scripts/prepare_terminal_data.py --difficulty medium
Offline: python scripts/prepare_terminal_data.py --input raw.jsonl

Each source row becomes at most one trajectory. No difficulty, command, length,
success, or deduplication filter is applied. This parser targets the published
Terminus-2 conversations format, not arbitrary chat/tool logs. Source reference:
https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import islice
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable

DATASET = "nvidia/Nemotron-Terminal-Corpus"
PARSE_ERROR = "Previous response had parsing errors:"
WARNING = "Previous response had warnings:"
OBSERVATION = re.compile(
    r"^(?:New Terminal Output:|Current terminal state:|Current Terminal Screen:)[ \t]*(?:\r?\n|$)",
    re.MULTILINE | re.IGNORECASE,
)
CONFIRMATION = (
    "\n\nAre you sure you want to mark the task as complete? "
    "This will trigger your solution to be graded and you won't be able to "
    'make any further corrections. If so, include "task_complete": true '
    "in your JSON response again."
)


class InvalidRecord(ValueError):
    """A record cannot be paired without guessing."""


def message_text(message: object) -> tuple[str, str]:
    if not isinstance(message, dict):
        raise InvalidRecord("invalid_message")
    role, content = message.get("role"), message.get("content")
    if role not in ("system", "user", "assistant") or not isinstance(content, str):
        raise InvalidRecord("unsupported_message_schema")
    return role, content


def extract_observation(content: str) -> str | None:
    """Remove known harness wrappers, never strip/normalize terminal text itself."""
    if content.startswith(PARSE_ERROR):
        return None
    matches = list(OBSERVATION.finditer(content))
    if not matches:
        return None
    match = matches[0]
    preamble = content[:match.start()]
    if preamble.strip() and not preamble.startswith(WARNING):
        return None
    output = content[match.end():]
    if match.group().lower().startswith("current terminal state:"):
        if output.endswith(CONFIRMATION):
            output = output[:-len(CONFIRMATION)]
            # Completion feedback can wrap an already formatted observation.
            # Only unwrap this known envelope, not header-like shell output.
            inner = OBSERVATION.match(output)
            if inner and inner.group().lower().startswith(
                ("new terminal output:", "current terminal screen:")
            ):
                output = output[inner.end():]
    return output


def parse_action(content: str) -> dict:
    """Read the actual JSON action outside a leading DeepSeek thinking block."""
    body = content.lstrip()
    if body.startswith("<think>"):
        end = body.find("</think>", len("<think>"))
        if end == -1:
            raise InvalidRecord("unclosed_thinking")
        body = body[end + len("</think>"):]

    # raw_decode handles braces inside quoted commands and fenced JSON. Advance
    # past whole objects so an example nested in analysis cannot become an action.
    decoder = json.JSONDecoder()
    candidates = []
    position = 0
    while position < len(body):
        start = body.find("{", position)
        if start == -1:
            break
        try:
            value, end = decoder.raw_decode(body, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        position = end
        if isinstance(value, dict) and "commands" in value:
            candidates.append(value)
    if len(candidates) != 1:
        raise InvalidRecord("missing_or_ambiguous_action")
    action = candidates[0]
    commands = action["commands"]
    if not isinstance(commands, list):
        raise InvalidRecord("invalid_commands")
    for command in commands:
        if not isinstance(command, dict) or not isinstance(command.get("keystrokes"), str):
            raise InvalidRecord("invalid_keystrokes")
    return action


def encode_command(action: dict) -> str:
    # Keep only text. Preserve source newlines and partial keystrokes verbatim;
    # inserting separators would change interactive input. Timing is omitted.
    return "".join(command["keystrokes"] for command in action["commands"])


def convert_record(record: object, stats: Counter) -> dict | None:
    """Keep paired prefixes; never skip an unobserved action and join its suffix."""
    stats["records_read"] += 1
    steps: list[dict[str, str]] = []
    executed_steps = 0
    try:
        if not isinstance(record, dict):
            raise InvalidRecord("invalid_record")
        messages = record.get("conversations")
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError as exc:
                raise InvalidRecord("invalid_conversation_json") from exc
        if not isinstance(messages, list) or not messages:
            raise InvalidRecord("missing_conversations")

        index = 0
        # Leading prompts/screens are not command-result pairs.
        while index < len(messages):
            role, content = message_text(messages[index])
            if role == "assistant":
                break
            index += 1

        while index < len(messages):
            role, content = message_text(messages[index])
            if role != "assistant":
                raise InvalidRecord("unexpected_message_order")
            if index + 1 == len(messages):
                # The last assistant response has no observed result. A null or
                # completion-only tail is normal; it is not a terminal target.
                try:
                    tail = parse_action(content)
                except InvalidRecord:
                    tail = None
                if tail is not None and tail["commands"]:
                    raise InvalidRecord("missing_final_observation")
                stats["unobserved_tail_messages"] += 1
                break
            next_role, feedback = message_text(messages[index + 1])
            if next_role != "user":
                raise InvalidRecord("missing_observation")
            if feedback.startswith(PARSE_ERROR):
                # Quality-first: stop at the first rejected call, even if later
                # retries are valid. Never resume after a problematic call.
                raise InvalidRecord("harness_parse_error")

            action = parse_action(content)
            observation = extract_observation(feedback)
            if observation is None:
                raise InvalidRecord("unknown_observation_format")
            if not action["commands"] and action.get("task_complete") is True:
                # A completion confirmation is not a terminal action. End here
                # rather than turn a repeated screen into an empty-command pair.
                stats["completion_only_endings"] += 1
                break
            steps.append({"command": encode_command(action), "output": observation})
            executed_steps += 1
            index += 2
    except InvalidRecord as exc:
        stats[f"issue:{exc}"] += 1
        if executed_steps:
            stats["prefix_only_records"] += 1

    if not executed_steps:
        stats["records_skipped"] += 1
        return None
    stats["trajectories_written"] += 1
    stats["interaction_steps"] += executed_steps
    return {"trajectory": steps}


def input_rows(args: argparse.Namespace) -> Iterable[dict]:
    if args.input and args.input.suffix == ".jsonl":
        with args.input.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {args.input}:{number}") from exc
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependency: python -m pip install datasets") from exc
    if args.input:
        if args.input.is_dir():
            root = args.input / "synthetic_tasks" / "skill_based" / args.difficulty
            if not root.is_dir():
                root = args.input
            files = sorted(root.glob("*/data_filtered.parquet"))
            if not files:
                raise ValueError(f"No {args.difficulty} */data_filtered.parquet files found in --input")
        elif args.input.suffix == ".parquet":
            files = [args.input]
        else:
            raise ValueError("--input must be a JSONL, Parquet, dataset root, or difficulty directory")
        yield from load_dataset("parquet", data_files=[str(p) for p in files],
                                split="train", streaming=True)
    else:
        yield from load_dataset(DATASET, name=f"skill_based_{args.difficulty}", split="train", streaming=True,
                                revision=args.revision)


def write_dataset(rows: Iterable[dict], output: Path, *, overwrite: bool = False) -> Counter:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    # A network/read error must not leave a partial file looking like a full set.
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                converted = convert_record(row, stats)
                if converted is not None:
                    target.write(json.dumps(converted, ensure_ascii=False, allow_nan=False) + "\n")
                if stats["records_read"] % 1000 == 0:
                    print(f"Read {stats['records_read']:,}; wrote "
                          f"{stats['trajectories_written']:,} trajectories", file=sys.stderr)
            if not stats["trajectories_written"]:
                raise ValueError(f"No usable trajectories; verify source schema. Counts: {dict(stats)}")
        if overwrite:
            os.replace(temporary, output)
        else:
            # Atomic publication without overwriting an existing file, including
            # one created by another process while this conversion was running.
            os.link(temporary, output)
        return stats
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--difficulty", choices=("easy", "medium"), default="easy")
    parser.add_argument("--output", type=Path,
                        help="Default: terminal_trajs.jsonl (easy), terminal_trajs_medium.jsonl (medium)")
    parser.add_argument("--input", type=Path, help="Optional local raw JSONL/Parquet, dataset root, or difficulty directory")
    parser.add_argument("--revision", default="main", help="Hugging Face revision (default: main)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Trial run: read only the first N source records (default: all)")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("terminal_trajs.jsonl" if args.difficulty == "easy"
                           else "terminal_trajs_medium.jsonl")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    try:
        if args.input and not args.input.exists():
            raise FileNotFoundError(args.input)
        if args.input and args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output must be different paths")
        rows = input_rows(args)
        if args.limit is not None:
            rows = islice(rows, args.limit)
            print(f"Trial run: reading at most {args.limit} source records; "
                  "invalid records may produce fewer trajectories.", file=sys.stderr)
        stats = write_dataset(rows, args.output, overwrite=args.overwrite)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {args.output}", file=sys.stderr)
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False), file=sys.stderr)
    if stats["prefix_only_records"] or stats["records_skipped"]:
        print("Some records were incomplete or unsupported; see counts above. "
              "No command/length/difficulty filtering was applied.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
