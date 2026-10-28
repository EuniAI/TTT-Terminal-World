"""Content-verified JSONL manifest, fixed ordering, lazy trajectory reads."""
import argparse
import hashlib
import json
from pathlib import Path
from terminal_world.model.data import Event


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_from_line(line):
    data = json.loads(line)
    trajectory = data["trajectory"]
    if not isinstance(trajectory, list):
        raise ValueError("trajectory must be a list")
    result = []
    for step in trajectory:
        if not isinstance(step, dict) or not isinstance(step.get("command"), str) or not isinstance(step.get("output"), str):
            raise ValueError("each complete observation requires string command/output")
        if step.get("timeout") or step.get("timed_out") or step.get("incomplete"):
            raise ValueError("incomplete observations must be removed before training")
        result.append({"command": step["command"], "output": step["output"]})
    return result


def build_manifest(paths, test_fraction=0.1, split_seed=0, *, validation_fraction=None):
    # Preserve the old keyword for callers; new manifests use canonical names.
    if validation_fraction is not None:
        test_fraction = validation_fraction
    if not 0 <= test_fraction < 1:
        raise ValueError("test_fraction must be in [0, 1)")
    files, entries = [], []
    for name in paths:
        path = Path(name).resolve()
        file_index = len(files)
        digest = hashlib.sha256()
        with path.open("rb") as f:
            line_number = 0
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                digest.update(line)
                line_number += 1
                trajectory = trajectory_from_line(line)
                canonical = json.dumps(trajectory, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                content_hash = hashlib.sha256(canonical.encode()).hexdigest()
                bucket = int(hashlib.sha256(f"{split_seed}:{content_hash}".encode()).hexdigest(), 16) / 2**256
                entries.append({"file": file_index, "offset": offset, "line": line_number,
                                "steps": len(trajectory), "content_hash": content_hash,
                                "split": "test" if bucket < test_fraction else "training"})
        files.append({"path": str(path), "sha256": digest.hexdigest()})
    return {"version": 1, "files": files, "entries": entries,
            "test_fraction": test_fraction, "split_seed": split_seed}


def build_file_split_manifest(training_paths, test_paths):
    """Index already-split files without sampling or changing their row order."""
    training_paths, test_paths = list(training_paths), list(test_paths)
    if not training_paths or not test_paths:
        raise ValueError("both training and test files are required")
    paths = training_paths + test_paths
    if len({Path(p).resolve() for p in paths}) != len(paths):
        raise ValueError("each input file must be listed only once")
    manifest = build_manifest(paths, test_fraction=0)
    hashes = {"training": set(), "test": set()}
    counts = {"training": 0, "test": 0}
    for entry in manifest["entries"]:
        split = "training" if entry["file"] < len(training_paths) else "test"
        entry["split"] = split
        hashes[split].add(entry["content_hash"])
        counts[split] += 1
    if not all(counts.values()):
        raise ValueError("training and test splits must both contain trajectories")
    if hashes["training"] & hashes["test"]:
        raise ValueError("identical trajectories occur in both training and test")
    manifest.update(test_fraction=counts["test"] / sum(counts.values()),
                    split_seed=None, split_strategy="preassigned_files")
    return manifest


def canonical_split(split):
    aliases = {"training": "training", "test": "test", "train": "training", "validation": "test"}
    if split not in aliases:
        raise ValueError(f"unsupported split: {split!r}; expected training or test")
    return aliases[split]


class EventStream:
    def __init__(self, manifest, split="training", *, verify=True):
        if manifest.get("version") != 1:
            raise ValueError("unsupported manifest/split")
        split = canonical_split(split)
        self.manifest, self.split = manifest, split
        self.digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if verify:
            for item in manifest["files"]:
                if digest_file(item["path"]) != item["sha256"]:
                    raise ValueError(f"data file changed: {item['path']}")
        # Normalize only for selection: do not change legacy manifest digests.
        self.entries = [e for e in manifest["entries"] if canonical_split(e["split"]) == split]
        self.total_events = sum(e["steps"] + 2 for e in self.entries)

    def state_at(self, cursor):
        if not 0 <= cursor <= self.total_events:
            raise ValueError("event cursor out of range")
        start = 0
        for entry in self.entries:
            end = start + entry["steps"] + 2
            if start <= cursor < end:
                return cursor > start
            start = end
        return False

    def from_cursor(self, cursor=0):
        self.state_at(cursor)
        start = 0
        for entry in self.entries:
            end = start + entry["steps"] + 2
            if end <= cursor:
                start = end
                continue
            path = self.manifest["files"][entry["file"]]["path"]
            with open(path, "rb") as f:
                f.seek(entry["offset"])
                trajectory = trajectory_from_line(f.readline())
            canonical = json.dumps(trajectory, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if len(trajectory) != entry["steps"] or hashlib.sha256(canonical.encode()).hexdigest() != entry["content_hash"]:
                raise ValueError("trajectory no longer matches manifest")
            events = [Event("control", "open", cursor=start)]
            events += [Event("shell", s["command"], s["output"], start + i + 1) for i, s in enumerate(trajectory)]
            events.append(Event("control", "close", cursor=end - 1))
            yield from (e for e in events if e.cursor >= cursor)
            start = end


def main():
    parser = argparse.ArgumentParser(description="Create an immutable ordered training manifest")
    parser.add_argument("paths", nargs="*", help="unsplit JSONL files for hash-based splitting")
    parser.add_argument("--training-files", nargs="+", help="already-split training JSONL files")
    parser.add_argument("--test-files", nargs="+", help="already-split test JSONL files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-fraction", "--validation-fraction", dest="test_fraction", type=float)
    parser.add_argument("--split-seed", type=int)
    args = parser.parse_args()
    if args.training_files is not None or args.test_files is not None:
        if args.paths or not args.training_files or not args.test_files:
            parser.error("use both --training-files and --test-files without positional paths")
        if args.test_fraction is not None or args.split_seed is not None:
            parser.error("preassigned files cannot be combined with sampling options")
        result = build_file_split_manifest(args.training_files, args.test_files)
    else:
        if not args.paths:
            parser.error("provide input paths or explicit training/test files")
        result = build_manifest(args.paths, 0.1 if args.test_fraction is None else args.test_fraction,
                                0 if args.split_seed is None else args.split_seed)
    with open(args.output, "x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
