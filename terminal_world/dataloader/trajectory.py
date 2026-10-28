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


def build_manifest(paths, validation_fraction=0.1, split_seed=0):
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
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
                                "split": "validation" if bucket < validation_fraction else "train"})
        files.append({"path": str(path), "sha256": digest.hexdigest()})
    return {"version": 1, "files": files, "entries": entries,
            "validation_fraction": validation_fraction, "split_seed": split_seed}


class EventStream:
    def __init__(self, manifest, split="train", *, verify=True):
        if manifest.get("version") != 1 or split not in ("train", "validation"):
            raise ValueError("unsupported manifest/split")
        self.manifest, self.split = manifest, split
        self.digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if verify:
            for item in manifest["files"]:
                if digest_file(item["path"]) != item["sha256"]:
                    raise ValueError(f"data file changed: {item['path']}")
        self.entries = [e for e in manifest["entries"] if e["split"] == split]
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
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    result = build_manifest(args.paths, args.validation_fraction, args.split_seed)
    with open(args.output, "x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
