"""Protocol IDs are inserted only here; raw text never invokes special tokens."""
import json
import torch
from terminal_world.model.data import StepBatch

PROTOCOL = ("pad", "command", "output", "traj_open", "traj_close", "obs_end")


class ByteTokenizer:
    """Reversible, dependency-free tokenizer for tiny correctness runs."""
    vocab_size = 262
    ids = dict(zip(PROTOCOL, range(256, 262)))
    bos_id = None

    def encode_text(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def state_dict(self):
        return {"type": "bytes-v1", "ids": self.ids, "bos_id": None}


class HFTokenizer:
    """Serialized fast tokenizer with special-token matching disabled for raw text."""
    def __init__(self, backend_json, ids, vocab_size, bos_id=None):
        from tokenizers import Tokenizer
        data = json.loads(backend_json)
        # Special tokens remain in the model vocabulary but lose AddedToken matching.
        data["added_tokens"] = [t for t in data.get("added_tokens", []) if not t["special"]]
        self.backend = Tokenizer.from_str(json.dumps(data))
        self.backend_json = backend_json
        self.ids, self.vocab_size, self.bos_id = ids, vocab_size, bos_id

    def encode_text(self, text):
        result = self.backend.encode(text, add_special_tokens=False).ids
        if set(result) & set(self.ids.values()):
            raise ValueError("raw text tokenizer emitted a reserved protocol ID")
        return result

    def decode(self, ids):
        return self.backend.decode(ids, skip_special_tokens=False)

    def state_dict(self):
        return {"type": "hf-fast-v1", "backend_json": self.backend_json,
                "ids": self.ids, "vocab_size": self.vocab_size, "bos_id": self.bos_id}


def tokenizer_from_state(state):
    if state["type"] == "bytes-v1":
        result = ByteTokenizer()
        if state != result.state_dict():
            raise ValueError("byte tokenizer protocol mismatch")
        return result
    if state["type"] == "hf-fast-v1":
        return HFTokenizer(**{k: v for k, v in state.items() if k != "type"})
    raise ValueError("unsupported tokenizer state")


class Encoder:
    def __init__(self, tokenizer, max_output_tokens=2048, max_seq_len=4096):
        if max_output_tokens < 1:
            raise ValueError("positive output budget required")
        self.tokenizer = tokenizer
        self.max_output_tokens, self.max_seq_len = max_output_tokens, max_seq_len

    def prompt(self, event):
        t = self.tokenizer
        if event.kind == "shell":
            prompt = [t.ids["command"]] + t.encode_text(event.command)
        else:
            prompt = [t.ids["traj_" + event.command]]
        prompt += [t.ids["output"]]
        if t.bos_id is not None:
            prompt = [t.bos_id] + prompt
        if len(prompt) + self.max_output_tokens > self.max_seq_len:
            raise ValueError("full command plus output budget exceeds model capacity")
        return prompt

    def encode(self, event):
        prompt = self.prompt(event)
        body = self.tokenizer.encode_text(event.output)
        n = self.max_output_tokens
        target = (body + [self.tokenizer.ids["obs_end"]])[:n]
        seq = prompt + target
        length = len(seq) - 1
        return StepBatch(
            torch.tensor(seq[:-1], dtype=torch.long),
            torch.tensor(seq[1:], dtype=torch.long),
            torch.tensor([False] * (len(prompt) - 1) + [True] * len(target)),
            torch.ones(length, dtype=torch.bool), torch.arange(length), event.kind,
            {"cursor": event.cursor, "command": event.command if event.kind == "control" else None,
             "body_tokens": len(body), "kept_body_tokens": min(len(body), n),
             "body_truncated": len(body) > n, "has_end_token": len(body) < n},
        )
