from dataclasses import dataclass, field
import torch


@dataclass(frozen=True)
class Event:
    kind: str
    command: str
    output: str = ""
    cursor: int = -1

    def __post_init__(self):
        if self.kind not in ("shell", "control"):
            raise ValueError("unknown event kind")
        if not isinstance(self.command, str) or not isinstance(self.output, str):
            raise ValueError("command/output must be strings")
        if self.kind == "control" and (self.command not in ("open", "close") or self.output):
            raise ValueError("invalid control event")


@dataclass
class StepBatch:
    input_ids: torch.Tensor
    target_tokens: torch.Tensor
    loss_masks: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    kind: str
    metadata: dict = field(default_factory=dict)

    @property
    def length(self):
        return self.input_ids.numel()

    def to(self, device):
        return StepBatch(*(getattr(self, name).to(device) for name in (
            "input_ids", "target_tokens", "loss_masks", "attention_mask", "position_ids"
        )), self.kind, self.metadata)


@dataclass
class WindowBatch:
    steps: list[StepBatch]
    next_cursor: int
    is_open: bool
