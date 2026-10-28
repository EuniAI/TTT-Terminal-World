"""Explicit run configuration; defaults are for correctness experiments only."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import torch

E2E_REVISION = "8a2d685c080d895a4b138eb6b5ff69a840e40968"
FLASHBOB_REVISION = "f2bd4a38a9b65bae455ef4336d7b54f0acd3caca"


@dataclass
class ModelConfig:
    vocab_size: int = 262
    hidden_size: int = 32
    intermediate_size: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 4
    adaptive_layers: int = 1
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_backend: str = "math"
    dtype: str = "float32"
    tie_embeddings: bool = False
    model_id: str = "tiny-random"
    revision: str = "native-v1"

    def validate(self):
        missing = [k for k, v in asdict(self).items() if v is None]
        if missing:
            raise ValueError(f"model configuration is incomplete: {missing}")
        for key in ("vocab_size", "hidden_size", "intermediate_size", "num_layers",
                    "num_heads", "num_kv_heads", "max_seq_len"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if not 1 <= self.adaptive_layers <= self.num_layers:
            raise ValueError("adaptive_layers must select a nonempty suffix")
        if self.hidden_size % self.num_heads or self.num_heads % self.num_kv_heads:
            raise ValueError("incompatible attention heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if self.dtype not in ("float32", "float64", "bfloat16", "float16"):
            raise ValueError("unsupported dtype")
        if self.attention_backend not in ("math", "flashbob"):
            raise ValueError("unknown attention backend")
        if not self.model_id or not self.revision:
            raise ValueError("model_id and pinned revision are required")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("invalid RoPE or normalization configuration")

    @property
    def torch_dtype(self):
        return getattr(torch, self.dtype)


@dataclass
class TrainConfig:
    inner_lr: float
    outer_lr: float
    unroll_events: int
    prefix_batch_size: int
    max_output_tokens: int = 2048
    inner_clip_norm: float | None = None
    outer_weight_decay: float = 0.0
    seed: int = 0

    def validate(self):
        missing = [k for k, v in asdict(self).items() if v is None and k != "inner_clip_norm"]
        if missing:
            raise ValueError(f"training configuration is incomplete: {missing}")
        for key in ("inner_lr", "outer_lr"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.unroll_events < 2 or self.prefix_batch_size < 1 or self.max_output_tokens < 1:
            raise ValueError("unroll_events >= 2; prefix batch and output budget >= 1")
        if self.inner_clip_norm is not None and (
            not math.isfinite(self.inner_clip_norm) or self.inner_clip_norm <= 0
        ):
            raise ValueError("inner_clip_norm must be finite and positive")
        if not math.isfinite(self.outer_weight_decay) or self.outer_weight_decay < 0:
            raise ValueError("invalid outer weight decay")


@dataclass
class RunConfig:
    model: ModelConfig
    training: TrainConfig

    def validate(self):
        self.model.validate()
        self.training.validate()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        result = cls(ModelConfig(**data["model"]), TrainConfig(**data["training"]))
        result.validate()
        return result

    @classmethod
    def load(cls, path):
        import yaml
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))
