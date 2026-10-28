"""Llama-style decoder with external prime-MLP weights and no hidden memory.

Explicit F.linear calls for phi avoid torch.func transforms around FlashBoB.
The nn.Module owns theta only; optimizer membership cannot accidentally include phi.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from terminal_world.config import ModelConfig
from terminal_world.model.attention import causal_attention, validate_flashbob


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        work = x if x.dtype == torch.float64 else x.float()
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, m = config.hidden_size, config.intermediate_size
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(h, m, False), nn.Linear(h, m, False), nn.Linear(m, h, False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def prime_mlp(x, weights, prefix):
    gate = F.linear(x, weights[prefix + "gate_proj.weight"])
    up = F.linear(x, weights[prefix + "up_proj.weight"])
    return F.linear(F.silu(gate) * up, weights[prefix + "down_proj.weight"])


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        h, d = config.hidden_size, config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(h, config.num_heads * d, bias=config.qkv_bias)
        self.k_proj = nn.Linear(h, config.num_kv_heads * d, bias=config.qkv_bias)
        self.v_proj = nn.Linear(h, config.num_kv_heads * d, bias=config.qkv_bias)
        self.o_proj = nn.Linear(h, h, False)
        self.flash_fn = None

    def forward(self, x, positions):
        c = self.config
        b, n, _ = x.shape
        d = c.hidden_size // c.num_heads
        q = self.q_proj(x).view(b, n, c.num_heads, d).transpose(1, 2)
        k = self.k_proj(x).view(b, n, c.num_kv_heads, d).transpose(1, 2)
        v = self.v_proj(x).view(b, n, c.num_kv_heads, d).transpose(1, 2)
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        inv = 1 / c.rope_theta ** (torch.arange(0, d, 2, device=x.device, dtype=work_dtype) / d)
        angles = positions.to(work_dtype)[..., None] * inv
        angles = torch.cat((angles, angles), -1).unsqueeze(1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        def rotate(t):
            return torch.cat((-t[..., d // 2:], t[..., :d // 2]), -1)
        q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
        out = causal_attention(q, k, v, c.attention_backend, self.flash_fn)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, c.hidden_size))


class Block(nn.Module):
    def __init__(self, config, adaptive):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)
        self.prime_norm = RMSNorm(config.hidden_size, config.rms_norm_eps) if adaptive else None

    def forward(self, x, positions, fast=None, index=None):
        x = x + self.self_attn(self.input_layernorm(x), positions)
        if self.prime_norm is not None and fast is not None:
            x = x + prime_mlp(self.prime_norm(x), fast, f"layers.{index}.prime.")
        return x + self.mlp(self.post_attention_layernorm(x))


class TerminalTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.split = config.num_layers - config.adaptive_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Block(config, i >= self.split) for i in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.to(dtype=config.torch_dtype)
        self.eval()

    def configure_backend(self):
        c = self.config
        if c.attention_backend == "flashbob":
            parameter = next(self.parameters())
            fn = validate_flashbob(parameter.device, parameter.dtype, c.hidden_size // c.num_heads, c.num_heads, c.num_kv_heads)
            for layer in self.layers:
                layer.self_attn.flash_fn = fn

    def fast_shapes(self):
        c = self.config
        return {f"layers.{i}.prime.{name}.weight": shape
                for i in range(self.split, c.num_layers)
                for name, shape in (("gate_proj", (c.intermediate_size, c.hidden_size)),
                                    ("up_proj", (c.intermediate_size, c.hidden_size)),
                                    ("down_proj", (c.hidden_size, c.intermediate_size)))}

    def initialize_fast_weights(self):
        """Only called by the fresh-model factory; resume never invokes this."""
        reference = next(self.parameters())
        weights = {}
        for name, shape in self.fast_shapes().items():
            value = torch.empty(shape, device=reference.device, dtype=reference.dtype)
            if "down_proj" in name:
                value.zero_()
            else:
                nn.init.kaiming_uniform_(value, a=math.sqrt(5))
            weights[name] = value.requires_grad_(True)
        return weights

    def validate_fast(self, fast):
        expected = self.fast_shapes()
        if set(fast) != set(expected):
            raise ValueError("fast-weight names mismatch; initialization fallback is forbidden")
        reference = next(self.parameters())
        storages = {p.untyped_storage().data_ptr() for p in self.parameters()}
        for name, value in fast.items():
            if tuple(value.shape) != expected[name] or value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"fast-weight shape/device/dtype mismatch: {name}")
            pointer = value.untyped_storage().data_ptr()
            if pointer in storages:
                raise ValueError("theta/phi or phi/phi storage alias")
            storages.add(pointer)

    def prefix(self, ids, positions):
        if ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input exceeds model capacity")
        x = self.embed_tokens(ids)
        for i in range(self.split):
            x = self.layers[i](x, positions)
        return x

    def suffix(self, x, positions, fast):
        for i in range(self.split, len(self.layers)):
            x = self.layers[i](x, positions, fast, i)
        return self.lm_head(self.norm(x))

    def forward(self, ids, fast, positions=None):
        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
        return self.suffix(self.prefix(ids, positions), positions, fast)


def create_model(config, device="cpu"):
    model = TerminalTransformer(config).to(device)
    model.configure_backend()
    fast = model.initialize_fast_weights()
    model.validate_fast(fast)
    return model, fast
