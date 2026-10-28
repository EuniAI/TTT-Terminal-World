"""Full causal attention with an explicit double-backward-capable baseline."""
import importlib.metadata
import json
import torch
from terminal_world.config import FLASHBOB_REVISION


def validate_flashbob(device, dtype, head_dim, heads, kv_heads):
    if torch.device(device).type != "cuda":
        raise ValueError("FlashBoB requires CUDA in this project; select math for CPU")
    torch_version = ".".join(torch.__version__.split(".")[:2])
    expected = {"2.8": "3.4", "2.9": "3.5"}.get(torch_version)
    if expected is None:
        raise RuntimeError("FlashBoB requires PyTorch 2.8/Triton 3.4 or 2.9/3.5")
    import triton
    if not triton.__version__.startswith(expected + "."):
        raise RuntimeError("unsupported Triton version")
    if head_dim not in (32, 64, 128) or dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FlashBoB requires FP16/BF16 and head dimension 32/64/128")
    if heads != kv_heads and (dtype != torch.bfloat16 or heads // kv_heads > 32):
        raise ValueError("FlashBoB GQA requires BF16 and head ratio <= 32")
    import bob
    # PEP 610 records the exact commit for the documented pip VCS installation.
    distributions = importlib.metadata.packages_distributions().get("bob", [])
    revisions = []
    for name in distributions:
        direct = importlib.metadata.distribution(name).read_text("direct_url.json")
        if direct:
            revisions.append(json.loads(direct).get("vcs_info", {}).get("commit_id"))
    if FLASHBOB_REVISION not in revisions:
        raise RuntimeError("install FlashBoB from the pinned VCS requirement (commit provenance missing)")
    return bob.sdpa_bob


def causal_attention(q, k, v, backend="math", flash_fn=None):
    if backend == "flashbob":
        if flash_fn is None:
            raise RuntimeError("FlashBoB must pass startup validation")
        return flash_fn(q, k, v, is_causal=True, scale=q.shape[-1] ** -0.5, window_size=0)
    if backend != "math":
        raise ValueError("unknown attention backend")
    ratio = q.shape[1] // k.shape[1]
    if ratio != 1:
        k, v = k.repeat_interleave(ratio, 1), v.repeat_interleave(ratio, 1)
    # FP32 accumulations for low precision, retain FP64 for finite-difference tests.
    work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    scores = q.to(work_dtype) @ k.to(work_dtype).transpose(-2, -1) * q.shape[-1] ** -0.5
    mask = torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool).triu(1)
    weights = scores.masked_fill(mask, float("-inf")).softmax(-1)
    return (weights @ v.to(work_dtype)).to(q.dtype)
