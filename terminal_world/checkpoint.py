"""Atomic trusted-local checkpoints; missing current memory is always an error."""
import importlib.metadata
import os
import platform
import random
import tempfile
import uuid
import warnings
from pathlib import Path
import torch
from terminal_world.config import RunConfig, E2E_REVISION, FLASHBOB_REVISION
from terminal_world.dataloader.encoding import Encoder, tokenizer_from_state
from terminal_world.model.loop import TrainingState
from terminal_world.model.transformer import TerminalTransformer
from terminal_world.model.meta_model import ensure_finite


def runtime_info():
    result = {"python": platform.python_version(), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              "e2e_revision": E2E_REVISION, "flashbob_revision": FLASHBOB_REVISION}
    for name in ("triton", "tokenizers", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def retained_atomic_save(payload, path, *, keep=3):
    """Keep recent window snapshots; the public path is a hard link to the latest.

    Publish before pruning so a failed write never removes the last good checkpoint.
    Re-saving a window replaces that snapshot rather than consuming a history slot.
    """
    if keep < 1:
        raise ValueError("checkpoint retention must be positive")
    path = Path(path)
    history = path.parent / (path.stem + ".checkpoints")
    snapshot = history / f"window-{payload['windows_completed']:012d}.pt"
    atomic_save(payload, snapshot)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.link"
    try:
        os.link(snapshot, temporary)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    # Sort by save time, not window number, so resuming an older snapshot is supported.
    candidates = [p for p in history.iterdir()
                  if p.name.startswith("window-") and p.suffix == ".pt"
                  and p.stem.removeprefix("window-").isdigit()]
    older = sorted((p for p in candidates if p != snapshot),
                   key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for obsolete in older[keep - 1:]:
        obsolete.unlink()


def save_checkpoint(path, model, state, optimizer, config, tokenizer, manifest, manifest_digest, *, mode="train", extra=None,
                    keep=1):
    model.validate_fast(state.fast_weights)
    ensure_finite(model.parameters(), "checkpoint theta")
    ensure_finite(state.fast_weights.values(), "checkpoint phi")
    if any(not t.is_leaf for t in state.fast_weights.values()):
        raise ValueError("checkpoints may only be saved after graph truncation")
    payload = {"format": 1, "mode": mode, "theta": model.state_dict(),
               "fast_weights": {k: v.detach() for k, v in state.fast_weights.items()},
               "optimizer": optimizer.state_dict() if optimizer is not None else None,
               "config": config.to_dict(), "tokenizer": tokenizer.state_dict(),
               "manifest": manifest, "manifest_digest": manifest_digest,
               "cursor": state.next_cursor, "epoch": state.epoch, "is_open": state.is_open,
               "windows_completed": state.windows_completed, "runtime": runtime_info(),
               "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                       "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               "extra": extra or {}}
    if keep > 1:
        retained_atomic_save(payload, path, keep=keep)
    else:
        atomic_save(payload, path)


def load_checkpoint(path, *, device="cpu", expected_manifest_digest=None, training=True, restore_rng=True):
    # Contains Python RNG state: only load checkpoints from trusted sources.
    data = torch.load(path, map_location="cpu", weights_only=False)
    required = {"theta", "fast_weights", "config", "tokenizer", "manifest_digest", "manifest",
                "cursor", "epoch", "is_open", "rng", "runtime", "windows_completed", "mode", "optimizer"}
    if data.get("format") != 1 or not required <= data.keys() or not data["fast_weights"]:
        raise ValueError("incomplete checkpoint; cannot fall back to initial fast weights")
    if expected_manifest_digest is not None and expected_manifest_digest != data["manifest_digest"]:
        raise ValueError("checkpoint data manifest mismatch")
    if training and (data["mode"] != "train" or data["optimizer"] is None):
        raise ValueError("training resume requires a training checkpoint and optimizer")
    if data["runtime"]["e2e_revision"] != E2E_REVISION or data["runtime"]["flashbob_revision"] != FLASHBOB_REVISION:
        raise ValueError("upstream revision mismatch")
    config = RunConfig.from_dict(data["config"])
    # Constructing a parameter container must not disturb the caller's RNG during evaluation.
    with torch.random.fork_rng(devices=[]):
        model = TerminalTransformer(config.model).to(device)
    model.load_state_dict(data["theta"], strict=True)
    model.configure_backend()
    old_dtypes = {v.dtype for v in (*data["theta"].values(), *data["fast_weights"].values())}
    if old_dtypes & {torch.bfloat16, torch.float16}:
        warnings.warn("Promoting legacy low-precision checkpoint weights/state to FP32 for updates; "
                      "previously rounded-off updates cannot be recovered.")
    fast = {k: v.to(device=device, dtype=config.model.parameter_dtype).detach().requires_grad_(True)
            for k, v in data["fast_weights"].items()}
    model.validate_fast(fast)
    ensure_finite(model.parameters(), "restored theta")
    ensure_finite(fast.values(), "restored phi")
    state = TrainingState(fast, data["cursor"], data["epoch"], data["is_open"], data["windows_completed"])
    tokenizer = tokenizer_from_state(data["tokenizer"])
    if tokenizer.vocab_size != config.model.vocab_size:
        raise ValueError("tokenizer/model vocabulary mismatch")
    encoder = Encoder(tokenizer, config.training.max_output_tokens, config.model.max_seq_len)
    optimizer = None
    if training:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.outer_lr, weight_decay=config.training.outer_weight_decay)
        optimizer.load_state_dict(data["optimizer"])
        # Explicitly retain FP32 moments when migrating a legacy BF16 AdamW state.
        for parameter, values in optimizer.state.items():
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if name in values:
                    values[name] = values[name].to(device=parameter.device, dtype=parameter.dtype)
    if restore_rng:
        random.setstate(data["rng"]["python"])
        torch.set_rng_state(data["rng"]["torch"])
        if data["rng"]["cuda"]:
            if not torch.cuda.is_available() or len(data["rng"]["cuda"]) != torch.cuda.device_count():
                raise ValueError("CUDA RNG topology mismatch")
            torch.cuda.set_rng_state_all(data["rng"]["cuda"])
    return model, state, optimizer, config, encoder, data


def resume_session(path, device="cpu"):
    from terminal_world.session import Session
    model, state, _, config, encoder, data = load_checkpoint(path, device=device, training=False, restore_rng=False)
    session = Session(model, state.fast_weights, encoder, alpha=config.training.inner_lr,
                      clip_norm=config.training.inner_clip_norm, is_open=state.is_open)
    session.observations = data["extra"].get("observations", 0)
    return session


def save_session(path, session, config, *, manifest=None, manifest_digest=None):
    state = TrainingState(session.fast_weights, is_open=session.is_open)
    save_checkpoint(path, session.model, state, None, config, session.encoder.tokenizer,
                    manifest, manifest_digest, mode="session", extra={"observations": session.observations})
