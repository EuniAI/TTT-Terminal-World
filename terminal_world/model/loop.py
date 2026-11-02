from dataclasses import dataclass
import torch
from terminal_world.model.meta_model import detach_fast, ensure_finite, loss_for_window, step_loss


@dataclass
class TrainingState:
    fast_weights: dict
    next_cursor: int = 0
    epoch: int = 0
    is_open: bool = False
    windows_completed: int = 0


def train_window(model, optimizer, state, window, config, *, pad_id=256):
    device = next(model.parameters()).device
    steps = [s.to(device) for s in window.steps]
    optimizer.zero_grad(set_to_none=True)
    parameters = tuple(model.parameters())
    if model.config.adaptive_layers:
        loss, next_fast, metrics = loss_for_window(
            model, state.fast_weights, steps, alpha=config.inner_lr,
            prefix_batch_size=config.prefix_batch_size, pad_id=pad_id, clip_norm=config.inner_clip_norm,
        )
        grads = torch.autograd.grad(loss, parameters)
        for parameter, grad in zip(parameters, grads, strict=True):
            parameter.grad = grad
    else:
        # Ordinary SFT: each sample's graph is freed after backward. Larger
        # batches accumulate token-normalized gradients, without any unroll.
        model.validate_fast(state.fast_weights)
        total_tokens = sum(int(s.loss_masks.sum()) for s in steps)
        loss, next_fast, metrics = torch.zeros((), device=device), {}, []
        for step in steps:
            nll, n = step_loss(model, {}, step)
            ensure_finite([nll], "SFT loss")
            (nll / total_tokens).backward()
            loss = loss + nll.detach() / total_tokens
            metrics.append({"kind": step.kind, "nll_sum": float(nll.detach()), "tokens": n,
                            **step.metadata, "update_norm": 0.0, "clipped": False})
    ensure_finite([p.grad for p in parameters], "training gradients")
    optimizer.step()
    # A failing optimizer update makes this in-memory instance unusable. Never checkpoint it.
    ensure_finite(parameters, "outer parameters; restart from the last valid checkpoint")
    for values in optimizer.state.values():
        ensure_finite([v for v in values.values() if torch.is_tensor(v)], "optimizer state")
    state.fast_weights = detach_fast(next_fast)
    state.next_cursor, state.is_open = window.next_cursor, window.is_open
    state.windows_completed += 1
    return float(loss.detach()), metrics
