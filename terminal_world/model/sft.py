"""Ordinary parallel, right-padded SFT batches; no fast-weight updates."""
import torch
from torch.nn import functional as F

from terminal_world.model.meta_model import ensure_finite


def batch_loss(model, steps, *, pad_id):
    if model.config.adaptive_layers:
        raise ValueError("SFT requires adaptive_layers=0")
    if not steps:
        raise ValueError("empty SFT batch")
    device = next(model.parameters()).device
    steps = [step.to(device) for step in steps]
    shape = (len(steps), max(step.length for step in steps))
    ids = torch.full(shape, pad_id, dtype=torch.long, device=device)
    targets = torch.zeros(shape, dtype=torch.long, device=device)
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    for row, step in enumerate(steps):
        ids[row, :step.length] = step.input_ids
        targets[row, :step.length] = step.target_tokens
        mask[row, :step.length] = step.loss_masks & step.attention_mask
    counts = mask.sum(-1)
    if (counts == 0).any():
        raise ValueError("each SFT sample needs a target")
    positions = torch.arange(shape[1], device=device).expand_as(ids)
    # One transformer pass for the entire batch. Right padding is strictly
    # after real tokens, so causal attention cannot expose it to real queries.
    features = model.prefix(ids, positions)
    # Project only supervised positions, avoiding vocabulary-sized logits for
    # historical context and padding. All supervised tokens remain parallel.
    logits = model.lm_head(model.norm(features[mask]))
    work = logits if logits.dtype == torch.float64 else logits.float()
    losses = F.cross_entropy(work, targets[mask], reduction="none")
    loss = losses.mean()
    ensure_finite([loss], "SFT loss")
    metrics, offset = [], 0
    for step, count in zip(steps, counts.tolist(), strict=True):
        metrics.append({"kind": step.kind, "nll_sum": float(losses[offset:offset + count].detach().sum()),
                        "tokens": count, **step.metadata, "update_norm": 0.0, "clipped": False})
        offset += count
    return loss, metrics


def train_batch(model, optimizer, state, batch, *, pad_id):
    model.validate_fast(state.fast_weights)
    optimizer.zero_grad(set_to_none=True)
    loss, metrics = batch_loss(model, batch.steps, pad_id=pad_id)
    loss.backward()
    ensure_finite([p.grad for p in model.parameters()], "SFT gradients")
    optimizer.step()
    ensure_finite(model.parameters(), "SFT parameters; restart from the last valid checkpoint")
    for values in optimizer.state.values():
        ensure_finite([v for v in values.values() if torch.is_tensor(v)], "optimizer state")
    state.next_cursor, state.is_open = batch.next_cursor, batch.is_open
    state.windows_completed += 1  # Existing checkpoint format counts optimizer steps.
    return float(loss.detach()), metrics
