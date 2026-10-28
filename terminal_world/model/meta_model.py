import torch
from torch.nn import functional as F


def ensure_finite(tensors, label):
    for tensor in tensors:
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"nonfinite {label}; window was not committed")


def detach_fast(fast):
    return {name: value.detach().requires_grad_(True) for name, value in fast.items()}


def encode_prefix_batch(model, steps, batch_size, pad_id=256):
    if not steps or batch_size < 1:
        raise ValueError("nonempty steps and positive prefix batch required")
    outputs = []
    for start in range(0, len(steps), batch_size):
        group = steps[start:start + batch_size]
        max_len = max(s.length for s in group)
        ids = torch.full((len(group), max_len), pad_id, device=group[0].input_ids.device, dtype=torch.long)
        for i, step in enumerate(group):
            ids[i, :step.length] = step.input_ids
        positions = torch.arange(max_len, device=ids.device).expand_as(ids)
        features = model.prefix(ids, positions)
        outputs.extend(features[i:i + 1, :step.length] for i, step in enumerate(group))
    return outputs


def step_logits(model, fast, step, prefix_output=None):
    if prefix_output is None:
        return model(step.input_ids.unsqueeze(0), fast, step.position_ids.unsqueeze(0))[0]
    return model.suffix(prefix_output, step.position_ids.unsqueeze(0), fast)[0]


def step_loss(model, fast, step, *, prefix_output=None):
    logits = step_logits(model, fast, step, prefix_output)
    count = int(step.loss_masks.sum())
    if count < 1:
        raise ValueError("every event must have at least one target")
    work = logits if logits.dtype == torch.float64 else logits.float()
    loss = F.cross_entropy(work[step.loss_masks], step.target_tokens[step.loss_masks], reduction="sum")
    return loss, count


def inner_loop_step(fast, mean_loss, *, alpha, create_graph, clip_norm=None, return_stats=False):
    names, values = tuple(fast), tuple(fast.values())
    grads = torch.autograd.grad(mean_loss, values, create_graph=create_graph)
    ensure_finite(grads, "inner gradients")
    work_grads = [g if g.dtype == torch.float64 else g.float() for g in grads]
    squared_norm = sum(g.square().sum() for g in work_grads)
    # Avoid sqrt(0)'s singular derivative in the differentiable clipping path.
    norm = torch.sqrt(squared_norm + 1e-30)
    factor = (clip_norm / norm).clamp(max=1) if clip_norm is not None else 1.0
    result = {name: value - alpha * (grad * factor).to(value.dtype)
              for name, value, grad in zip(names, values, grads, strict=True)}
    ensure_finite(result.values(), "updated fast weights")
    if return_stats:
        return result, {"inner_grad_norm": float(norm.detach()),
                        "update_norm": float((norm * factor * alpha).detach()),
                        "clipped": clip_norm is not None and float(norm.detach()) > clip_norm}
    return result


def loss_for_window(model, fast, steps, *, alpha, prefix_batch_size=1,
                    pad_id=256, clip_norm=None, serial=False):
    if not steps:
        raise ValueError("empty window")
    prefix = [None] * len(steps) if serial else encode_prefix_batch(model, steps, prefix_batch_size, pad_id)
    sums, counts, metrics = [], 0, []
    for step, features in zip(steps, prefix, strict=True):
        nll, n = step_loss(model, fast, step, prefix_output=features)
        ensure_finite([nll], "event loss")
        fast, stats = inner_loop_step(fast, nll / n, alpha=alpha, create_graph=True,
                                      clip_norm=clip_norm, return_stats=True)
        sums.append(nll)
        counts += n
        metrics.append({"kind": step.kind, "nll_sum": float(nll.detach()), "tokens": n,
                        **step.metadata, **stats})
    return torch.stack(sums).sum() / counts, fast, metrics
