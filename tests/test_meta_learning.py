import copy
import pytest
import torch
from terminal_world.model.data import Event
from terminal_world.model.meta_model import encode_prefix_batch, step_logits, loss_for_window, inner_loop_step, detach_fast


def sample_steps(encoder):
    return [encoder.encode(e) for e in [Event("control", "open"), Event("shell", "x", "a"),
                                       Event("shell", "longer", ""), Event("control", "close"),
                                       Event("control", "open"), Event("shell", "z", "bc")]]


@pytest.mark.parametrize("batch_size", [1, 2, 6])
def test_prefix_batch_matches_serial_entire_unroll(tiny, batch_size):
    model, fast, encoder, config = tiny
    steps = sample_steps(encoder)
    serial_loss, serial_fast, serial_metrics = loss_for_window(model, fast, steps, alpha=0.2, serial=True)
    serial_grads = torch.autograd.grad(serial_loss, tuple(model.parameters()))
    batch_loss, batch_fast, batch_metrics = loss_for_window(model, fast, steps, alpha=0.2, prefix_batch_size=batch_size)
    batch_grads = torch.autograd.grad(batch_loss, tuple(model.parameters()))
    torch.testing.assert_close(batch_loss, serial_loss, rtol=1e-11, atol=1e-12)
    for a, b in zip(batch_grads, serial_grads):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)
    for name in fast:
        torch.testing.assert_close(batch_fast[name], serial_fast[name], rtol=1e-10, atol=1e-12)
    assert [r["nll_sum"] for r in batch_metrics] == pytest.approx([r["nll_sum"] for r in serial_metrics], abs=1e-11)
    assert model.embed_tokens.weight.grad is None
    assert serial_grads[0].norm() > 0


def test_prefix_logits_padding_causality_and_event_isolation(tiny):
    model, fast, encoder, _ = tiny
    steps = sample_steps(encoder)
    features = encode_prefix_batch(model, steps, 6)
    for step, feature in zip(steps, features):
        torch.testing.assert_close(step_logits(model, fast, step, feature), step_logits(model, fast, step))
    original = model(steps[1].input_ids[None], fast)
    changed = steps[1].input_ids.clone()
    changed[-1] = (changed[-1] + 1) % 256
    modified = model(changed[None], fast)
    torch.testing.assert_close(original[:, :-1], modified[:, :-1])
    steps[-1] = encoder.encode(Event("shell", "new completely different command", "other"))
    new_features = encode_prefix_batch(model, steps, 6)
    for old, new in zip(features[:-1], new_features[:-1]):
        torch.testing.assert_close(old, new)


def test_meta_gradient_finite_difference_and_inner_path(tiny):
    model, fast, encoder, _ = tiny
    steps = sample_steps(encoder)
    loss, _, _ = loss_for_window(model, fast, steps, alpha=0.2, prefix_batch_size=3)
    parameter = model.layers[0].self_attn.q_proj.weight
    gradient = torch.autograd.grad(loss, parameter)[0]
    direction = torch.randn_like(parameter)
    direction /= direction.norm()
    expected = (gradient * direction).sum()
    original = parameter.detach().clone()
    def value(delta):
        with torch.no_grad():
            parameter.copy_(original + delta * direction)
        result, _, _ = loss_for_window(model, fast, steps, alpha=0.2, prefix_batch_size=2)
        return result.detach()
    eps = 1e-5
    numerical = (value(eps) - value(-eps)) / (2 * eps)
    with torch.no_grad():
        parameter.copy_(original)
    torch.testing.assert_close(expected, numerical, atol=2e-8, rtol=2e-4)
    # The final loss must depend on the update learning rate, proving a nonzero adaptation path.
    alpha = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    phi = fast
    from terminal_world.model.meta_model import step_loss
    for step in steps:
        nll, n = step_loss(model, phi, step)
        phi = inner_loop_step(phi, nll / n, alpha=alpha, create_graph=True)
    assert abs(float(torch.autograd.grad(nll, alpha)[0])) > 1e-8


def test_storage_initialization_and_truncation(tiny):
    model, fast, encoder, _ = tiny
    model.validate_fast(fast)
    assert all(torch.count_nonzero(v) == 0 for k, v in fast.items() if "down_proj" in k)
    assert all(torch.count_nonzero(v) > 0 for k, v in fast.items() if "down_proj" not in k)
    ids = encoder.encode(Event("shell", "x", "a")).input_ids[None]
    torch.testing.assert_close(model(ids, fast), model(ids, None))
    _, updated, _ = loss_for_window(model, fast, sample_steps(encoder), alpha=0.2, prefix_batch_size=4)
    detached = detach_fast(updated)
    for key in updated:
        torch.testing.assert_close(detached[key], updated[key], rtol=0, atol=0)
        assert detached[key].is_leaf and detached[key].requires_grad
    assert any(not torch.equal(updated[k], fast[k]) for k in fast)


def test_differentiable_clipping():
    theta = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    phi = {"x": torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)}
    next_phi = inner_loop_step(phi, (phi["x"] * theta).square().sum(), alpha=0.1, create_graph=True, clip_norm=0.5)
    assert torch.isfinite(torch.autograd.grad(next_phi["x"].sum(), theta)[0])
