from dataclasses import replace
import pytest
import torch
from terminal_world.config import ModelConfig
from terminal_world.model.attention import causal_attention, validate_flashbob
from terminal_world.model.transformer import create_model
from terminal_world.model.meta_model import encode_prefix_batch, step_logits, loss_for_window
from terminal_world.dataloader.encoding import ByteTokenizer, Encoder
from terminal_world.model.data import Event


def test_flashbob_does_not_silently_fall_back(tiny):
    with pytest.raises(ValueError, match="CUDA"):
        create_model(replace(tiny[3].model, attention_backend="flashbob"))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_flashbob_operator_and_window(kv_heads):
    # BF16 acceptance tolerances: 3% activations, 10% derivatives plus small absolute floor.
    # This test intentionally fails rather than skipping unsupported software on a CUDA host.
    fn = validate_flashbob("cuda", torch.bfloat16, 32, 2, kv_heads)
    q = torch.randn(1, 2, 17, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, kv_heads, 17, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    results = []
    probe = torch.randn_like(q)
    directions = [torch.randn_like(t) for t in (q, k, v)]
    for backend in ("math", "flashbob"):
        out = causal_attention(q, k, v, backend, fn)
        grads = torch.autograd.grad((out * probe).float().sum(), (q, k, v), create_graph=True)
        hessian_vector = torch.autograd.grad(sum((g * d).float().sum() for g, d in zip(grads, directions)), (q, k, v))
        results.append((out, grads, hessian_vector))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0.03, rtol=0.03)
    for a, b in zip(results[0][1] + results[0][2], results[1][1] + results[1][2]):
        torch.testing.assert_close(a, b, atol=0.08, rtol=0.1)

    config = ModelConfig(hidden_size=64, intermediate_size=96, num_heads=2, num_kv_heads=kv_heads,
                         dtype="bfloat16", max_seq_len=128)
    math_model, math_fast = create_model(config, "cuda")
    flash_model, flash_fast = create_model(replace(config, attention_backend="flashbob"), "cuda")
    flash_model.load_state_dict(math_model.state_dict())
    flash_fast = {k: v.detach().clone().requires_grad_(True) for k, v in math_fast.items()}
    enc = Encoder(ByteTokenizer(), 8, 128)
    events = [Event("control", "open"), Event("shell", "long command", "abc"), Event("control", "close"),
              Event("control", "open"), Event("shell", "x", ""), Event("control", "close")]
    steps = [enc.encode(e).to("cuda") for e in events]
    outputs = []
    for model, fast in ((math_model, math_fast), (flash_model, flash_fast)):
        features = encode_prefix_batch(model, steps, 3)
        logits = [step_logits(model, fast, s, f).detach() for s, f in zip(steps, features)]
        loss, final_fast, metrics = loss_for_window(model, fast, steps, alpha=0.03, prefix_batch_size=3)
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        outputs.append((logits, loss, final_fast, metrics, gradients))
    a, b = outputs
    for x, y in zip(a[0], b[0]):
        torch.testing.assert_close(x, y, atol=0.04, rtol=0.03)
    torch.testing.assert_close(a[1], b[1], atol=0.03, rtol=0.03)
    for x, y in zip(a[4], b[4]):
        torch.testing.assert_close(x, y, atol=0.003, rtol=0.1)
    for name in a[2]:
        torch.testing.assert_close(a[2][name], b[2][name], atol=0.003, rtol=0.1)
    for x, y in zip(a[3], b[3]):
        assert x["nll_sum"] == pytest.approx(y["nll_sum"], rel=0.03, abs=0.03)
