"""Cached generation must preserve causal positions and event isolation."""
import pytest
import torch
from terminal_world.model.data import Event
from terminal_world.session import Session


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_cached_logits_match_full_forward(tiny, chunk_size):
    model, fast, _, _ = tiny
    # Exercise nonzero adaptive weights, not just the initial zero down projection.
    with torch.no_grad():
        for name, value in fast.items():
            if "down_proj" in name:
                value.normal_(std=0.1)
    ids = torch.randint(0, model.config.vocab_size, (2, 13))
    with torch.no_grad():
        logits, cache = model.forward_cached(ids[:, :4], fast)
        torch.testing.assert_close(logits, model(ids[:, :4], fast)[:, -1:])
        for start in range(4, ids.shape[1], chunk_size):
            end = min(start + chunk_size, ids.shape[1])
            logits, cache = model.forward_cached(ids[:, start:end], fast, cache)
            torch.testing.assert_close(logits, model(ids[:, :end], fast)[:, -1:],
                                       rtol=1e-10, atol=1e-11)
            assert all(c["k"].shape[-2] == end for c in cache)
            assert all(c["k"].shape[1] == model.config.num_kv_heads for c in cache)
            assert all(not c["k"].requires_grad for c in cache)


@torch.no_grad()
def uncached_tokens(session, command):
    prompt = session.encoder.prompt(Event("shell", command))
    ids = torch.tensor([prompt], device=next(session.model.parameters()).device)
    generated = []
    end = session.encoder.tokenizer.ids["obs_end"]
    forbidden = [v for k, v in session.encoder.tokenizer.ids.items() if k != "obs_end"]
    for _ in range(session.encoder.max_output_tokens):
        logits = session.model(ids, session.fast_weights)[:, -1]
        logits[:, forbidden] = -torch.inf
        token = int(logits.argmax(-1))
        generated.append(token)
        if token == end:
            break
        ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
    return generated


def test_session_cache_is_local_and_only_decodes_one_token(tiny):
    model, fast, encoder, config = tiny
    session = Session(model, fast, encoder, alpha=config.training.inner_lr)
    session.control("open")
    for command in ("x", "longer command"):
        expected = uncached_tokens(session, command)
        lengths = []
        hook = model.embed_tokens.register_forward_pre_hook(
            lambda module, args: lengths.append(args[0].shape[1]))
        before = {k: v.clone() for k, v in session.fast_weights.items()}
        try:
            assert session.generate_tokens(command) == expected
        finally:
            hook.remove()
        assert lengths[0] == len(encoder.prompt(Event("shell", command)))
        assert lengths[1:] == [1] * (len(expected) - 1)
        assert all(torch.equal(v, session.fast_weights[k]) for k, v in before.items())
        session.observe(command, "real output")
    session.control("close")
    session.control("open")
    assert session.generate_tokens("new") == uncached_tokens(session, "new")


def test_cached_capacity_limit(tiny):
    model, fast, _, _ = tiny
    ids = torch.zeros((1, model.config.max_seq_len), dtype=torch.long)
    _, cache = model.forward_cached(ids, fast)
    with pytest.raises(ValueError, match="capacity"):
        model.forward_cached(ids[:, :1], fast, cache)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_cached_logits_low_precision(dtype):
    from terminal_world.config import ModelConfig
    from terminal_world.model.transformer import create_model
    model, fast = create_model(ModelConfig(hidden_size=32, intermediate_size=48,
                                           num_heads=4, num_kv_heads=2, dtype=dtype))
    ids = torch.randint(0, model.config.vocab_size, (1, 12))
    _, cache = model.forward_cached(ids[:, :7], fast)
    with torch.no_grad():
        for end in range(8, 13):
            actual, cache = model.forward_cached(ids[:, end-1:end], fast, cache)
            expected = model(ids[:, :end], fast)[:, -1:]
            tol = 0.03 if dtype == "bfloat16" else 1e-5
            torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_flashbob_cached_decode():
    from terminal_world.config import ModelConfig
    from terminal_world.model.transformer import create_model
    model, fast = create_model(ModelConfig(hidden_size=64, intermediate_size=96,
                                           num_heads=2, num_kv_heads=1, qkv_bias=True,
                                           dtype="bfloat16", attention_backend="flashbob"), "cuda")
    ids = torch.randint(0, model.config.vocab_size, (1, 17), device="cuda")
    _, cache = model.forward_cached(ids[:, :9], fast)
    with torch.no_grad():
        for end in range(10, 18):
            actual, cache = model.forward_cached(ids[:, end-1:end], fast, cache)
            torch.testing.assert_close(actual, model(ids[:, :end], fast)[:, -1:],
                                       atol=0.04, rtol=0.03)
