"""One-time import of standard Hugging Face Llama and Qwen2 checkpoints.

Architectures/position encodings we have not implemented fail explicitly. Official
TTT checkpoints require a separate parameter converter and are not treated as Llama.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import re
import torch
from terminal_world.dataloader.encoding import HFTokenizer, PROTOCOL
from terminal_world.model.transformer import create_model


def import_llama(reference, options, *, device="cpu"):
    return _import_decoder(reference, options, model_type="llama", device=device)


def import_qwen2(reference, options, *, device="cpu"):
    return _import_decoder(reference, options, model_type="qwen2", device=device)


def _import_decoder(reference, options, *, model_type, device):
    c = reference.config
    if c.model_type != model_type or c.hidden_act != "silu":
        raise ValueError(f"only standard {model_type}/SwiGLU checkpoints are supported")
    scaling = getattr(c, "rope_scaling", None)
    if scaling and scaling.get("rope_type", scaling.get("type", "default")) != "default":
        raise ValueError("scaled RoPE requires a dedicated adapter; cannot silently change positions")
    if getattr(c, "mlp_bias", False) or (model_type == "llama" and getattr(c, "attention_bias", False)):
        raise ValueError("biased Llama/MLP projections are not supported")
    if getattr(c, "use_sliding_window", False) or "sliding_attention" in (getattr(c, "layer_types", None) or []):
        raise ValueError("sliding-window checkpoints require a dedicated adapter")
    if getattr(c, "head_dim", c.hidden_size // c.num_attention_heads) != c.hidden_size // c.num_attention_heads:
        raise ValueError("nonstandard head dimension")
    if options.max_seq_len > c.max_position_embeddings:
        raise ValueError("requested context exceeds pretrained capacity")
    config = replace(options, vocab_size=c.vocab_size + len(PROTOCOL), hidden_size=c.hidden_size,
                     intermediate_size=c.intermediate_size, num_layers=c.num_hidden_layers,
                     num_heads=c.num_attention_heads, num_kv_heads=c.num_key_value_heads,
                     rms_norm_eps=c.rms_norm_eps, rope_theta=c.rope_theta,
                     tie_embeddings=c.tie_word_embeddings, qkv_bias=model_type == "qwen2")
    model, fast = create_model(config, device)
    source = reference.state_dict()
    consumed = set()
    with torch.no_grad():
        for name, target in model.state_dict().items():
            if ".prime_norm." in name:
                # New normalization belongs to theta; identity prime output preserves base logits.
                continue
            source_name = name if name.startswith("lm_head.") else "model." + name
            if source_name not in source:
                raise ValueError(f"missing pretrained parameter: {source_name}")
            value = source[source_name]
            consumed.add(source_name)
            if name in ("embed_tokens.weight", "lm_head.weight"):
                if tuple(value.shape) != (c.vocab_size, c.hidden_size):
                    raise ValueError("pretrained embedding shape mismatch")
                target[:c.vocab_size].copy_(value)
                # Conservative initialization for new protocol vocabulary rows.
                target[c.vocab_size:].copy_(value.float().mean(0).to(target))
            else:
                if target.shape != value.shape:
                    raise ValueError(f"pretrained shape mismatch: {name}")
                target.copy_(value)
    extra = set(source) - consumed
    if extra:
        raise ValueError(f"unsupported pretrained parameters: {sorted(extra)}")
    return model, fast, config


@contextmanager
def _hf_eager_alignment(model, reference):
    """Use HF's exact attention kernel only for unpadded import probes."""
    if reference.config.model_type == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
    else:
        from transformers.models.llama.modeling_llama import eager_attention_forward
    changed = []
    try:
        for layer, source in zip(model.layers, reference.model.layers, strict=True):
            attention = layer.self_attn
            previous = getattr(attention, "_alignment_attention", None)

            def eager(q, k, v, source_attention=source.self_attn):
                mask = torch.full((q.shape[-2], k.shape[-2]), torch.finfo(q.dtype).min,
                                  dtype=q.dtype, device=q.device).triu(1)
                output, _ = eager_attention_forward(
                    source_attention, q, k, v, mask[None, None], scaling=q.shape[-1] ** -0.5, dropout=0.0,
                )
                return output.transpose(1, 2)

            changed.append((attention, previous))
            attention._alignment_attention = eager
        yield
    finally:
        for attention, previous in changed:
            if previous is None:
                del attention._alignment_attention
            else:
                attention._alignment_attention = previous


def _backend_metrics(actual, baseline, ids):
    """Report drift on original-vocabulary logits, in bounded-memory chunks."""
    if not torch.isfinite(actual).all() or not torch.isfinite(baseline).all():
        raise FloatingPointError("non-finite logits during backend comparison")
    actual, baseline = actual.float(), baseline.float()
    delta = (actual - baseline).abs()
    result = {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
              "top1_agreement": (actual.argmax(-1) == baseline.argmax(-1)).float().mean().item()}
    kl, tv = 0.0, 0.0
    a, b = actual.flatten(0, -2), baseline.flatten(0, -2)
    for start in range(0, a.shape[0], 32):
        log_a = a[start:start + 32].log_softmax(-1)
        log_b = b[start:start + 32].log_softmax(-1)
        kl += (log_b.exp() * (log_b - log_a)).sum().item()
        tv += ((log_b.exp() - log_a.exp()).abs().sum(-1) * 0.5).sum().item()
    result.update(kl_baseline_to_backend=max(0.0, kl / a.shape[0]),
                  mean_total_variation=tv / a.shape[0])
    result.update(nll_baseline=None, nll_backend=None, nll_delta=None)
    if ids.shape[-1] > 1:
        targets = ids[..., 1:].cpu().reshape(-1)
        def nll(logits):
            return torch.nn.functional.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.shape[-1]), targets).item()
        result.update(nll_baseline=nll(baseline), nll_backend=nll(actual))
        result["nll_delta"] = result["nll_backend"] - result["nll_baseline"]
    return result


def load_pretrained(options, *, device="cpu"):
    if not re.fullmatch(r"[0-9a-f]{40}", options.revision):
        raise ValueError("pretrained revision must be a full commit SHA")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    reference = AutoModelForCausalLM.from_pretrained(
        options.model_id, revision=options.revision, dtype=options.torch_dtype,
        attn_implementation="eager", trust_remote_code=False,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(options.model_id, revision=options.revision,
                                              use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast or len(tokenizer) > reference.config.vocab_size:
        raise ValueError("fast tokenizer with vocabulary matching model required")
    importers = {"llama": import_llama, "qwen2": import_qwen2}
    model_type = reference.config.model_type
    if model_type not in importers:
        raise ValueError(f"unsupported pretrained architecture: {model_type}")
    model, fast, config = importers[model_type](reference, options, device=device)
    base_vocab = reference.config.vocab_size
    encoded = HFTokenizer(tokenizer.backend_tokenizer.to_str(),
                          dict(zip(PROTOCOL, range(base_vocab, base_vocab + len(PROTOCOL)))),
                          config.vocab_size, tokenizer.bos_token_id if getattr(tokenizer, "add_bos_token", False) else None)
    tolerance = 0.04 if options.dtype in ("bfloat16", "float16") else 2e-5
    # Probe ordinary terminal text at several lengths, without chat templates or protocol IDs.
    # Raw special-token spellings intentionally have different semantics in our encoder.
    probes = [("pwd\n", config.max_seq_len),
              ("printf 'hello\\n' | cat\nhello\n", config.max_seq_len),
              ("文件.txt\t  error: not found\n", config.max_seq_len),
              (("$ ls -l\n-rw-r--r-- 1 user user 42 notes.txt\n" * 32), min(128, config.max_seq_len))]
    for probe_index, (text, limit) in enumerate(probes):
        ids = encoded.encode_text(text)
        if ids != tokenizer.encode(text, add_special_tokens=False):
            raise ValueError("serialized tokenizer changed ordinary-text encoding")
        probe = torch.tensor([ids[:limit]], dtype=torch.long, device=device)
        with torch.no_grad():
            expected = reference(probe, use_cache=False).logits.float().cpu()
            with _hf_eager_alignment(model, reference):
                actual = model(probe, fast)[..., :base_vocab].float().cpu()
        torch.testing.assert_close(
            actual, expected, atol=tolerance, rtol=tolerance,
            msg=lambda details: (
                "imported adapter did not preserve base logits with matched HF eager attention\n"
                f"probe_index={probe_index}, tokens={probe.shape[1]}, dtype={options.dtype}, "
                f"reference_device={next(reference.parameters()).device}, "
                f"adapter_device={next(model.parameters()).device}, "
                f"adapter_backend={config.attention_backend}, "
                f"torch={torch.__version__}, cuda={torch.version.cuda}\n"
                f"{details}"
            ),
        )
        print(json.dumps({"event": "pretrained_alignment", "probe": probe_index,
                          "tokens": probe.shape[1], "attention": "hf_eager", "status": "passed"}), flush=True)
        with torch.no_grad():
            configured = model(probe, fast)[..., :base_vocab].float().cpu()
        metrics = _backend_metrics(configured, actual, probe)
        print(json.dumps({"event": "pretrained_backend_comparison", "probe": probe_index,
                          "tokens": probe.shape[1], "baseline": "hf_eager",
                          "backend": config.attention_backend, "diagnostic_only": True,
                          **metrics}, allow_nan=False), flush=True)
    return model, fast, encoded, config
