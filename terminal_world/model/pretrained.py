"""One-time import of standard Hugging Face Llama and Qwen2 checkpoints.

Architectures/position encodings we have not implemented fail explicitly. Official
TTT checkpoints require a separate parameter converter and are not treated as Llama.
"""
from dataclasses import replace
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


def load_pretrained(options, *, device="cpu"):
    if not re.fullmatch(r"[0-9a-f]{40}", options.revision):
        raise ValueError("pretrained revision must be a full commit SHA")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    reference = AutoModelForCausalLM.from_pretrained(
        options.model_id, revision=options.revision, torch_dtype=options.torch_dtype,
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
    for text in ("pwd\n", "printf 'hello\\n' | cat\nhello\n", "文件.txt\t  error: not found\n"):
        ids = encoded.encode_text(text)
        if ids != tokenizer.encode(text, add_special_tokens=False):
            raise ValueError("serialized tokenizer changed ordinary-text encoding")
        probe = torch.tensor([ids[:config.max_seq_len]], dtype=torch.long, device=device)
        with torch.no_grad():
            expected = reference(probe, use_cache=False).logits.float().cpu()
            actual = model(probe, fast)[..., :base_vocab].float().cpu()
        torch.testing.assert_close(
            actual, expected, atol=tolerance, rtol=tolerance,
            msg=lambda details: (
                "imported adapter did not preserve base logits\n"
                f"probe={text!r}, tokens={probe.shape[1]}, dtype={options.dtype}, "
                f"reference_device={next(reference.parameters()).device}, "
                f"adapter_device={next(model.parameters()).device}, "
                f"adapter_backend={config.attention_backend}, "
                f"torch={torch.__version__}, cuda={torch.version.cuda}\n"
                f"{details}"
            ),
        )
    return model, fast, encoded, config
