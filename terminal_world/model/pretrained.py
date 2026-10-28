"""One-time import of a standard Hugging Face Llama checkpoint.

Architectures/position encodings we have not implemented fail explicitly. Official
TTT checkpoints require a separate parameter converter and are not treated as Llama.
"""
from dataclasses import replace
import re
import torch
from terminal_world.dataloader.encoding import HFTokenizer, PROTOCOL
from terminal_world.model.transformer import create_model


def import_llama(reference, options, *, device="cpu"):
    c = reference.config
    if c.model_type != "llama" or c.hidden_act != "silu":
        raise ValueError("only standard Llama/SwiGLU checkpoints are supported")
    scaling = getattr(c, "rope_scaling", None)
    if scaling and scaling.get("rope_type", scaling.get("type", "default")) != "default":
        raise ValueError("scaled RoPE requires a dedicated adapter; cannot silently change positions")
    if getattr(c, "attention_bias", False) or getattr(c, "mlp_bias", False):
        raise ValueError("biased Llama projections are not supported")
    if getattr(c, "head_dim", c.hidden_size // c.num_attention_heads) != c.hidden_size // c.num_attention_heads:
        raise ValueError("nonstandard head dimension")
    if options.max_seq_len > c.max_position_embeddings:
        raise ValueError("requested context exceeds pretrained capacity")
    config = replace(options, vocab_size=c.vocab_size + len(PROTOCOL), hidden_size=c.hidden_size,
                     intermediate_size=c.intermediate_size, num_layers=c.num_hidden_layers,
                     num_heads=c.num_attention_heads, num_kv_heads=c.num_key_value_heads,
                     rms_norm_eps=c.rms_norm_eps, rope_theta=c.rope_theta,
                     tie_embeddings=c.tie_word_embeddings)
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
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(options.model_id, revision=options.revision,
                                              use_fast=True, trust_remote_code=False)
    if not tokenizer.is_fast or len(tokenizer) > reference.config.vocab_size:
        raise ValueError("fast tokenizer with vocabulary matching model required")
    model, fast, config = import_llama(reference, options, device=device)
    base_vocab = reference.config.vocab_size
    encoded = HFTokenizer(tokenizer.backend_tokenizer.to_str(),
                          dict(zip(PROTOCOL, range(base_vocab, base_vocab + len(PROTOCOL)))),
                          config.vocab_size, tokenizer.bos_token_id if getattr(tokenizer, "add_bos_token", False) else None)
    # Verify initial adapter preserves old-vocabulary logits before allowing training.
    probe = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        expected = reference(probe, use_cache=False).logits.float().cpu()
        actual = model(probe.to(device), fast)[..., :base_vocab].float().cpu()
    tolerance = 0.04 if options.dtype in ("bfloat16", "float16") else 2e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance,
                               msg="imported adapter did not preserve base logits")
    return model, fast, encoded, config
