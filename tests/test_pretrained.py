from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from terminal_world.model.pretrained import import_llama
from terminal_world.model.transformer import TerminalTransformer
from terminal_world.config import ModelConfig


def test_import_mapping_and_old_vocabulary_logits():
    config = ModelConfig(vocab_size=30, hidden_size=8, intermediate_size=12, num_heads=2, num_kv_heads=1,
                         max_seq_len=128, dtype="float64", tie_embeddings=True)
    base = TerminalTransformer(config)
    state = {("model." + k if not k.startswith("lm_head.") else k): v
             for k, v in base.state_dict().items() if ".prime_norm." not in k}
    hf_config = SimpleNamespace(model_type="llama", hidden_act="silu", rope_scaling=None,
                                hidden_size=8, intermediate_size=12, num_attention_heads=2, num_key_value_heads=1,
                                num_hidden_layers=2, vocab_size=30, rms_norm_eps=1e-6, rope_theta=10000.,
                                max_position_embeddings=128, tie_word_embeddings=True)
    reference = SimpleNamespace(config=hf_config, state_dict=lambda: state)
    imported, fast, actual_config = import_llama(reference, config)
    ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(imported(ids, fast)[..., :30], base(ids, None), atol=1e-12, rtol=1e-12)
    assert imported.embed_tokens.weight is imported.lm_head.weight
    assert actual_config.vocab_size == 36
    hf_config.rope_scaling = {"rope_type": "llama3"}
    with pytest.raises(ValueError, match="scaled RoPE"):
        import_llama(reference, config)


def test_real_transformers_adapter_when_available():
    transformers = pytest.importorskip("transformers")
    hf_config = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                        max_position_embeddings=128)
    reference = transformers.LlamaForCausalLM(hf_config).double().eval()
    reference.config._attn_implementation = "eager"
    model, fast, _ = import_llama(reference, ModelConfig(dtype="float64", max_seq_len=128))
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = reference(ids, use_cache=False).logits
        actual = model(ids, fast)[..., :32]
    # HF RoPE/RMS accumulate in FP32 even with a float64 model.
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
