from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from terminal_world.model.pretrained import import_llama, import_qwen2
from terminal_world.model.transformer import TerminalTransformer
from terminal_world.config import ModelConfig


def test_backend_metrics_distribution_and_next_token_loss():
    from terminal_world.model.pretrained import _backend_metrics
    baseline = torch.tensor([[[2., 0.], [0., 2.], [1., 1.]]])
    ids = torch.tensor([[0, 1, 0]])
    same = _backend_metrics(baseline, baseline, ids)
    assert same["kl_baseline_to_backend"] == 0
    assert same["mean_total_variation"] == 0
    assert same["top1_agreement"] == 1
    assert same["nll_delta"] == 0
    assert same["nll_baseline"] == pytest.approx(torch.log1p(torch.exp(torch.tensor(2.))).item())
    # A constant logit offset changes raw logits, but not probabilities or loss.
    shifted = _backend_metrics(baseline + 1, baseline, ids)
    assert shifted["max_abs"] == 1
    assert shifted["mean_total_variation"] == pytest.approx(0, abs=1e-7)
    assert shifted["nll_delta"] == pytest.approx(0, abs=1e-7)
    changed = _backend_metrics(-baseline, baseline, ids)
    assert changed["kl_baseline_to_backend"] > 0
    assert changed["mean_total_variation"] > 0
    assert changed["nll_delta"] < 0
    assert _backend_metrics(baseline[:, :1], baseline[:, :1], ids[:, :1])["nll_delta"] is None
    with pytest.raises(FloatingPointError, match="non-finite"):
        _backend_metrics(baseline * float("nan"), baseline, ids)


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_matched_attention_alignment_detects_corruption_and_restores(kind, dtype):
    transformers = pytest.importorskip("transformers")
    from terminal_world.model.pretrained import _hf_eager_alignment
    config_cls = transformers.Qwen2Config if kind == "qwen2" else transformers.LlamaConfig
    model_cls = transformers.Qwen2ForCausalLM if kind == "qwen2" else transformers.LlamaForCausalLM
    config = config_cls(vocab_size=64, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=128)
    config._attn_implementation = "eager"
    reference = model_cls(config).to(dtype=getattr(torch, dtype)).eval()
    importer = import_qwen2 if kind == "qwen2" else import_llama
    model, fast, _ = importer(reference, ModelConfig(dtype=dtype, max_seq_len=128))
    ids = torch.arange(32).unsqueeze(0)
    with torch.no_grad():
        expected = reference(ids, use_cache=False).logits
        with _hf_eager_alignment(model, reference):
            actual = model(ids, fast)[..., :64]
        tolerance = 0.04 if dtype == "bfloat16" else 2e-5
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        assert all(not hasattr(layer.self_attn, "_alignment_attention") for layer in model.layers)
        with pytest.raises(AssertionError):
            with _hf_eager_alignment(model, reference):
                model.lm_head.weight.zero_()
                corrupted = model(ids, fast)[..., :64]
                torch.testing.assert_close(corrupted, expected, atol=tolerance, rtol=tolerance)
        assert all(not hasattr(layer.self_attn, "_alignment_attention") for layer in model.layers)
        assert model.config.attention_backend == "math"


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


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_real_transformers_adapter_when_available(kind):
    transformers = pytest.importorskip("transformers")
    config_cls = transformers.LlamaConfig if kind == "llama" else transformers.Qwen2Config
    model_cls = transformers.LlamaForCausalLM if kind == "llama" else transformers.Qwen2ForCausalLM
    importer = import_llama if kind == "llama" else import_qwen2
    hf_config = config_cls(vocab_size=32, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                        max_position_embeddings=128)
    hf_config._attn_implementation = "eager"
    reference = model_cls(hf_config).double().eval()
    # HF initializes biases to zero: make them nonzero to detect omitted/miswired bias.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith(".bias"):
                parameter.uniform_(-0.2, 0.2)
    model, fast, _ = importer(reference, ModelConfig(dtype="float64", max_seq_len=128))
    ids = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]])
    with torch.no_grad():
        expected = reference(ids, use_cache=False).logits
        actual = model(ids, fast)[..., :32]
    # HF RoPE/RMS accumulate in FP32 even with a float64 model.
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def qwen_reference():
    config = ModelConfig(vocab_size=30, hidden_size=8, intermediate_size=12, num_heads=2,
                         num_kv_heads=1, max_seq_len=128, dtype="float64",
                         tie_embeddings=True, qkv_bias=True)
    base = TerminalTransformer(config)
    state = {("model." + k if not k.startswith("lm_head.") else k): v
             for k, v in base.state_dict().items() if ".prime_norm." not in k}
    hf_config = SimpleNamespace(model_type="qwen2", hidden_act="silu", rope_scaling=None,
                               hidden_size=8, intermediate_size=12, num_attention_heads=2,
                               num_key_value_heads=1, num_hidden_layers=2, vocab_size=30,
                               rms_norm_eps=1e-6, rope_theta=10000., max_position_embeddings=128,
                               tie_word_embeddings=True, use_sliding_window=False)
    return base, SimpleNamespace(config=hf_config, state_dict=lambda: state)


def test_qwen_bias_import_and_strict_mapping():
    base, reference = qwen_reference()
    model, fast, config = import_qwen2(reference, ModelConfig(dtype="float64", max_seq_len=128))
    assert config.qkv_bias
    assert model.embed_tokens.weight is model.lm_head.weight
    for i, layer in enumerate(model.layers):
        assert layer.self_attn.o_proj.bias is None
        for name in ("q_proj", "k_proj", "v_proj"):
            bias = getattr(layer.self_attn, name).bias
            assert torch.count_nonzero(bias) > 0
            torch.testing.assert_close(bias, getattr(base.layers[i].self_attn, name).bias)
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = base(ids, None)
    torch.testing.assert_close(model(ids, fast)[..., :30], expected, atol=1e-12, rtol=1e-12)
    with torch.no_grad():
        for layer in model.layers:
            layer.self_attn.v_proj.bias.zero_()
    assert not torch.allclose(model(ids, fast)[..., :30], expected)
    del reference.state_dict()["model.layers.0.self_attn.q_proj.bias"]
    with pytest.raises(ValueError, match="missing pretrained parameter"):
        import_qwen2(reference, ModelConfig(max_seq_len=128))


@pytest.mark.parametrize("field,value,match", [
    ("use_sliding_window", True, "sliding-window"),
    ("rope_scaling", {"rope_type": "yarn"}, "scaled RoPE"),
    ("hidden_act", "gelu", "SwiGLU"),
])
def test_qwen_unsupported_variants_rejected(field, value, match):
    _, reference = qwen_reference()
    setattr(reference.config, field, value)
    with pytest.raises(ValueError, match=match):
        import_qwen2(reference, ModelConfig(max_seq_len=128))


def test_qwen_local_load_tokenizer_and_session_roundtrip(tmp_path, capsys):
    """Exercise AutoModel dispatch and tokenizer serialization without downloading weights."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from terminal_world.model.pretrained import load_pretrained
    from terminal_world.config import RunConfig, TrainConfig
    from terminal_world.dataloader.encoding import Encoder, tokenizer_from_state
    from terminal_world.checkpoint import save_session, resume_session
    from terminal_world.session import Session

    texts = ["pwd\n", "printf 'hello\\n' | cat\nhello\n", "文件.txt\t  error: not found\n",
             "", "  a\t b\n\n", "<|command|> <|im_start|>"]
    backend = tokenizers.Tokenizer(tokenizers.models.BPE())
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    backend.train_from_iterator(texts, tokenizers.trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<|im_start|>"],
    ))
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"], clean_up_tokenization_spaces=False,
    )
    source = tmp_path / "qwen"
    tokenizer.save_pretrained(source)
    hf_config = transformers.Qwen2Config(
        vocab_size=len(tokenizer) + 8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, tie_word_embeddings=True,
    )
    reference = transformers.Qwen2ForCausalLM(hf_config).eval()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith(".bias"):
                parameter.uniform_(-0.1, 0.1)
    reference.save_pretrained(source)
    model, fast, encoded, actual = load_pretrained(ModelConfig(
        model_id=str(source), revision="a" * 40, max_seq_len=128,
    ))
    import json
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    alignment = [r for r in reports if r["event"] == "pretrained_alignment"]
    comparisons = [r for r in reports if r["event"] == "pretrained_backend_comparison"]
    assert len(alignment) == len(comparisons) == 4
    assert all(r["status"] == "passed" for r in alignment)
    assert all(r["diagnostic_only"] and r["backend"] == "math" for r in comparisons)
    assert comparisons[-1]["tokens"] == 128
    assert all(r["nll_delta"] is not None for r in comparisons)
    assert all(not hasattr(layer.self_attn, "_alignment_attention") for layer in model.layers)
    assert actual.qkv_bias and actual.tie_embeddings
    restored = tokenizer_from_state(encoded.state_dict())
    for text in texts:
        ids = encoded.encode_text(text)
        assert not set(ids) & set(encoded.ids.values())
        assert restored.encode_text(text) == ids
        assert restored.decode(ids) == text
    encoder = Encoder(encoded, max_output_tokens=8, max_seq_len=128)
    session = Session(model, fast, encoder, alpha=0.01)
    session.control("open")
    session.observe("pwd", "/tmp\n")
    path = tmp_path / "session.pt"
    config = RunConfig(actual, TrainConfig(0.01, 1e-5, 4, 1, max_output_tokens=8))
    save_session(path, session, config)
    loaded = resume_session(path)
    assert loaded.model.config.qkv_bias
    assert loaded.observe("pwd", "/tmp\n") == session.observe("pwd", "/tmp\n")
