import pytest
from terminal_world.config import RunConfig


def test_qwen_experiment_is_complete_and_uses_quarter_suffix():
    config = RunConfig.load("configs/experiment/terminal_ttt.yaml")
    assert config.model.model_id == "Qwen/Qwen2.5-Coder-3B"
    assert len(config.model.revision) == 40
    assert config.model.adaptive_layers / config.model.num_layers == 0.25
    assert config.model.qkv_bias and config.model.tie_embeddings
    assert config.model.adaptive_layers * 3 * config.model.hidden_size * config.model.intermediate_size == 608698368


def test_legacy_config_defaults_to_bias_free():
    config = RunConfig.load("configs/experiment/tiny.yaml")
    data = config.to_dict()
    del data["model"]["qkv_bias"]
    assert not RunConfig.from_dict(data).model.qkv_bias
    data["model"]["qkv_bias"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        RunConfig.from_dict(data)


def test_tiny_config_valid():
    config = RunConfig.load("configs/experiment/tiny.yaml")
    assert config.training.max_output_tokens == 2048
    assert config.model.attention_backend == "math"
