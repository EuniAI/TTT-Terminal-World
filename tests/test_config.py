import pytest
from terminal_world.config import RunConfig


def test_formal_template_fails_until_explicitly_configured():
    with pytest.raises(ValueError, match="incomplete"):
        RunConfig.load("configs/experiment/terminal_ttt.yaml")


def test_tiny_config_valid():
    config = RunConfig.load("configs/experiment/tiny.yaml")
    assert config.training.max_output_tokens == 2048
    assert config.model.attention_backend == "math"
