import pytest
import torch
from terminal_world.config import ModelConfig, RunConfig, TrainConfig
from terminal_world.dataloader.encoding import ByteTokenizer, Encoder
from terminal_world.model.transformer import create_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(123)


@pytest.fixture
def tiny():
    config = RunConfig(ModelConfig(hidden_size=8, intermediate_size=12, num_heads=2, num_kv_heads=1,
                                   max_seq_len=128, dtype="float64"),
                       TrainConfig(inner_lr=0.2, outer_lr=0.001, unroll_events=4, prefix_batch_size=4, max_output_tokens=8))
    model, fast = create_model(config.model)
    encoder = Encoder(ByteTokenizer(), 8, 128)
    return model, fast, encoder, config
