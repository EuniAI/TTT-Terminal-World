import pytest
import torch
from terminal_world.config import ModelConfig, RunConfig, TrainConfig
from terminal_world.model.transformer import create_model
from terminal_world.model.meta_model import loss_for_window
from terminal_world.model.loop import TrainingState, train_window
from terminal_world.dataloader.encoding import ByteTokenizer, Encoder
from terminal_world.dataloader.events import windows
from terminal_world.model.data import Event
from terminal_world.checkpoint import save_checkpoint, load_checkpoint


def setup_training():
    config = RunConfig(ModelConfig(hidden_size=16, intermediate_size=24, num_heads=2,
                                    num_kv_heads=1, max_seq_len=128, dtype='bfloat16'),
                       TrainConfig(.1, 5e-4, 4, 2, 8, 1.))
    model, fast = create_model(config.model)
    encoder = Encoder(ByteTokenizer(), 8, 128)
    events = [Event('control', 'open', cursor=0), Event('shell', 'x', 'a', cursor=1),
              Event('shell', 'y', 'b', cursor=2), Event('control', 'close', cursor=3)]
    window = next(windows(events, encoder, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.outer_lr, weight_decay=0)
    return config, model, TrainingState(fast), encoder, window, optimizer


def test_bf16_parameters_updates_and_meta_gradients():
    config, model, state, encoder, window, optimizer = setup_training()
    ids = window.steps[1].input_ids[None]
    assert model(ids, state.fast_weights).dtype == torch.bfloat16
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert all(p.dtype == torch.bfloat16 for p in state.fast_weights.values())
    # The inner update still participates in the meta-gradient graph.
    alpha = torch.tensor(.1, requires_grad=True)
    loss, _, _ = loss_for_window(model, state.fast_weights, window.steps, alpha=alpha)
    derivative = torch.autograd.grad(loss, alpha)[0]
    assert torch.isfinite(derivative) and derivative.abs() > 0
    before = [p.detach().clone() for p in model.parameters()]
    train_window(model, optimizer, state, window, config.training)
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))
    assert model.norm.weight.grad.dtype == torch.bfloat16
    for values in optimizer.state.values():
        assert values['exp_avg'].dtype == values['exp_avg_sq'].dtype == torch.bfloat16


@pytest.mark.parametrize('legacy', [False, True])
def test_precision_checkpoint_migration_and_resume(tmp_path, legacy):
    config, model, state, encoder, window, optimizer = setup_training()
    train_window(model, optimizer, state, window, config.training)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, model, state, optimizer, config, encoder.tokenizer, {}, 'digest')
    payload = torch.load(path, weights_only=False)
    if legacy:
        for field in ('theta', 'fast_weights'):
            payload[field] = {k: v.float() + 1e-6 for k, v in payload[field].items()}
        for values in payload['optimizer']['state'].values():
            for name in ('exp_avg', 'exp_avg_sq'):
                values[name] = values[name].float()
        torch.save(payload, path)
        with pytest.warns(UserWarning, match='Converting checkpoint'):
            restored = load_checkpoint(path)
    else:
        restored = load_checkpoint(path)
    m, s, opt, cfg, _, _ = restored
    assert s.windows_completed == state.windows_completed
    assert s.next_cursor == state.next_cursor
    assert all(p.dtype == torch.bfloat16 for p in m.parameters())
    for key, value in s.fast_weights.items():
        assert value.dtype == torch.bfloat16
        torch.testing.assert_close(value, payload['fast_weights'][key].bfloat16(), atol=0, rtol=0)
    for key, value in m.state_dict().items():
        torch.testing.assert_close(value, payload['theta'][key].bfloat16(), atol=0, rtol=0)
    for values in opt.state.values():
        assert values['exp_avg'].dtype == values['exp_avg_sq'].dtype == torch.bfloat16
    train_window(m, opt, s, window, cfg.training)
    if not legacy:
        train_window(model, optimizer, state, window, config.training)
        for a, b in zip(m.parameters(), model.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        for key in s.fast_weights:
            torch.testing.assert_close(s.fast_weights[key], state.fast_weights[key], atol=0, rtol=0)
