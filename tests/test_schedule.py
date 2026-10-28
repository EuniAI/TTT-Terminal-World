import pytest
from terminal_world.config import TrainConfig
from terminal_world.schedule import make_schedule, outer_learning_rate


def config(**kwargs):
    return TrainConfig(inner_lr=0.1, outer_lr=4e-4, unroll_events=4,
                       prefix_batch_size=1, outer_lr_schedule="warmup_cosine", **kwargs)


def test_schedule_endpoints_and_monotonicity():
    cfg = config()
    plan = make_schedule(cfg, 400, 1)
    rates = [outer_learning_rate(cfg, i, plan) for i in range(110)]
    assert plan == {"total_windows": 100, "warmup_windows": 10}
    assert rates[0] == 0
    assert rates[5] == pytest.approx(2e-4)
    assert rates[10] == pytest.approx(4e-4)
    assert rates[99:] == [1e-5] * 11
    assert all(a <= b for a, b in zip(rates[:10], rates[1:11]))
    assert all(a >= b for a, b in zip(rates[10:], rates[11:]))


def test_epoch_tail_windows_and_short_runs():
    cfg = config()
    assert make_schedule(cfg, 5, 2)["total_windows"] == 4
    plan = make_schedule(cfg, 9, 1)
    assert [outer_learning_rate(cfg, i, plan) for i in range(3)] == [0, 4e-4, 1e-5]
    with pytest.raises(ValueError, match="at least 3"):
        make_schedule(cfg, 4, 1)
    cfg.outer_warmup_ratio = 0
    assert outer_learning_rate(cfg, 0, make_schedule(cfg, 12, 1)) == 4e-4


@pytest.mark.parametrize("field,value", [
    ("outer_lr_schedule", "bad"), ("outer_warmup_ratio", -0.1),
    ("outer_warmup_ratio", 1), ("outer_warmup_ratio", float("nan")),
    ("outer_end_lr", -1), ("outer_end_lr", float("inf")), ("outer_end_lr", 1),
])
def test_invalid_schedule_config(field, value):
    cfg = config()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_missing_resume_plan_and_legacy_defaults():
    with pytest.raises(ValueError, match="saved outer_lr_plan"):
        outer_learning_rate(config(), 5, None)
    cfg = TrainConfig(inner_lr=0.1, outer_lr=1e-6, unroll_events=4, prefix_batch_size=1)
    cfg.validate()
    assert make_schedule(cfg, 1, 1) is None
    assert outer_learning_rate(cfg, 100, None) == 1e-6
