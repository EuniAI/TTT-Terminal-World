import os
import pytest
import torch
from terminal_world.checkpoint import retained_atomic_save


def test_retains_three_distinct_windows_and_latest_alias(tmp_path):
    latest = tmp_path / 'checkpoint.pt'
    history = tmp_path / 'checkpoint.checkpoints'
    for window in (0, 50, 100, 150):
        retained_atomic_save({'windows_completed': window}, latest)
    assert sorted(p.name for p in history.iterdir()) == [
        f'window-{w:012d}.pt' for w in (50, 100, 150)]
    assert os.path.samefile(latest, history / 'window-000000000150.pt')
    assert torch.load(latest, weights_only=False)['windows_completed'] == 150
    retained_atomic_save({'windows_completed': 150, 'updated': True}, latest)
    assert len(list(history.iterdir())) == 3
    assert torch.load(latest, weights_only=False)['updated']
    # An explicit rollback still publishes the selected window and preserves three versions.
    retained_atomic_save({'windows_completed': 25}, latest)
    assert torch.load(latest, weights_only=False)['windows_completed'] == 25
    assert len(list(history.iterdir())) == 3


def test_failed_write_keeps_latest_and_history(tmp_path, monkeypatch):
    latest = tmp_path / 'checkpoint.pt'
    for window in (0, 50, 100):
        retained_atomic_save({'windows_completed': window}, latest)
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='disk full'):
        retained_atomic_save({'windows_completed': 150}, latest)
    assert torch.load(latest, weights_only=False)['windows_completed'] == 100
    assert len(list((tmp_path / 'checkpoint.checkpoints').iterdir())) == 3
