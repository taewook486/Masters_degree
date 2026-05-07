"""HPO 루프 체크포인트 저장/복원 테스트 (REQ-RI-008).

루프 상태(현재 trial 인덱스, 전략명, repeat_id)를 저장/복원한다.
"""

from __future__ import annotations

import json
import os
import tempfile


def test_save_checkpoint_creates_file():
    """save_checkpoint는 체크포인트 파일을 생성해야 한다."""
    from src.utils.checkpoint import save_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        state = {"completed_trials": 5, "strategy": "autoresearch", "repeat_id": 0}
        save_checkpoint(tmpdir, state)
        files = os.listdir(tmpdir)
        assert any("checkpoint" in f for f in files)


def test_load_checkpoint_returns_saved_state():
    """load_checkpoint는 저장된 상태를 정확히 복원해야 한다."""
    from src.utils.checkpoint import load_checkpoint, save_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        state = {"completed_trials": 10, "strategy": "optuna", "repeat_id": 2}
        save_checkpoint(tmpdir, state)
        loaded = load_checkpoint(tmpdir)
        assert loaded is not None
        assert loaded["completed_trials"] == 10
        assert loaded["strategy"] == "optuna"
        assert loaded["repeat_id"] == 2


def test_load_checkpoint_returns_none_when_no_checkpoint():
    """체크포인트 파일이 없으면 None을 반환해야 한다."""
    from src.utils.checkpoint import load_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        loaded = load_checkpoint(tmpdir)
        assert loaded is None


def test_save_checkpoint_overwrites_previous():
    """새 체크포인트가 이전 것을 덮어써야 한다."""
    from src.utils.checkpoint import load_checkpoint, save_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(tmpdir, {"completed_trials": 5})
        save_checkpoint(tmpdir, {"completed_trials": 15})
        loaded = load_checkpoint(tmpdir)
        assert loaded["completed_trials"] == 15


def test_save_checkpoint_creates_dir_if_needed():
    """체크포인트 디렉토리가 없으면 자동 생성해야 한다."""
    from src.utils.checkpoint import save_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        nested = os.path.join(tmpdir, "sub", "ckpt")
        save_checkpoint(nested, {"completed_trials": 1})
        assert os.path.isdir(nested)
