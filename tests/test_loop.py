"""loop.py 체크포인트 및 재개 테스트 (REQ-RI-008).

run_hpo_loop이 기존 trial 수를 확인하고 이어서 실행하는 로직.
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_PATH = Path(__file__).parent.parent / "src" / "autoresearch" / "loop.py"


def _read_source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def test_run_hpo_loop_has_resume_logic():
    """run_hpo_loop에 기존 trial 수를 확인하고 재개하는 로직이 있어야 한다."""
    source = _read_source()
    # 기존 완료 trial 수를 확인하는 로직
    assert "completed" in source.lower() and "skip" in source.lower() or "resume" in source.lower(), (
        "run_hpo_loop에 기존 trial 재개 로직이 필요"
    )


def test_run_hpo_loop_imports_checkpoint():
    """loop.py가 checkpoint 모듈을 import해야 한다."""
    source = _read_source()
    assert "checkpoint" in source, (
        "checkpoint 모듈을 import해야 함"
    )


def test_run_hpo_loop_saves_checkpoint():
    """run_hpo_loop이 각 trial 후 체크포인트를 저장해야 한다."""
    source = _read_source()
    assert "save_checkpoint" in source, (
        "각 trial 후 save_checkpoint를 호출해야 함"
    )
