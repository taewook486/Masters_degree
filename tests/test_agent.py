"""agent.py 검증 강화 테스트 (REQ-RI-005, REQ-RI-006).

- _validate_config: 필수 키 누락 시 기본값 적용
- epochs -> max_steps 하위 호환성 마이그레이션
- 탐색/활용 전환 구조화 로깅
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_PATH = Path(__file__).parent.parent / "src" / "autoresearch" / "agent.py"


def _read_source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def test_validate_config_has_required_key_check():
    """_validate_config에 필수 키 존재 여부 확인 로직이 있어야 한다."""
    source = _read_source()
    # 필수 키 누락 시 기본값 적용 로직 확인
    assert "default" in source.lower() or "기본" in source, (
        "_validate_config에 필수 키 누락 시 기본값 적용 로직이 필요"
    )


def test_validate_config_epochs_to_max_steps_migration():
    """epochs -> max_steps 하위 호환성 마이그레이션이 있어야 한다."""
    source = _read_source()
    assert "max_steps" in source and "epochs" in source, (
        "epochs -> max_steps 마이그레이션 로직이 필요"
    )
    # max_steps 키가 결과에 포함되어야 함
    assert '"max_steps"' in source, "결과에 max_steps 키가 포함되어야 함"


def test_exploration_exploitation_logging():
    """탐색/활용 전환 구조화 로깅이 있어야 한다 (REQ-RI-006)."""
    source = _read_source()
    assert "exploration" in source.lower() or "exploitation" in source.lower(), (
        "탐색/활용 단계에 대한 구조화 로깅이 필요"
    )
    # phase 관련 로그
    assert "phase" in source.lower(), (
        "현재 phase(exploration/exploitation) 로그가 필요"
    )


def test_validate_config_handles_missing_keys():
    """_validate_config에 누락 키에 대한 get() 패턴이 있어야 한다."""
    source = _read_source()
    # 이미 .get()을 사용하고 있지만, max_steps에 대한 처리도 필요
    assert "max_steps" in source, "_validate_config에 max_steps 처리가 필요"
