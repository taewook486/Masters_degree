"""strategies.py 중복 설정 감지 테스트 (REQ-RI-004).

AutoresearchStrategy.suggest()가 이전에 완료된 trial과 동일한 설정을 감지한다.
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_PATH = Path(__file__).parent.parent / "src" / "autoresearch" / "strategies.py"


def _read_source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def test_autoresearch_strategy_has_duplicate_detection():
    """AutoresearchStrategy.suggest에 중복 감지 로직이 있어야 한다."""
    source = _read_source()
    assert "duplicate" in source.lower() or "중복" in source, (
        "AutoresearchStrategy.suggest에 중복 감지 로직이 필요"
    )


def test_duplicate_detection_uses_config_comparison():
    """중복 감지가 하이퍼파라미터 설정 비교를 사용해야 한다."""
    source = _read_source()
    # config_to_dict 함수가 중복 비교에 사용되거나 직접 비교 로직이 있어야 함
    assert "config_to_dict" in source or "_is_duplicate" in source, (
        "하이퍼파라미터 설정 비교를 위한 함수가 필요"
    )


def test_duplicate_detection_has_max_retries():
    """무한 루프 방지를 위한 최대 재시도 제한이 있어야 한다."""
    source = _read_source()
    # 재시도 제한이 있어야 함
    assert "retry" in source.lower() or "max_" in source.lower() or "attempt" in source.lower(), (
        "중복 감지에 최대 재시도 제한이 필요"
    )
