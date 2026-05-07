"""evaluate_zero_shot.py 변경 사항 테스트 (REQ-RI-001, REQ-RI-003, REQ-RI-007).

ML 라이브러리 의존성 없이 소스 코드 분석으로 검증한다.
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_PATH = Path(__file__).parent.parent / "src" / "baseline" / "evaluate_zero_shot.py"


def _read_source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def test_evaluate_with_loaded_model_calls_bertscore():
    """evaluate_with_loaded_model이 compute_bertscore=True로 호출해야 한다."""
    source = _read_source()
    assert "compute_bertscore=True" in source, (
        "compute_overall_accuracy 호출에 compute_bertscore=True를 명시해야 함"
    )


def test_result_metadata_has_environment_field():
    """결과 딕셔너리의 metadata에 environment 필드가 포함되어야 한다."""
    source = _read_source()
    assert "environment" in source, (
        "결과 JSON의 metadata에 environment 필드가 포함되어야 함"
    )


def test_imports_environment_module():
    """evaluate_zero_shot이 get_environment_info를 import해야 한다."""
    source = _read_source()
    assert "get_environment_info" in source, (
        "get_environment_info를 import해야 함"
    )


def test_main_uses_setup_logging():
    """main()에서 setup_logging을 사용해야 한다."""
    source = _read_source()
    assert "setup_logging" in source, (
        "main()에서 setup_logging을 사용해야 함"
    )
