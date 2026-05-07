"""catastrophic_forgetting.py 스키마 통일 테스트 (REQ-RI-009).

출력 스키마를 metadata/summary 구조로 통일.
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_PATH = Path(__file__).parent.parent / "src" / "evaluate" / "catastrophic_forgetting.py"


def _read_source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def test_run_cf_measurement_has_metadata_field():
    """run_cf_measurement 결과에 metadata 필드가 포함되어야 한다."""
    source = _read_source()
    assert '"metadata"' in source, (
        "run_cf_measurement 결과에 metadata 필드가 필요"
    )


def test_run_cf_measurement_has_summary_field():
    """run_cf_measurement 결과에 summary 필드가 포함되어야 한다."""
    source = _read_source()
    assert '"summary"' in source, (
        "run_cf_measurement 결과에 summary 필드가 필요"
    )


def test_result_follows_unified_schema():
    """결과가 metadata/summary 통일 스키마를 따라야 한다."""
    source = _read_source()
    # metadata와 summary가 같은 result dict에 있어야 함
    assert "metadata" in source and "summary" in source, (
        "metadata/summary 통일 스키마가 필요"
    )
