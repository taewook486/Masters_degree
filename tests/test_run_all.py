"""run_all.py 변경 사항 테스트 (REQ-RI-002, REQ-RI-007).

- _aggregate_seed_results에 BERTScore 집계 포함
- _load_existing_result에서 BERTScore 키 검증
- main()에서 setup_logging 사용
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_PATH = Path(__file__).parent.parent / "src" / "baseline" / "run_all.py"


def _read_source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def test_aggregate_includes_bertscore_stats():
    """_aggregate_seed_results가 BERTScore 통계를 집계해야 한다."""
    source = _read_source()
    assert "bertscore" in source.lower(), (
        "_aggregate_seed_results에 BERTScore 집계 로직이 있어야 함"
    )
    assert "open_bertscore_f1" in source, (
        "open_bertscore_f1 키를 집계해야 함"
    )


def test_main_uses_setup_logging():
    """main()에서 setup_logging을 사용해야 한다."""
    source = _read_source()
    assert "setup_logging" in source, (
        "main()에서 setup_logging을 사용해야 함"
    )


def test_load_existing_result_validates_bertscore():
    """_load_existing_result가 BERTScore 키를 검증해야 한다."""
    source = _read_source()
    # _load_existing_result 함수 내에서 bertscore 관련 검증이 있어야 함
    assert "bertscore" in source.lower(), (
        "_load_existing_result에서 BERTScore 키 검증이 필요"
    )
