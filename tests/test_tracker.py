"""tracker.py JSON 내보내기 및 phase/temperature 필드 테스트 (REQ-RI-006, REQ-RI-009).

- JSON 내보내기 메서드 추가
- TrialResult에 phase/temperature 메타데이터 필드 추가
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

_SOURCE_PATH = Path(__file__).parent.parent / "src" / "autoresearch" / "tracker.py"


def _read_source() -> str:
    return _SOURCE_PATH.read_text(encoding="utf-8")


def test_tracker_has_export_json_method():
    """ExperimentTracker에 JSON 내보내기 메서드가 있어야 한다."""
    source = _read_source()
    assert "export_json" in source or "to_json" in source, (
        "ExperimentTracker에 JSON 내보내기 메서드가 필요"
    )


def test_trial_result_has_phase_field():
    """TrialResult에 phase 필드가 있어야 한다 (REQ-RI-006)."""
    source = _read_source()
    assert "phase" in source, "TrialResult에 phase 필드가 필요"


def test_trial_result_has_temperature_field():
    """TrialResult에 temperature 필드가 있어야 한다 (REQ-RI-006)."""
    source = _read_source()
    assert "temperature" in source, "TrialResult에 temperature 필드가 필요"


def test_export_json_produces_metadata_summary_structure():
    """JSON 내보내기가 metadata/summary 구조를 따라야 한다 (REQ-RI-009)."""
    source = _read_source()
    assert "metadata" in source and "summary" in source, (
        "JSON 내보내기가 metadata/summary 구조를 따라야 함"
    )
