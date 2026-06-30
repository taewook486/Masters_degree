"""tracker.py 테스트 (REQ-RI-006, REQ-RI-009).

ExperimentTracker와 TrialResult의 CRUD, export_json, summary_text 테스트.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.autoresearch.tracker import ExperimentTracker, TrialResult, TSV_COLUMNS


def _make_trial(trial_id: int = 0, strategy: str = "random", repeat_id: int = 0,
                val_accuracy: float = 0.5, status: str = "completed", **kwargs) -> TrialResult:
    """테스트용 TrialResult 생성."""
    return TrialResult(
        trial_id=trial_id, strategy=strategy, repeat_id=repeat_id,
        val_accuracy=val_accuracy, status=status, **kwargs,
    )


def test_trial_result_defaults():
    """TrialResult 기본값이 올바르게 설정된다."""
    t = TrialResult(trial_id=0, strategy="manual", repeat_id=0)
    assert t.lora_rank == 16
    assert t.lora_alpha == 32
    assert t.status == "pending"
    assert t.phase == ""
    assert t.temperature == 0.0


def test_tsv_columns_includes_phase_and_temperature():
    """TSV_COLUMNS에 phase와 temperature가 포함된다 (REQ-RI-006)."""
    assert "phase" in TSV_COLUMNS
    assert "temperature" in TSV_COLUMNS


def test_tracker_creates_header():
    """ExperimentTracker 생성 시 TSV 헤더가 작성된다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results.tsv"
        ExperimentTracker(path)
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            header = f.readline().strip()
        assert "trial_id" in header
        assert "strategy" in header


def test_tracker_append_and_load():
    """append 후 load_all로 정확히 로드된다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results.tsv"
        tracker = ExperimentTracker(path)
        trial = _make_trial(trial_id=0, val_accuracy=0.85, train_loss=0.3)
        tracker.append(trial)

        loaded = tracker.load_all()
        assert len(loaded) == 1
        assert loaded[0].trial_id == 0
        assert loaded[0].val_accuracy == 0.85
        assert loaded[0].train_loss == 0.3


def test_tracker_load_by_strategy():
    """load_by_strategy가 전략별로 필터링한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "results.tsv")
        tracker.append(_make_trial(trial_id=0, strategy="random", repeat_id=0))
        tracker.append(_make_trial(trial_id=1, strategy="optuna", repeat_id=0))
        tracker.append(_make_trial(trial_id=2, strategy="random", repeat_id=1))

        random_all = tracker.load_by_strategy("random")
        assert len(random_all) == 2

        random_r0 = tracker.load_by_strategy("random", repeat_id=0)
        assert len(random_r0) == 1
        assert random_r0[0].trial_id == 0


def test_tracker_next_trial_id():
    """next_trial_id가 올바른 다음 ID를 반환한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "results.tsv")
        assert tracker.next_trial_id() == 0

        tracker.append(_make_trial(trial_id=5))
        assert tracker.next_trial_id() == 6


def test_tracker_best_trial():
    """best_trial이 가장 높은 val_accuracy를 반환한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "results.tsv")
        tracker.append(_make_trial(trial_id=0, strategy="random", val_accuracy=0.3))
        tracker.append(_make_trial(trial_id=1, strategy="random", val_accuracy=0.9))
        tracker.append(_make_trial(trial_id=2, strategy="random", val_accuracy=0.6))

        best = tracker.best_trial("random")
        assert best is not None
        assert best.trial_id == 1
        assert best.val_accuracy == 0.9


def test_tracker_best_trial_none_for_failed_only():
    """완료된 trial이 없으면 None을 반환한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "results.tsv")
        tracker.append(_make_trial(trial_id=0, strategy="random", status="failed"))
        assert tracker.best_trial("random") is None


def test_tracker_export_json(tmp_path):
    """export_json이 metadata/summary 구조의 JSON을 생성한다 (REQ-RI-009)."""
    tracker = ExperimentTracker(tmp_path / "results.tsv")
    tracker.append(_make_trial(trial_id=0, strategy="random", val_accuracy=0.7))
    tracker.append(_make_trial(trial_id=1, strategy="random", val_accuracy=0.9))
    tracker.append(_make_trial(trial_id=2, strategy="optuna", val_accuracy=0.8))

    out = tracker.export_json(tmp_path / "export.json", strategy="random")
    assert out.exists()

    with open(out, encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "summary" in data
    assert "trials" in data
    assert data["metadata"]["strategy"] == "random"
    assert data["metadata"]["completed_trials"] == 2
    assert data["summary"]["best_val_accuracy"] == 0.9
    assert len(data["trials"]) == 2


def test_tracker_export_json_all_strategies(tmp_path):
    """strategy=None이면 전체 trial을 내보낸다."""
    tracker = ExperimentTracker(tmp_path / "results.tsv")
    tracker.append(_make_trial(trial_id=0, strategy="random"))
    tracker.append(_make_trial(trial_id=1, strategy="optuna"))

    out = tracker.export_json(tmp_path / "all.json")
    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["strategy"] == "all"
    assert len(data["trials"]) == 2


def test_tracker_summary_text():
    """summary_text가 읽기 쉬운 텍스트를 반환한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "results.tsv")
        tracker.append(_make_trial(trial_id=0, strategy="random", repeat_id=0, val_accuracy=0.85))

        text = tracker.summary_text("random", 0)
        assert "Strategy: random" in text
        assert "0.85" in text


def test_tracker_summary_text_no_completed():
    """완료된 trial이 없으면 안내 메시지를 반환한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "results.tsv")
        text = tracker.summary_text("random", 0)
        assert "No completed trials" in text


def test_tracker_load_nonexistent_file():
    """존재하지 않는 파일에서 load_all은 빈 리스트를 반환한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results.tsv"
        tracker = ExperimentTracker(path)
        path.unlink()
        assert tracker.load_all() == []
