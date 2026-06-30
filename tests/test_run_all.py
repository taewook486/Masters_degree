"""run_all.py 테스트 (REQ-RI-002, REQ-RI-007).

_load_existing_result, _aggregate_seed_results, _save_intermediate,
generate_summary_csv 함수를 직접 호출하여 테스트한다.

run_all.py는 torch, pandas, src.baseline을 import하므로
exec() 기반 모듈 로딩을 사용한다.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock

import pytest

# --- 모듈 로드 ---
_MODULE = None
_IMPORT_ERROR = None


def _load_module():
    """run_all 모듈을 의존성 mock과 함께 로드한다."""
    saved = {}

    # torch mock
    mock_torch = MagicMock()
    mock_torch.cuda.OutOfMemoryError = RuntimeError
    mock_torch.cuda.is_available.return_value = False
    saved["torch"] = sys.modules.get("torch")
    sys.modules["torch"] = mock_torch

    # transformers mock
    saved["transformers"] = sys.modules.get("transformers")
    sys.modules["transformers"] = MagicMock()

    # pandas mock
    saved["pandas"] = sys.modules.get("pandas")
    mock_pd = MagicMock()
    mock_pd.DataFrame = MagicMock
    sys.modules["pandas"] = mock_pd

    # src.baseline 패키지 mock
    saved["src.baseline"] = sys.modules.get("src.baseline")
    baseline_pkg = types.ModuleType("src.baseline")
    baseline_pkg.__path__ = [
        os.path.join(os.path.dirname(__file__), "..", "src", "baseline")
    ]
    baseline_pkg.__package__ = "src.baseline"
    sys.modules["src.baseline"] = baseline_pkg

    saved["src.baseline.model_loader"] = sys.modules.get("src.baseline.model_loader")
    sys.modules["src.baseline.model_loader"] = MagicMock()

    saved["src.baseline.evaluate_zero_shot"] = sys.modules.get("src.baseline.evaluate_zero_shot")
    sys.modules["src.baseline.evaluate_zero_shot"] = MagicMock()

    # 깨진 모듈 제거
    sys.modules.pop("src.baseline.run_all", None)

    # 새 모듈 객체 생성 후 exec
    mod = types.ModuleType("src.baseline.run_all")
    mod.__file__ = os.path.join(
        os.path.dirname(__file__), "..", "src", "baseline", "run_all.py"
    )
    mod.__package__ = "src.baseline"
    sys.modules["src.baseline.run_all"] = mod

    src_path = os.path.normpath(mod.__file__)
    with open(src_path, encoding="utf-8") as f:
        source = f.read()
    code = compile(source, src_path, "exec")
    exec(code, mod.__dict__)  # noqa: S102

    # 복원
    for key, val in saved.items():
        if val is not None:
            sys.modules[key] = val
        else:
            sys.modules.pop(key, None)

    return mod


try:
    _MODULE = _load_module()
except Exception as e:
    _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    _MODULE is None,
    reason=f"run_all import 실패: {_IMPORT_ERROR}",
)


# --- _load_existing_result 테스트 ---


def test_load_existing_result_returns_none_for_missing_file():
    """결과 파일이 없으면 None을 반환한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _MODULE._load_existing_result(tmpdir, "model_x", "pathvqa", 42)
        assert result is None


def test_load_existing_result_returns_none_for_cpu_result():
    """peak_vram_mb가 0이면 CPU 실행으로 판단하고 None을 반환한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "metadata": {"num_samples": 100},
            "summary": {"peak_vram_mb": 0, "overall_accuracy": 0.8},
        }
        fpath = os.path.join(tmpdir, "model_x_pathvqa_seed42.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)
        result = _MODULE._load_existing_result(tmpdir, "model_x", "pathvqa", 42)
        assert result is None


def test_load_existing_result_returns_summary_for_valid():
    """유효한 GPU 결과가 있으면 summary를 반환한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "metadata": {"num_samples": 100},
            "summary": {"peak_vram_mb": 512, "overall_accuracy": 0.8},
        }
        fpath = os.path.join(tmpdir, "model_x_pathvqa_seed42.json")
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f)
        result = _MODULE._load_existing_result(tmpdir, "model_x", "pathvqa", 42)
        assert result is not None
        assert result["overall_accuracy"] == 0.8


# --- _aggregate_seed_results 테스트 ---


def test_aggregate_seed_results_basic():
    """기본 집계 결과가 올바른 구조를 가진다."""
    results = [
        {"closed_accuracy": 0.8, "open_accuracy": 0.6, "overall_accuracy": 0.7,
         "avg_time_ms": 100, "peak_vram_mb": 512},
        {"closed_accuracy": 0.9, "open_accuracy": 0.7, "overall_accuracy": 0.8,
         "avg_time_ms": 110, "peak_vram_mb": 600},
    ]
    agg = _MODULE._aggregate_seed_results("test_model", "pathvqa", results)
    assert agg["model_name"] == "test_model"
    assert agg["dataset"] == "pathvqa"
    assert agg["num_seeds"] == 2
    assert "closed_acc_mean" in agg
    assert "closed_acc_std" in agg
    assert "overall_acc_mean" in agg
    assert agg["peak_vram_mb"] == 600  # max


def test_aggregate_includes_bertscore():
    """BERTScore 통계가 집계 결과에 포함된다 (REQ-RI-002)."""
    results = [
        {"closed_accuracy": 0.8, "open_accuracy": 0.6, "overall_accuracy": 0.7,
         "open_bertscore_f1": 0.85, "open_bertscore_accuracy": 0.9,
         "avg_time_ms": 100, "peak_vram_mb": 512},
    ]
    agg = _MODULE._aggregate_seed_results("m", "d", results)
    assert "open_bertscore_f1_mean" in agg
    assert "open_bertscore_accuracy_mean" in agg
    assert agg["open_bertscore_f1_mean"] == 0.85


# --- _save_intermediate 테스트 ---


def test_save_intermediate_creates_json():
    """중간 결과 JSON 파일이 생성된다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results = [{"model_name": "test", "dataset": "pathvqa", "num_seeds": 1}]
        _MODULE._save_intermediate(results, tmpdir)

        recovery_file = os.path.join(tmpdir, "phase1_intermediate.json")
        assert os.path.exists(recovery_file)

        with open(recovery_file, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1


# --- generate_summary_csv 테스트 ---


def test_generate_summary_csv():
    """summary CSV 파일이 생성된다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # mock DataFrame with to_csv that writes a real file
        mock_df = MagicMock()
        def _to_csv(path, **kwargs):
            from pathlib import Path
            Path(path).write_text("model_name,dataset\nm1,pathvqa\n", encoding="utf-8")
        mock_df.to_csv = _to_csv

        csv_path = _MODULE.generate_summary_csv(mock_df, tmpdir)
        assert csv_path.exists()
        assert csv_path.name == "phase1_summary.csv"


# --- run_all_conditions 오류 경로 테스트 (lines 74-79) ---


def test_run_all_conditions_no_configs_raises():
    """설정 파일이 없는 디렉토리에서 FileNotFoundError를 발생시킨다 (line 78)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="No YAML configs"):
            _MODULE.run_all_conditions(
                config_dir=tmpdir,
                output_dir=tmpdir,
                seeds=[42],
            )


def test_run_all_conditions_disabled_config_skipped():
    """enabled: false인 config는 건너뛰고 정상 종료한다 (lines 96-98)."""
    import yaml
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "model_a.yaml")
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump({"enabled": False, "model_name": "test"}, f)

        # enabled=False이므로 load_model이 호출되지 않고 정상 종료
        result = _MODULE.run_all_conditions(
            config_dir=tmpdir,
            output_dir=tmpdir,
            seeds=[42],
        )
        # pandas가 mock이므로 DataFrame(mock).called 확인 대신 예외 없음 확인
        assert result is not None
