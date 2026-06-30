"""catastrophic_forgetting.py 테스트 (REQ-RI-009).

measure_catastrophic_forgetting, run_cf_measurement 테스트.
evaluate_on_vqav2는 모델 의존이므로 mock 처리.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock, patch

import pytest

# --- 모듈 로드 ---
_MODULE = None
_IMPORT_ERROR = None


def _load_module():
    """catastrophic_forgetting 모듈을 의존성 mock과 함께 로드한다."""
    saved = {}

    # src.data.general_vqa mock
    if "src.data" not in sys.modules or sys.modules.get("src.data") is None:
        data_pkg = types.ModuleType("src.data")
        data_pkg.__path__ = [
            os.path.join(os.path.dirname(__file__), "..", "src", "data")
        ]
        sys.modules["src.data"] = data_pkg
    if "src.data.general_vqa" not in sys.modules or sys.modules.get("src.data.general_vqa") is None:
        sys.modules["src.data.general_vqa"] = MagicMock()

    # src.baseline.model_loader mock (generate_answer)
    saved["transformers"] = sys.modules.get("transformers")
    sys.modules["transformers"] = MagicMock()

    saved["src.baseline"] = sys.modules.get("src.baseline")
    baseline_pkg = types.ModuleType("src.baseline")
    baseline_pkg.__path__ = [
        os.path.join(os.path.dirname(__file__), "..", "src", "baseline")
    ]
    baseline_pkg.__package__ = "src.baseline"
    sys.modules["src.baseline"] = baseline_pkg

    saved["src.baseline.model_loader"] = sys.modules.get("src.baseline.model_loader")
    sys.modules["src.baseline.model_loader"] = MagicMock()

    # 깨진 모듈 제거
    sys.modules.pop("src.evaluate.catastrophic_forgetting", None)

    mod = types.ModuleType("src.evaluate.catastrophic_forgetting")
    mod.__file__ = os.path.join(
        os.path.dirname(__file__), "..", "src", "evaluate", "catastrophic_forgetting.py"
    )
    mod.__package__ = "src.evaluate"
    sys.modules["src.evaluate.catastrophic_forgetting"] = mod

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
    reason=f"catastrophic_forgetting import 실패: {_IMPORT_ERROR}",
)


# --- measure_catastrophic_forgetting 테스트 ---


def test_measure_cf_computes_degradation():
    """base와 finetuned 결과로 degradation을 올바르게 계산한다."""
    base = {"closed_accuracy": 0.8, "open_accuracy": 0.6, "overall_accuracy": 0.7}
    ft = {"closed_accuracy": 0.7, "open_accuracy": 0.5, "overall_accuracy": 0.6}

    metrics = _MODULE.measure_catastrophic_forgetting(base, ft)

    assert "base_overall_accuracy" in metrics
    assert "finetuned_overall_accuracy" in metrics
    assert "degradation_overall_accuracy_pct" in metrics
    # (0.7 - 0.6) / 0.7 * 100 = 14.29%
    assert abs(metrics["degradation_overall_accuracy_pct"] - 14.29) < 0.1


def test_measure_cf_zero_base_no_division_error():
    """base가 0이면 degradation도 0이다."""
    base = {"closed_accuracy": 0.0, "open_accuracy": 0.0, "overall_accuracy": 0.0}
    ft = {"closed_accuracy": 0.5, "open_accuracy": 0.3, "overall_accuracy": 0.4}

    metrics = _MODULE.measure_catastrophic_forgetting(base, ft)
    assert metrics["degradation_overall_accuracy_pct"] == 0.0


def test_measure_cf_no_degradation():
    """finetuned가 더 좋으면 음수 degradation이 나온다."""
    base = {"closed_accuracy": 0.5, "open_accuracy": 0.4, "overall_accuracy": 0.45}
    ft = {"closed_accuracy": 0.6, "open_accuracy": 0.5, "overall_accuracy": 0.55}

    metrics = _MODULE.measure_catastrophic_forgetting(base, ft)
    assert metrics["degradation_overall_accuracy_pct"] < 0


# --- run_cf_measurement 테스트 ---


@patch.object(_MODULE, "evaluate_on_vqav2")
def test_run_cf_measurement_returns_unified_schema(mock_eval):
    """run_cf_measurement가 metadata/summary 구조를 반환한다 (REQ-RI-009)."""
    mock_eval.return_value = {
        "closed_accuracy": 0.7, "open_accuracy": 0.5, "overall_accuracy": 0.6,
        "closed_count": 50, "open_count": 50, "total_count": 100, "eval_time_sec": 30.0,
    }
    base_result = {
        "closed_accuracy": 0.8, "open_accuracy": 0.6, "overall_accuracy": 0.7,
        "closed_count": 50, "open_count": 50, "total_count": 100, "eval_time_sec": 25.0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _MODULE.run_cf_measurement(
            model=MagicMock(), processor=MagicMock(), config=MagicMock(),
            base_vqav2_result=base_result,
            output_dir=tmpdir, model_name="test_model",
            dataset_name="pathvqa", seed=42,
        )

        # metadata/summary 구조 확인
        assert "metadata" in result
        assert "summary" in result
        assert result["metadata"]["model_name"] == "test_model"
        assert result["metadata"]["measurement_type"] == "catastrophic_forgetting"
        assert "degradation_overall_accuracy_pct" in result["summary"]

        # JSON 파일 생성 확인
        cf_file = os.path.join(tmpdir, "cf_result.json")
        assert os.path.exists(cf_file)

        with open(cf_file, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["metadata"]["model_name"] == "test_model"


@patch.object(_MODULE, "evaluate_on_vqav2")
def test_run_cf_measurement_without_base_result(mock_eval):
    """base_vqav2_result이 None이면 finetuned 결과만 반환한다."""
    mock_eval.return_value = {
        "closed_accuracy": 0.7, "open_accuracy": 0.5, "overall_accuracy": 0.6,
        "closed_count": 50, "open_count": 50, "total_count": 100, "eval_time_sec": 30.0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _MODULE.run_cf_measurement(
            model=MagicMock(), processor=MagicMock(), config=MagicMock(),
            base_vqav2_result=None,
            output_dir=tmpdir, model_name="test_model",
            dataset_name="pathvqa", seed=42,
        )

    assert "finetuned_vqav2" in result
    assert "metadata" not in result  # base가 없으면 full 구조가 아님


# --- evaluate_on_vqav2 테스트 (lines 51-103, REQ-RI-009) ---


def test_evaluate_on_vqav2_mixed_question_types():
    """closed/open 샘플이 섞인 경우 올바른 통계를 반환한다 (lines 51-111)."""
    # 가짜 샘플 생성
    closed_sample = MagicMock()
    closed_sample.question = "Is this abnormal?"
    closed_sample.answer = "yes"
    closed_sample.question_type = "closed"

    open_sample = MagicMock()
    open_sample.question = "What does this show?"
    open_sample.answer = "cancer"
    open_sample.question_type = "open"

    # model_loader mock: generate_answer가 올바른 답 반환
    mock_model_loader = MagicMock()
    mock_model_loader.generate_answer.return_value = "yes"

    with patch.object(_MODULE, "load_vqav2_subset", return_value=[closed_sample, open_sample]):
        with patch.dict(sys.modules, {"src.baseline.model_loader": mock_model_loader}):
            result = _MODULE.evaluate_on_vqav2(
                model=MagicMock(),
                processor=MagicMock(),
                config=MagicMock(),
                data_dir="data",
                max_samples=2,
            )

    assert result["total_count"] == 2
    assert result["closed_count"] == 1
    assert result["open_count"] == 1
    assert "overall_accuracy" in result
    assert "eval_time_sec" in result


def test_evaluate_on_vqav2_inference_exception_handled():
    """generate_answer 예외 시 빈 문자열로 처리한다 (lines 68-78)."""
    sample = MagicMock()
    sample.question = "Is this abnormal?"
    sample.answer = "yes"
    sample.question_type = "closed"

    mock_model_loader = MagicMock()
    mock_model_loader.generate_answer.side_effect = RuntimeError("inference error")

    with patch.object(_MODULE, "load_vqav2_subset", return_value=[sample]):
        with patch.dict(sys.modules, {"src.baseline.model_loader": mock_model_loader}):
            result = _MODULE.evaluate_on_vqav2(
                model=MagicMock(),
                processor=MagicMock(),
                config=MagicMock(),
            )

    # 예외가 발생해도 total_count는 1이어야 함
    assert result["total_count"] == 1


def test_evaluate_on_vqav2_empty_samples():
    """샘플이 없으면 전체 정확도 0.0을 반환한다 (line 101)."""
    mock_model_loader = MagicMock()

    with patch.object(_MODULE, "load_vqav2_subset", return_value=[]):
        with patch.dict(sys.modules, {"src.baseline.model_loader": mock_model_loader}):
            result = _MODULE.evaluate_on_vqav2(
                model=MagicMock(),
                processor=MagicMock(),
                config=MagicMock(),
            )

    assert result["total_count"] == 0
    assert result["overall_accuracy"] == 0.0
