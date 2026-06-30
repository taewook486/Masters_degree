"""evaluate_zero_shot.py 실제 함수 테스트 (REQ-RI-001, REQ-RI-003, REQ-RI-007).

transformers 미설치 및 torch reimport 버그, pytest-cov 환경에서도
테스트 가능하도록 exec()으로 모듈을 로드한다.
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
    """evaluate_zero_shot 모듈을 의존성 mock과 함께 로드한다.

    pytest-cov가 coverage tracer를 설치하면 importlib.util.exec_module이
    'cannot load module more than once per process'를 발생시키므로,
    exec()으로 소스 코드를 직접 실행하여 모듈 네임스페이스를 구성한다.
    """
    # 1. torch reimport 버그 방지: mock으로 교체
    mock_torch = MagicMock()
    mock_torch.cuda.OutOfMemoryError = RuntimeError
    saved = {}
    saved["torch"] = sys.modules.get("torch")
    sys.modules["torch"] = mock_torch

    # 2. transformers mock
    saved["transformers"] = sys.modules.get("transformers")
    sys.modules["transformers"] = MagicMock()

    # 3. src.baseline 패키지를 빈 모듈로 대체
    saved["src.baseline"] = sys.modules.get("src.baseline")
    baseline_pkg = types.ModuleType("src.baseline")
    baseline_pkg.__path__ = [
        os.path.join(os.path.dirname(__file__), "..", "src", "baseline")
    ]
    baseline_pkg.__package__ = "src.baseline"
    sys.modules["src.baseline"] = baseline_pkg

    # 4. model_loader mock
    saved["src.baseline.model_loader"] = sys.modules.get("src.baseline.model_loader")
    sys.modules["src.baseline.model_loader"] = MagicMock()

    # 5. src.data.dataset mock
    if "src.data" not in sys.modules or sys.modules["src.data"] is None:
        data_pkg = types.ModuleType("src.data")
        data_pkg.__path__ = [
            os.path.join(os.path.dirname(__file__), "..", "src", "data")
        ]
        sys.modules["src.data"] = data_pkg
    if "src.data.dataset" not in sys.modules or sys.modules["src.data.dataset"] is None:
        sys.modules["src.data.dataset"] = MagicMock()

    # 6. 기존 깨진 모듈 항목 제거
    saved["src.baseline.evaluate_zero_shot"] = sys.modules.pop(
        "src.baseline.evaluate_zero_shot", None
    )

    # 7. 새 모듈 객체를 만들고 exec()으로 소스 실행
    mod = types.ModuleType("src.baseline.evaluate_zero_shot")
    mod.__file__ = os.path.join(
        os.path.dirname(__file__), "..", "src", "baseline", "evaluate_zero_shot.py"
    )
    mod.__package__ = "src.baseline"
    sys.modules["src.baseline.evaluate_zero_shot"] = mod

    src_path = os.path.normpath(mod.__file__)
    with open(src_path, encoding="utf-8") as f:
        source = f.read()

    code = compile(source, src_path, "exec")
    exec(code, mod.__dict__)  # noqa: S102

    # 8. torch를 원래대로 복원
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

# import 실패 시 모든 테스트 skip
pytestmark = pytest.mark.skipif(
    _MODULE is None,
    reason=f"evaluate_zero_shot import 실패: {_IMPORT_ERROR}",
)


def _make_sample(question: str, answer: str, qtype: str):
    """가짜 VQA 샘플 생성."""
    s = MagicMock()
    s.image = MagicMock()
    s.question = question
    s.answer = answer
    s.question_type = qtype
    return s


# --- _is_correct 함수 테스트 ---


def test_is_correct_closed_yes():
    """closed 질문에서 yes/yes 일치 시 True."""
    assert _MODULE._is_correct("yes", "yes", "closed") is True


def test_is_correct_closed_mismatch():
    """closed 질문에서 불일치 시 False."""
    assert _MODULE._is_correct("yes", "no", "closed") is False


def test_is_correct_open_exact():
    """open 질문에서 정확히 일치 시 True."""
    assert _MODULE._is_correct("the heart", "the heart", "open") is True


def test_is_correct_open_recall():
    """open 질문에서 gold가 prediction에 포함되면 True."""
    assert _MODULE._is_correct("it is the heart and lungs", "heart", "open") is True


def test_is_correct_open_mismatch():
    """open 질문에서 불일치 시 False."""
    assert _MODULE._is_correct("liver", "heart", "open") is False


def test_is_correct_closed_verbose():
    """closed 질문에서 verbose 출력도 처리."""
    assert _MODULE._is_correct("Yes, the image shows...", "yes", "closed") is True


# --- evaluate_with_loaded_model 테스트 ---


@patch.object(_MODULE, "load_medical_vqa_dataset")
@patch.object(_MODULE, "generate_answer")
@patch.object(_MODULE, "get_vram_usage", return_value={"peak_mb": 0, "current_mb": 0})
@patch.object(_MODULE, "reset_peak_stats")
@patch.object(_MODULE, "set_seed")
@patch.object(_MODULE, "compute_overall_accuracy")
@patch.object(_MODULE, "get_environment_info", return_value={"python_version": "3.11"})
def test_evaluate_with_loaded_model_returns_summary(
    mock_env, mock_coa, mock_seed, mock_reset, mock_vram,
    mock_gen, mock_load_ds,
):
    """evaluate_with_loaded_model이 summary dict를 반환한다."""
    samples = [
        _make_sample("Is X-ray?", "yes", "closed"),
        _make_sample("What organ?", "heart", "open"),
    ]
    mock_load_ds.return_value = samples
    mock_gen.side_effect = ["yes", "heart"]
    mock_coa.return_value = {
        "closed_accuracy": 1.0, "open_accuracy": 1.0, "overall_accuracy": 1.0,
        "closed_count": 1, "open_count": 1, "total_count": 2,
        "open_bertscore_f1": 0.9, "open_bertscore_accuracy": 1.0,
    }

    config = MagicMock()
    config.model_name = "test_model"
    config.model_id = "test/model-v1"

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = _MODULE.evaluate_with_loaded_model(
            model=MagicMock(), processor=MagicMock(), config=config,
            dataset_name="pathvqa", output_dir=tmpdir, seed=42, batch_size=1,
        )

    assert isinstance(summary, dict)
    assert "closed_accuracy" in summary


@patch.object(_MODULE, "load_medical_vqa_dataset")
@patch.object(_MODULE, "generate_answer")
@patch.object(_MODULE, "get_vram_usage", return_value={"peak_mb": 512, "current_mb": 256})
@patch.object(_MODULE, "reset_peak_stats")
@patch.object(_MODULE, "set_seed")
@patch.object(_MODULE, "compute_overall_accuracy")
@patch.object(_MODULE, "get_environment_info", return_value={"python_version": "3.11", "os": "Linux"})
def test_evaluate_saves_json_with_environment(
    mock_env, mock_coa, mock_seed, mock_reset, mock_vram,
    mock_gen, mock_load_ds,
):
    """결과 JSON에 metadata.environment가 포함된다 (REQ-RI-003)."""
    samples = [_make_sample("Q?", "yes", "closed")]
    mock_load_ds.return_value = samples
    mock_gen.side_effect = ["yes"]
    mock_coa.return_value = {
        "closed_accuracy": 1.0, "open_accuracy": 0.0, "overall_accuracy": 1.0,
        "closed_count": 1, "open_count": 0, "total_count": 1,
    }

    config = MagicMock()
    config.model_name = "test_model"
    config.model_id = "test/model-v1"

    with tempfile.TemporaryDirectory() as tmpdir:
        _MODULE.evaluate_with_loaded_model(
            model=MagicMock(), processor=MagicMock(), config=config,
            dataset_name="slake", output_dir=tmpdir, seed=42, batch_size=1,
        )

        json_files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
        assert len(json_files) == 1

        with open(os.path.join(tmpdir, json_files[0]), encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "environment" in data["metadata"]
        assert "summary" in data
        assert "per_sample" in data


@patch.object(_MODULE, "load_medical_vqa_dataset")
@patch.object(_MODULE, "generate_answers_batch")
@patch.object(_MODULE, "get_vram_usage", return_value={"peak_mb": 0, "current_mb": 0})
@patch.object(_MODULE, "reset_peak_stats")
@patch.object(_MODULE, "set_seed")
@patch.object(_MODULE, "compute_overall_accuracy")
@patch.object(_MODULE, "get_environment_info", return_value={"python_version": "3.11"})
def test_evaluate_batch_mode(
    mock_env, mock_coa, mock_seed, mock_reset, mock_vram,
    mock_gen_batch, mock_load_ds,
):
    """batch_size > 1일 때 _infer_batch 경로."""
    samples = [
        _make_sample("Q1?", "yes", "closed"),
        _make_sample("Q2?", "no", "closed"),
    ]
    mock_load_ds.return_value = samples
    mock_gen_batch.return_value = ["yes", "no"]
    mock_coa.return_value = {
        "closed_accuracy": 1.0, "open_accuracy": 0.0, "overall_accuracy": 1.0,
        "closed_count": 2, "open_count": 0, "total_count": 2,
    }

    config = MagicMock()
    config.model_name = "test_model"
    config.model_id = "test/model-v1"

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = _MODULE.evaluate_with_loaded_model(
            model=MagicMock(), processor=MagicMock(), config=config,
            dataset_name="vqa_rad", output_dir=tmpdir, seed=42, batch_size=4,
        )

    assert isinstance(summary, dict)


@patch.object(_MODULE, "load_medical_vqa_dataset")
@patch.object(_MODULE, "generate_answer")
@patch.object(_MODULE, "get_vram_usage", return_value={"peak_mb": 0, "current_mb": 0})
@patch.object(_MODULE, "reset_peak_stats")
@patch.object(_MODULE, "set_seed")
@patch.object(_MODULE, "compute_overall_accuracy")
@patch.object(_MODULE, "get_environment_info", return_value={"python_version": "3.11"})
@patch.object(_MODULE, "unload_model")
@patch.object(_MODULE, "load_model", return_value=(MagicMock(), MagicMock()))
@patch.object(_MODULE, "load_config")
def test_evaluate_single_condition_lifecycle(
    mock_load_cfg, mock_load_model, mock_unload,
    mock_env, mock_coa, mock_seed, mock_reset, mock_vram,
    mock_gen, mock_load_ds,
):
    """evaluate_single_condition이 모델을 로드/언로드한다."""
    mock_load_ds.return_value = []
    mock_cfg = MagicMock()
    mock_cfg.model_name = "t"
    mock_cfg.model_id = "t/1"
    mock_load_cfg.return_value = mock_cfg
    mock_coa.return_value = {
        "closed_accuracy": 0.0, "open_accuracy": 0.0, "overall_accuracy": 0.0,
        "closed_count": 0, "open_count": 0, "total_count": 0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        _MODULE.evaluate_single_condition(
            model_config_path="fake.yaml",
            dataset_name="pathvqa",
            output_dir=tmpdir,
        )

    mock_load_model.assert_called_once()
    mock_unload.assert_called_once()


@patch.object(_MODULE, "load_medical_vqa_dataset")
@patch.object(_MODULE, "generate_answer")
@patch.object(_MODULE, "get_vram_usage", return_value={"peak_mb": 0, "current_mb": 0})
@patch.object(_MODULE, "reset_peak_stats")
@patch.object(_MODULE, "set_seed")
@patch.object(_MODULE, "compute_overall_accuracy")
@patch.object(_MODULE, "get_environment_info", return_value={"python_version": "3.11"})
def test_evaluate_max_samples_limits_data(
    mock_env, mock_coa, mock_seed, mock_reset, mock_vram,
    mock_gen, mock_load_ds,
):
    """max_samples가 설정되면 데이터가 잘린다."""
    samples = [_make_sample(f"Q{i}?", "yes", "closed") for i in range(10)]
    mock_load_ds.return_value = samples
    mock_gen.return_value = "yes"
    mock_coa.return_value = {
        "closed_accuracy": 1.0, "open_accuracy": 0.0, "overall_accuracy": 1.0,
        "closed_count": 3, "open_count": 0, "total_count": 3,
    }

    config = MagicMock()
    config.model_name = "test_model"
    config.model_id = "test/model-v1"

    with tempfile.TemporaryDirectory() as tmpdir:
        _MODULE.evaluate_with_loaded_model(
            model=MagicMock(), processor=MagicMock(), config=config,
            dataset_name="pathvqa", output_dir=tmpdir, seed=42,
            batch_size=1, max_samples=3,
        )

    # generate_answer는 max_samples=3이므로 3번만 호출
    assert mock_gen.call_count == 3
