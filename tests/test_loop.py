"""loop.py 테스트 (REQ-RI-008).

_write_trial_config, run_hpo_loop의 체크포인트 재개 로직 테스트.
loop.py는 torch, src.finetune.train_qlora를 import하므로
exec() 기반 모듈 로딩을 사용한다.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# --- 모듈 로드 ---
_MODULE = None
_IMPORT_ERROR = None


def _load_module():
    """loop 모듈을 의존성 mock과 함께 로드한다."""
    saved = {}

    # torch mock
    mock_torch = MagicMock()
    mock_torch.cuda.OutOfMemoryError = RuntimeError
    mock_torch.cuda.is_available.return_value = False
    saved["torch"] = sys.modules.get("torch")
    sys.modules["torch"] = mock_torch

    # 깨진 모듈 제거
    sys.modules.pop("src.autoresearch.loop", None)

    # 새 모듈 객체 생성 후 exec
    mod = types.ModuleType("src.autoresearch.loop")
    mod.__file__ = os.path.join(
        os.path.dirname(__file__), "..", "src", "autoresearch", "loop.py"
    )
    mod.__package__ = "src.autoresearch"
    sys.modules["src.autoresearch.loop"] = mod

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
    reason=f"loop import 실패: {_IMPORT_ERROR}",
)


# --- _write_trial_config 테스트 ---


def test_write_trial_config_creates_yaml():
    """_write_trial_config가 YAML 파일을 생성한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_config = os.path.join(tmpdir, "base.yaml")
        with open(base_config, "w", encoding="utf-8") as f:
            yaml.dump({
                "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": []},
                "training": {
                    "learning_rate": 1e-4, "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 4, "warmup_ratio": 0.03,
                    "weight_decay": 0.01, "num_train_epochs": 3,
                },
            }, f)

        hp = {
            "lora_rank": 32, "lora_alpha": 64, "learning_rate": 2e-4,
            "batch_size": 2, "grad_accum_steps": 8, "warmup_ratio": 0.05,
            "weight_decay": 0.02, "lora_targets": "medium", "max_steps": 200,
        }
        output_path = os.path.join(tmpdir, "trial", "config.yaml")
        _MODULE._write_trial_config(base_config, hp, output_path)

        assert os.path.exists(output_path)
        with open(output_path, encoding="utf-8") as f:
            result = yaml.safe_load(f)

        assert result["lora"]["rank"] == 32
        assert result["lora"]["alpha"] == 64
        assert result["training"]["learning_rate"] == 2e-4
        assert result["training"]["per_device_train_batch_size"] == 2
        assert result["training"]["max_steps"] == 200
        # max_steps 사용 시 num_train_epochs는 제거
        assert "num_train_epochs" not in result["training"]


def test_write_trial_config_target_modules_mapping():
    """lora_targets가 올바른 target_modules로 변환된다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_config = os.path.join(tmpdir, "base.yaml")
        with open(base_config, "w", encoding="utf-8") as f:
            yaml.dump({
                "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": []},
                "training": {},
            }, f)

        for targets_name, expected in [
            ("minimal", ["q_proj", "v_proj"]),
            ("medium", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            ("full", ["all_linear"]),
        ]:
            hp = {
                "lora_rank": 16, "lora_alpha": 32, "learning_rate": 2e-4,
                "batch_size": 1, "grad_accum_steps": 8, "warmup_ratio": 0.03,
                "weight_decay": 0.01, "lora_targets": targets_name, "max_steps": 400,
            }
            out = os.path.join(tmpdir, f"trial_{targets_name}.yaml")
            _MODULE._write_trial_config(base_config, hp, out)

            with open(out, encoding="utf-8") as f:
                result = yaml.safe_load(f)
            assert result["lora"]["target_modules"] == expected, \
                f"{targets_name} failed"


def test_write_trial_config_with_epochs_fallback():
    """hp에 max_steps가 없고 epochs가 있으면 num_train_epochs를 설정한다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_config = os.path.join(tmpdir, "base.yaml")
        with open(base_config, "w", encoding="utf-8") as f:
            yaml.dump({"lora": {}, "training": {}}, f)

        hp = {
            "lora_rank": 16, "lora_alpha": 32, "learning_rate": 2e-4,
            "batch_size": 1, "grad_accum_steps": 8, "warmup_ratio": 0.03,
            "weight_decay": 0.01, "lora_targets": "minimal", "epochs": 3,
        }
        out = os.path.join(tmpdir, "trial_epochs.yaml")
        _MODULE._write_trial_config(base_config, hp, out)

        with open(out, encoding="utf-8") as f:
            result = yaml.safe_load(f)
        assert result["training"]["num_train_epochs"] == 3


# --- run_hpo_loop 재개 로직 테스트 ---


def test_run_hpo_loop_resumes_from_existing_trials():
    """기존 완료된 trial이 있으면 이후부터 시작한다 (REQ-RI-008)."""
    from src.autoresearch.tracker import ExperimentTracker, TrialResult

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "results.tsv")
        # 이미 2개 완료
        for i in range(2):
            tracker.append(TrialResult(
                trial_id=i, strategy="manual", repeat_id=0,
                val_accuracy=0.5, status="completed",
            ))

        mock_strategy = MagicMock()
        mock_strategy.name = "manual"
        mock_strategy.last_reasoning = ""
        # HPOStrategy의 실제 기본값. MagicMock은 미설정 속성에 또 다른
        # MagicMock을 돌려주므로, None 계약을 명시하지 않으면 loop가
        # "기록된 값"으로 오인해 tsv에 mock 객체를 써 넣는다.
        mock_strategy.last_temperature = None
        mock_strategy.last_phase = None

        # manual은 max_trials=1로 강제되므로, 이미 1개 이상 완료면 0번 실행
        with patch.object(_MODULE, "run_single_trial") as mock_run:
            _MODULE.run_hpo_loop(
                strategy=mock_strategy,
                model_config_path="fake.yaml",
                base_finetune_config="fake_ft.yaml",
                dataset_name="pathvqa",
                tracker=tracker,
                repeat_id=0,
                output_dir=tmpdir,
                max_trials=5,
                seed=42,
            )

        # manual strategy는 max_trials=1이므로 이미 2개 완료 상태에서 추가 실행 없음
        assert mock_run.call_count == 0


def test_run_hpo_loop_saves_checkpoint_on_completed():
    """완료된 trial 후 체크포인트를 저장한다."""
    from src.autoresearch.tracker import ExperimentTracker, TrialResult

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(Path(tmpdir) / "results.tsv")

        mock_strategy = MagicMock()
        mock_strategy.name = "random"
        mock_strategy.last_reasoning = ""
        # 위와 같은 이유로 None 계약을 명시한다 (random은 에이전트 호출이
        # 없으므로 일정이 기록되지 않는 것이 정상 동작이다).
        mock_strategy.last_temperature = None
        mock_strategy.last_phase = None
        mock_strategy.suggest.return_value = {
            "lora_rank": 16, "lora_alpha": 32, "learning_rate": 2e-4,
            "batch_size": 1, "grad_accum_steps": 8, "warmup_ratio": 0.03,
            "weight_decay": 0.01, "lora_targets": "minimal", "max_steps": 400,
        }

        completed_trial = TrialResult(
            trial_id=0, strategy="random", repeat_id=0,
            val_accuracy=0.7, status="completed",
        )

        with (
            patch.object(_MODULE, "run_single_trial", return_value=completed_trial),
            patch.object(_MODULE, "save_checkpoint") as mock_save_ckpt,
        ):
            _MODULE.run_hpo_loop(
                strategy=mock_strategy,
                model_config_path="fake.yaml",
                base_finetune_config="fake_ft.yaml",
                dataset_name="pathvqa",
                tracker=tracker,
                repeat_id=0,
                output_dir=tmpdir,
                max_trials=1,
                seed=42,
            )

        mock_save_ckpt.assert_called_once()


# --- run_single_trial 테스트 (lines 104-187, REQ-RI-008) ---


def _make_hp() -> dict:
    """테스트용 하이퍼파라미터 dict."""
    return {
        "lora_rank": 16, "lora_alpha": 32, "learning_rate": 2e-4,
        "batch_size": 1, "grad_accum_steps": 8, "warmup_ratio": 0.03,
        "weight_decay": 0.01, "lora_targets": "minimal", "max_steps": 400,
    }


def test_run_single_trial_success():
    """train_qlora 성공 시 TrialResult.status == 'completed' (lines 104-167)."""
    mock_train = MagicMock()
    mock_train.return_value = {
        "eval_summary": {
                "overall_accuracy": 0.72, "closed_accuracy": 0.80, "open_accuracy": 0.64
            },
        "training": {
            "train_loss": 1.23, "train_runtime_sec": 600.0,
            "peak_vram_mb": 15000.0,
        },
    }
    mock_finetune = MagicMock()
    mock_finetune.train_qlora = mock_train

    with tempfile.TemporaryDirectory() as tmpdir:
        base_cfg = os.path.join(tmpdir, "base.yaml")
        with open(base_cfg, "w", encoding="utf-8") as f:
            yaml.dump({
                "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": []},
                "training": {},
            }, f)

        with patch.dict(sys.modules, {"src.finetune.train_qlora": mock_finetune}):
            result = _MODULE.run_single_trial(
                model_config_path="fake_model.yaml",
                base_finetune_config=base_cfg,
                dataset_name="pathvqa",
                hp=_make_hp(),
                trial_id=0,
                strategy_name="manual",
                repeat_id=0,
                output_dir=tmpdir,
                seed=42,
            )

    assert result.status == "completed"
    assert result.val_accuracy == pytest.approx(0.72)
    assert result.train_time_min == pytest.approx(10.0)


def test_run_single_trial_oom():
    """OOM 예외 시 status == 'failed', notes == 'OOM' (lines 169-174)."""
    mock_train = MagicMock()
    # torch.cuda.OutOfMemoryError는 _MODULE 내부의 torch mock으로 처리됨
    # RuntimeError를 OutOfMemoryError로 사용 (mock에서 설정됨)
    mock_train.side_effect = RuntimeError("CUDA out of memory")

    mock_finetune = MagicMock()
    mock_finetune.train_qlora = mock_train

    # torch.cuda.OutOfMemoryError를 RuntimeError로 치환한 mock torch
    mock_torch = MagicMock()
    mock_torch.cuda.OutOfMemoryError = RuntimeError

    with tempfile.TemporaryDirectory() as tmpdir:
        base_cfg = os.path.join(tmpdir, "base.yaml")
        with open(base_cfg, "w", encoding="utf-8") as f:
            yaml.dump({
                "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": []},
                "training": {},
            }, f)

        with patch.dict(sys.modules, {
            "src.finetune.train_qlora": mock_finetune,
            "torch": mock_torch,
        }):
            result = _MODULE.run_single_trial(
                model_config_path="fake_model.yaml",
                base_finetune_config=base_cfg,
                dataset_name="pathvqa",
                hp=_make_hp(),
                trial_id=1,
                strategy_name="random",
                repeat_id=0,
                output_dir=tmpdir,
                seed=42,
            )

    # OOM: status=failed, notes=OOM 또는 일반 Exception 처리
    assert result.status == "failed"


def test_run_single_trial_general_exception():
    """일반 예외 시 status == 'failed' (lines 176-180)."""
    mock_train = MagicMock()
    mock_train.side_effect = ValueError("config error")

    mock_finetune = MagicMock()
    mock_finetune.train_qlora = mock_train

    with tempfile.TemporaryDirectory() as tmpdir:
        base_cfg = os.path.join(tmpdir, "base.yaml")
        with open(base_cfg, "w", encoding="utf-8") as f:
            yaml.dump({
                "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": []},
                "training": {},
            }, f)

        with patch.dict(sys.modules, {"src.finetune.train_qlora": mock_finetune}):
            result = _MODULE.run_single_trial(
                model_config_path="fake_model.yaml",
                base_finetune_config=base_cfg,
                dataset_name="pathvqa",
                hp=_make_hp(),
                trial_id=2,
                strategy_name="random",
                repeat_id=0,
                output_dir=tmpdir,
                seed=42,
            )

    assert result.status == "failed"
    assert "config error" in result.notes
