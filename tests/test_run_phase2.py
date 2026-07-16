"""run_phase2.py 조건별 병렬 실행(_run_jobs, _train_condition gpu_id) 테스트.

run_phase2.py는 pandas를 import하므로(이 개발 환경 venv에는 미설치)
exec() 기반 모듈 로딩 + pandas mock을 사용한다 (tests/test_run_all.py와 동일 패턴).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_MODULE = None
_IMPORT_ERROR = None


def _load_module():
    """run_phase2 모듈을 pandas mock과 함께 로드한다."""
    saved = {}

    saved["pandas"] = sys.modules.get("pandas")
    mock_pd = MagicMock()
    mock_pd.DataFrame = MagicMock
    sys.modules["pandas"] = mock_pd

    sys.modules.pop("src.finetune.run_phase2", None)

    mod = types.ModuleType("src.finetune.run_phase2")
    mod.__file__ = os.path.join(
        os.path.dirname(__file__), "..", "src", "finetune", "run_phase2.py"
    )
    mod.__package__ = "src.finetune"
    sys.modules["src.finetune.run_phase2"] = mod

    src_path = os.path.normpath(mod.__file__)
    with open(src_path, encoding="utf-8") as f:
        source = f.read()
    code = compile(source, src_path, "exec")
    exec(code, mod.__dict__)  # noqa: S102

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
    reason=f"run_phase2 import 실패: {_IMPORT_ERROR}",
)


# --- _run_jobs: skip_existing 동작 ---


def test_run_jobs_skips_existing_results():
    """이미 결과가 있는 job은 _train_condition을 호출하지 않고 기존 결과를 재사용한다."""
    existing = {"metadata": {"seed": 42}}

    def fake_load_existing(run_dir):
        return existing if run_dir == "dir_a" else None

    calls = []

    def fake_train_condition(**kwargs):
        calls.append(kwargs)
        return {"metadata": {"seed": kwargs["seed"]}}

    jobs = [
        {"run_dir": "dir_a", "log_label": "a", "seed": 1},
        {"run_dir": "dir_b", "log_label": "b", "seed": 2},
    ]

    with patch.object(_MODULE, "_load_existing_result", side_effect=fake_load_existing), \
         patch.object(_MODULE, "_train_condition", side_effect=fake_train_condition):
        results = _MODULE._run_jobs(jobs, skip_existing=True, max_parallel=1)

    assert results[0] is existing
    assert results[1] == {"metadata": {"seed": 2}}
    # dir_a는 스킵됐으므로 _train_condition은 dir_b(seed=2)에 대해서만 호출됨
    assert len(calls) == 1
    assert calls[0]["seed"] == 2


def test_run_jobs_sequential_when_max_parallel_1_passes_gpu_id_none():
    """max_parallel=1이면 기존 동작과 동일하게 gpu_id=None으로 호출한다."""
    calls = []

    def fake_train_condition(**kwargs):
        calls.append(kwargs.get("gpu_id"))
        return {"ok": True}

    jobs = [{"run_dir": f"dir_{i}", "log_label": f"job{i}", "seed": i} for i in range(3)]

    with patch.object(_MODULE, "_load_existing_result", return_value=None), \
         patch.object(_MODULE, "_train_condition", side_effect=fake_train_condition):
        results = _MODULE._run_jobs(jobs, skip_existing=True, max_parallel=1)

    assert all(gpu_id is None for gpu_id in calls)
    assert len(results) == 3
    assert all(r == {"ok": True} for r in results)


def test_run_jobs_assigns_round_robin_gpu_ids_when_parallel():
    """max_parallel=2이면 배치 내에서 gpu_id 0/1을 슬롯별로 배정한다."""
    seen_gpu_ids = []

    def fake_train_condition(**kwargs):
        seen_gpu_ids.append(kwargs.get("gpu_id"))
        return {"seed": kwargs["seed"]}

    jobs = [{"run_dir": f"dir_{i}", "log_label": f"job{i}", "seed": i} for i in range(4)]

    with patch.object(_MODULE, "_load_existing_result", return_value=None), \
         patch.object(_MODULE, "_train_condition", side_effect=fake_train_condition):
        results = _MODULE._run_jobs(jobs, skip_existing=True, max_parallel=2)

    # 4개 job, 배치당 2개 -> gpu_id는 매 배치마다 {0,1}에서 배정됨
    assert sorted(seen_gpu_ids) == [0, 0, 1, 1]
    assert len(results) == 4
    assert {r["seed"] for r in results} == {0, 1, 2, 3}


def test_run_jobs_isolates_per_job_failure():
    """한 job이 예외를 던져도 같은 배치의 다른 job 결과는 보존된다."""
    def fake_train_condition(**kwargs):
        if kwargs["seed"] == 1:
            raise RuntimeError("boom")
        return {"seed": kwargs["seed"]}

    jobs = [
        {"run_dir": "dir_0", "log_label": "job0", "seed": 0},
        {"run_dir": "dir_1", "log_label": "job1", "seed": 1},
    ]

    with patch.object(_MODULE, "_load_existing_result", return_value=None), \
         patch.object(_MODULE, "_train_condition", side_effect=fake_train_condition):
        results = _MODULE._run_jobs(jobs, skip_existing=True, max_parallel=2)

    assert results[0] == {"seed": 0}
    assert results[1] is None  # 실패한 job은 None으로 남고 예외가 전파되지 않음


# --- _train_condition: CUDA_VISIBLE_DEVICES env 배선 ---


def test_train_condition_sets_cuda_visible_devices_when_gpu_id_given():
    """gpu_id 지정 시 서브프로세스 env에 CUDA_VISIBLE_DEVICES가 그 값으로 설정된다."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = Path(tmpdir) / "train_result.json"
        result_file.write_text(json.dumps({"ok": True}), encoding="utf-8")

        captured = {}

        def fake_run(cmd, check, env):
            captured["env"] = env
            return MagicMock(returncode=0)

        with patch.object(_MODULE.subprocess, "run", side_effect=fake_run):
            _MODULE._train_condition(
                run_dir=tmpdir,
                model_config_path="m.yaml",
                finetune_config="f.yaml",
                dataset_name="pathvqa",
                seed=42,
                data_dir="data",
                gpu_id=1,
            )

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_train_condition_does_not_set_env_when_gpu_id_none():
    """gpu_id 미지정(None) 시 서브프로세스 env를 건드리지 않는다(기존 동작과 동일)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result_file = Path(tmpdir) / "train_result.json"
        result_file.write_text(json.dumps({"ok": True}), encoding="utf-8")

        captured = {}

        def fake_run(cmd, check, env):
            captured["env"] = env
            return MagicMock(returncode=0)

        with patch.object(_MODULE.subprocess, "run", side_effect=fake_run):
            _MODULE._train_condition(
                run_dir=tmpdir,
                model_config_path="m.yaml",
                finetune_config="f.yaml",
                dataset_name="pathvqa",
                seed=42,
                data_dir="data",
            )

    assert captured["env"] is None
