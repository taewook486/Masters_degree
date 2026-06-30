"""재현성 유틸리티 테스트 (REQ-011).

set_seed 함수가 random, numpy, torch 시드를 올바르게 설정하는지 확인한다.
"""

from __future__ import annotations

import os
import random
import sys
from unittest.mock import MagicMock

# Mock heavy modules before importing src
_mock_np = MagicMock()
_mock_np.random = MagicMock()
sys.modules.setdefault("numpy", _mock_np)
sys.modules.setdefault("numpy.random", _mock_np.random)

_mock_torch = MagicMock()
_mock_torch.cuda.is_available.return_value = False
_mock_torch.device = MagicMock()
_mock_torch.Generator = MagicMock
_mock_torch.backends = MagicMock()
_mock_torch.backends.cudnn = MagicMock()
sys.modules.setdefault("torch", _mock_torch)
sys.modules.setdefault("torch.cuda", _mock_torch.cuda)
sys.modules.setdefault("torch.backends", _mock_torch.backends)
sys.modules.setdefault("torch.backends.cudnn", _mock_torch.backends.cudnn)

from src.utils.seed import EXPERIMENT_SEEDS, set_seed  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Python random 모듈 재현성 테스트
# ---------------------------------------------------------------------------


def test_set_seed_python_random_reproducible():
    """set_seed(42) 호출 후 random.random() 결과가 동일해야 한다."""
    set_seed(42)
    val1 = random.random()

    set_seed(42)
    val2 = random.random()

    assert val1 == val2


def test_set_seed_different_seeds_different_results():
    """다른 시드 값은 다른 랜덤 결과를 생성해야 한다."""
    set_seed(42)
    val_42 = random.random()

    set_seed(123)
    val_123 = random.random()

    assert val_42 != val_123


# ---------------------------------------------------------------------------
# torch mock 호출 검증
# ---------------------------------------------------------------------------


def test_set_seed_calls_torch_manual_seed():
    """set_seed가 torch.manual_seed를 올바른 시드로 호출해야 한다."""
    # seed.py가 import한 torch 모듈의 mock을 직접 참조
    torch_mod = sys.modules["torch"]
    torch_mod.manual_seed.reset_mock()
    torch_mod.cuda.manual_seed_all.reset_mock()

    set_seed(42)

    torch_mod.manual_seed.assert_called_with(42)
    torch_mod.cuda.manual_seed_all.assert_called_with(42)


def test_set_seed_sets_cudnn_deterministic():
    """set_seed가 cudnn.deterministic=True, benchmark=False를 설정해야 한다."""
    set_seed(42)

    # backends.cudnn 속성이 설정되었는지 확인
    torch_mod = sys.modules["torch"]
    assert torch_mod.backends.cudnn.deterministic is True
    assert torch_mod.backends.cudnn.benchmark is False


# ---------------------------------------------------------------------------
# numpy mock 호출 검증
# ---------------------------------------------------------------------------


def test_set_seed_calls_numpy_seed():
    """set_seed가 np.random.seed를 올바른 시드로 호출해야 한다."""
    np_mod = sys.modules["numpy"]
    np_mod.random.seed.reset_mock()

    set_seed(456)

    np_mod.random.seed.assert_called_with(456)


# ---------------------------------------------------------------------------
# 환경 변수 및 기타 테스트
# ---------------------------------------------------------------------------


def test_set_seed_sets_pythonhashseed():
    """set_seed가 PYTHONHASHSEED 환경 변수를 설정해야 한다."""
    set_seed(42)
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_set_seed_callable_without_args():
    """set_seed()가 기본 인자(42)로 정상 호출되어야 한다."""
    set_seed()
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_experiment_seeds_constant():
    """EXPERIMENT_SEEDS 상수가 [42, 123, 456]이어야 한다."""
    assert EXPERIMENT_SEEDS == [42, 123, 456]


def test_set_seed_all_experiment_seeds():
    """모든 실험 시드에 대해 set_seed가 오류 없이 동작해야 한다."""
    for seed in EXPERIMENT_SEEDS:
        set_seed(seed)
        assert os.environ["PYTHONHASHSEED"] == str(seed)
