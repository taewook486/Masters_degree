"""set_seed 재현성 테스트 (REQ-011).

src/utils/seed.py의 set_seed와 EXPERIMENT_SEEDS를 테스트한다.

주의: sys.modules.setdefault 대신 patch.object를 사용한다.
      테스트 실행 순서에 관계없이 seed 모듈의 np/torch를 안전하게 교체하기 위함.
"""

from __future__ import annotations

import os
import random
from unittest.mock import MagicMock, patch

import pytest

import src.utils.seed as _seed_module
from src.utils.seed import EXPERIMENT_SEEDS, set_seed

# ---------------------------------------------------------------------------
# 모듈 수준 mock 객체 (각 테스트에서 재사용하되 fixture가 call history 초기화)
# ---------------------------------------------------------------------------

_mock_np = MagicMock()
_mock_np.random = MagicMock()

_mock_torch = MagicMock()
_mock_torch.cuda = MagicMock()
_mock_torch.backends = MagicMock()
_mock_torch.backends.cudnn = MagicMock()


@pytest.fixture(autouse=True)
def _patch_seed_deps():
    """각 테스트 전 seed 모듈의 np/torch를 mock으로 교체하고 call history 초기화.

    sys.modules를 조작하는 대신 patch.object를 사용하므로
    테스트 실행 순서와 무관하게 격리를 보장한다.
    """
    # 어설션에 사용하는 특정 mock 메서드의 call history만 초기화
    _mock_torch.manual_seed.reset_mock()
    _mock_torch.cuda.manual_seed_all.reset_mock()
    _mock_np.random.seed.reset_mock()

    with (
        patch.object(_seed_module, "np", _mock_np),
        patch.object(_seed_module, "torch", _mock_torch),
    ):
        yield


# ---------------------------------------------------------------------------
# set_seed 기능 테스트
# ---------------------------------------------------------------------------


def test_set_seed_python_random_reproducible():
    """같은 시드로 set_seed()를 두 번 호출하면 동일한 random 결과를 반환해야 한다."""
    set_seed(42)
    val1 = random.random()
    set_seed(42)
    val2 = random.random()
    assert val1 == val2


def test_set_seed_different_seeds_different_results():
    """다른 시드는 (매우 높은 확률로) 다른 결과를 낸다."""
    set_seed(42)
    val1 = random.random()
    set_seed(99)
    val2 = random.random()
    assert val1 != val2


def test_set_seed_calls_torch_manual_seed():
    """set_seed가 torch.manual_seed를 올바른 시드로 호출해야 한다."""
    set_seed(42)
    _mock_torch.manual_seed.assert_called_with(42)
    _mock_torch.cuda.manual_seed_all.assert_called_with(42)


def test_set_seed_sets_cudnn_deterministic():
    """set_seed가 torch.backends.cudnn.deterministic을 True로 설정해야 한다."""
    set_seed(42)
    assert _mock_torch.backends.cudnn.deterministic is True
    assert _mock_torch.backends.cudnn.benchmark is False


def test_set_seed_calls_numpy_seed():
    """set_seed가 np.random.seed를 올바른 시드로 호출해야 한다."""
    set_seed(456)
    _mock_np.random.seed.assert_called_with(456)


def test_set_seed_sets_pythonhashseed():
    """set_seed가 PYTHONHASHSEED 환경 변수를 설정해야 한다."""
    set_seed(123)
    assert os.environ.get("PYTHONHASHSEED") == "123"


def test_set_seed_callable_without_args():
    """set_seed()가 기본 인자(42)로 호출 가능해야 한다."""
    set_seed()
    _mock_torch.manual_seed.assert_called_with(42)


def test_experiment_seeds_constant():
    """EXPERIMENT_SEEDS가 [42, 123, 456]을 포함해야 한다."""
    assert EXPERIMENT_SEEDS == [42, 123, 456]
    assert len(EXPERIMENT_SEEDS) == 3


def test_set_seed_all_experiment_seeds():
    """EXPERIMENT_SEEDS의 모든 시드에 대해 set_seed()가 오류 없이 동작해야 한다."""
    for seed in EXPERIMENT_SEEDS:
        set_seed(seed)
        _mock_torch.manual_seed.assert_called_with(seed)
        _mock_np.random.seed.assert_called_with(seed)
