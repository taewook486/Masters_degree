"""strategies.py 테스트 (REQ-RI-004).

ManualStrategy, RandomSearchStrategy, AutoresearchStrategy의 suggest(),
config_to_dict, get_strategy, _is_duplicate 테스트.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.autoresearch.strategies import (
    SEARCH_SPACE,
    STRATEGIES,
    AutoresearchStrategy,
    ManualStrategy,
    OptunaTPEStrategy,
    RandomSearchStrategy,
    config_to_dict,
    get_strategy,
)
from src.autoresearch.tracker import TrialResult


def _make_trial(**kwargs) -> TrialResult:
    """테스트용 TrialResult 생성."""
    defaults = {
        "trial_id": 0, "strategy": "random", "repeat_id": 0,
        "lora_rank": 16, "lora_alpha": 32, "learning_rate": 2e-4,
        "batch_size": 1, "grad_accum_steps": 8, "warmup_ratio": 0.03,
        "weight_decay": 0.01, "lora_targets": "minimal", "max_steps": 400,
        "val_accuracy": 0.5, "status": "completed",
    }
    defaults.update(kwargs)
    return TrialResult(**defaults)


def test_manual_strategy_returns_fixed_config():
    """ManualStrategy는 항상 같은 설정을 반환한다."""
    strategy = ManualStrategy()
    config = strategy.suggest([])
    assert config["lora_rank"] == 16
    assert config["lora_alpha"] == 32
    assert config["learning_rate"] == 2e-4
    assert config["lora_targets"] == "minimal"
    assert config["max_steps"] == 400


def test_manual_strategy_name():
    """ManualStrategy 이름이 'manual'이다."""
    assert ManualStrategy.name == "manual"


def test_random_strategy_returns_valid_config():
    """RandomSearchStrategy가 search space 내의 유효한 설정을 반환한다."""
    strategy = RandomSearchStrategy()
    config = strategy.suggest([])

    assert config["lora_rank"] in SEARCH_SPACE["lora_rank"]
    assert config["batch_size"] in SEARCH_SPACE["batch_size"]
    assert config["grad_accum_steps"] in SEARCH_SPACE["grad_accum_steps"]
    assert config["lora_targets"] in SEARCH_SPACE["lora_targets"]
    assert config["max_steps"] in SEARCH_SPACE["max_steps"]

    lr_lo, lr_hi = SEARCH_SPACE["learning_rate"]
    assert lr_lo <= config["learning_rate"] <= lr_hi

    wu_lo, wu_hi = SEARCH_SPACE["warmup_ratio"]
    assert wu_lo <= config["warmup_ratio"] <= wu_hi

    wd_lo, wd_hi = SEARCH_SPACE["weight_decay"]
    assert wd_lo <= config["weight_decay"] <= wd_hi


def test_random_strategy_alpha_is_rank_multiple():
    """lora_alpha는 lora_rank * ratio여야 한다."""
    strategy = RandomSearchStrategy()
    config = strategy.suggest([])
    ratio = config["lora_alpha"] / config["lora_rank"]
    assert ratio in SEARCH_SPACE["lora_alpha_ratio"]


def test_config_to_dict():
    """config_to_dict가 TrialResult에서 올바른 dict를 추출한다."""
    trial = _make_trial(lora_rank=32, lora_alpha=64, learning_rate=1e-4)
    d = config_to_dict(trial)
    assert d["lora_rank"] == 32
    assert d["lora_alpha"] == 64
    assert d["learning_rate"] == 1e-4
    assert d["max_steps"] == 400


def test_get_strategy_manual():
    """get_strategy('manual')가 ManualStrategy를 반환한다."""
    s = get_strategy("manual")
    assert isinstance(s, ManualStrategy)


def test_get_strategy_random():
    """get_strategy('random')가 RandomSearchStrategy를 반환한다."""
    s = get_strategy("random")
    assert isinstance(s, RandomSearchStrategy)


def test_get_strategy_unknown_raises():
    """알 수 없는 전략 이름이면 ValueError를 발생시킨다."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("nonexistent")


def test_strategies_dict_has_all_four():
    """STRATEGIES에 4개 전략이 모두 포함된다."""
    assert "manual" in STRATEGIES
    assert "random" in STRATEGIES
    assert "optuna" in STRATEGIES
    assert "autoresearch" in STRATEGIES


def test_autoresearch_is_duplicate_exact_match():
    """_is_duplicate가 정확히 같은 설정을 감지한다 (REQ-RI-004)."""
    strategy = AutoresearchStrategy()
    trial = _make_trial(
        lora_rank=16, lora_alpha=32, learning_rate=2e-4,
        batch_size=1, grad_accum_steps=8, lora_targets="minimal", max_steps=400,
    )
    config = {
        "lora_rank": 16, "lora_alpha": 32, "learning_rate": 2e-4,
        "batch_size": 1, "grad_accum_steps": 8, "warmup_ratio": 0.03,
        "weight_decay": 0.01, "lora_targets": "minimal", "max_steps": 400,
    }
    assert strategy._is_duplicate(config, [trial]) is True


def test_autoresearch_is_duplicate_different_config():
    """_is_duplicate가 다른 설정에서는 False를 반환한다."""
    strategy = AutoresearchStrategy()
    trial = _make_trial(lora_rank=16, lora_alpha=32)
    config = {
        "lora_rank": 32, "lora_alpha": 64, "learning_rate": 2e-4,
        "batch_size": 1, "grad_accum_steps": 8, "warmup_ratio": 0.03,
        "weight_decay": 0.01, "lora_targets": "minimal", "max_steps": 400,
    }
    assert strategy._is_duplicate(config, [trial]) is False


def test_autoresearch_is_duplicate_skips_failed():
    """_is_duplicate가 실패한 trial은 무시한다."""
    strategy = AutoresearchStrategy()
    trial = _make_trial(status="failed")
    config = config_to_dict(trial)
    config["warmup_ratio"] = trial.warmup_ratio
    config["weight_decay"] = trial.weight_decay
    assert strategy._is_duplicate(config, [trial]) is False


def test_autoresearch_suggest_falls_back_on_api_error():
    """API 실패 시 RandomSearchStrategy로 폴백한다."""
    strategy = AutoresearchStrategy()
    with patch(
        "src.autoresearch.agent.ask_agent_for_config",
        side_effect=RuntimeError("API fail"),
    ):
        config = strategy.suggest([])
    assert "lora_rank" in config
    assert config["lora_rank"] in SEARCH_SPACE["lora_rank"]


# --- OptunaTPEStrategy 테스트 (lines 150, 154-187, 191-197, 210-217) ---


def _make_mock_optuna():
    """optuna mock 생성 헬퍼."""
    import numpy as np

    mock_optuna = MagicMock()

    # 실제 배포 객체처럼 동작하도록 설정
    mock_optuna.distributions.CategoricalDistribution = MagicMock(
        side_effect=lambda x: x
    )
    mock_optuna.distributions.FloatDistribution = MagicMock(
        side_effect=lambda lo, hi: (lo, hi)
    )

    # study.ask()가 반환하는 trial mock
    mock_trial = MagicMock()
    mock_trial.params = {
        "lora_rank": 16,
        "lora_alpha_ratio": 2,
        "log_learning_rate": float(np.log(2e-4)),
        "batch_size": 1,
        "grad_accum_steps": 8,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "lora_targets": "minimal",
        "max_steps": 400,
    }

    mock_study = MagicMock()
    mock_study.ask.return_value = mock_trial
    mock_optuna.create_study.return_value = mock_study
    mock_optuna.samplers.TPESampler.return_value = MagicMock()
    mock_optuna.logging.WARNING = 30

    return mock_optuna, mock_study, mock_trial


def test_optuna_tpe_suggest_empty_history():
    """히스토리 없이 suggest()가 유효한 설정을 반환한다 (lines 209-227)."""
    mock_optuna, mock_study, mock_trial = _make_mock_optuna()

    strategy = OptunaTPEStrategy()
    with patch.dict(sys.modules, {"optuna": mock_optuna}):
        config = strategy.suggest([])

    assert config["lora_rank"] == 16
    assert config["lora_alpha"] == 32  # rank * alpha_ratio = 16 * 2
    assert "learning_rate" in config
    assert "lora_targets" in config


def test_optuna_tpe_ensure_study_with_history():
    """완료된 trial이 있으면 study에 warm-start 데이터를 추가한다 (lines 163-187)."""
    mock_optuna, mock_study, mock_trial = _make_mock_optuna()
    mock_optuna.trial.create_trial = MagicMock(return_value=MagicMock())

    completed_trial = _make_trial(
        lora_rank=16, lora_alpha=32, learning_rate=2e-4,
        batch_size=1, grad_accum_steps=8, warmup_ratio=0.03,
        weight_decay=0.01, lora_targets="minimal", max_steps=400,
        val_accuracy=0.75, status="completed",
    )

    strategy = OptunaTPEStrategy()
    with patch.dict(sys.modules, {"optuna": mock_optuna}):
        strategy._ensure_study([completed_trial])

    # warm-start: add_trial이 한 번 호출되어야 함
    mock_study.add_trial.assert_called_once()


def test_optuna_tpe_ensure_study_skips_failed():
    """실패한 trial은 warm-start에 포함하지 않는다 (lines 165)."""
    mock_optuna, mock_study, _ = _make_mock_optuna()

    failed_trial = _make_trial(status="failed", val_accuracy=0.0)

    strategy = OptunaTPEStrategy()
    with patch.dict(sys.modules, {"optuna": mock_optuna}):
        strategy._ensure_study([failed_trial])

    # 실패 trial은 add_trial 호출 없음
    mock_study.add_trial.assert_not_called()


def test_optuna_tpe_distributions_returns_dict():
    """_distributions()가 9개 키를 가진 dict를 반환한다 (lines 190-207)."""
    mock_optuna, _, _ = _make_mock_optuna()

    with patch.dict(sys.modules, {"optuna": mock_optuna}):
        dists = OptunaTPEStrategy._distributions()

    assert isinstance(dists, dict)
    expected_keys = {
        "lora_rank", "lora_alpha_ratio", "log_learning_rate",
        "batch_size", "grad_accum_steps", "warmup_ratio",
        "weight_decay", "lora_targets", "max_steps",
    }
    assert expected_keys == set(dists.keys())


def test_get_strategy_optuna():
    """get_strategy('optuna')가 OptunaTPEStrategy를 반환한다."""
    s = get_strategy("optuna")
    assert isinstance(s, OptunaTPEStrategy)
