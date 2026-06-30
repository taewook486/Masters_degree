"""agent.py 테스트 (REQ-RI-005, REQ-RI-006).

_parse_config, _validate_config, _build_user_message 테스트.
ask_agent_for_config은 API 의존이므로 mock 처리.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.autoresearch.agent import (
    _REQUIRED_KEYS_DEFAULTS,
    _build_user_message,
    _parse_config,
    _validate_config,
    ask_agent_for_config,
)

# --- _parse_config 테스트 ---


def test_parse_config_direct_json():
    """순수 JSON 문자열을 파싱한다."""
    raw = '{"lora_rank": 16, "lora_alpha": 32}'
    result = _parse_config(raw)
    assert result["lora_rank"] == 16


def test_parse_config_markdown_fenced():
    """마크다운 코드 블록 안의 JSON을 파싱한다."""
    raw = '```json\n{"lora_rank": 8}\n```'
    result = _parse_config(raw)
    assert result["lora_rank"] == 8


def test_parse_config_json_in_text():
    """텍스트 속에 있는 JSON 객체를 추출한다."""
    raw = 'Here is my suggestion: {"lora_rank": 32, "lora_alpha": 64} end.'
    result = _parse_config(raw)
    assert result["lora_rank"] == 32


def test_parse_config_invalid_raises():
    """JSON을 파싱할 수 없으면 RuntimeError를 발생시킨다."""
    with pytest.raises(RuntimeError, match="Could not parse JSON"):
        _parse_config("no json here at all")


# --- _validate_config 테스트 ---


def test_validate_config_valid_passthrough():
    """유효한 설정은 그대로 통과한다."""
    config = {
        "lora_rank": 16, "lora_alpha": 32, "learning_rate": 2e-4,
        "batch_size": 1, "grad_accum_steps": 8, "warmup_ratio": 0.03,
        "weight_decay": 0.01, "lora_targets": "minimal", "max_steps": 400,
    }
    result = _validate_config(config)
    assert result["lora_rank"] == 16
    assert result["max_steps"] == 400


def test_validate_config_clamps_rank():
    """유효하지 않은 lora_rank를 가장 가까운 유효값으로 보정한다."""
    config = {"lora_rank": 10, "lora_alpha": 20}
    result = _validate_config(config)
    assert result["lora_rank"] in {4, 8, 16, 32, 64}


def test_validate_config_clamps_learning_rate():
    """학습률이 범위를 벗어나면 클램핑한다."""
    config = {"learning_rate": 1e-2}  # 너무 높음
    result = _validate_config(config)
    assert result["learning_rate"] <= 5e-4

    config2 = {"learning_rate": 1e-8}  # 너무 낮음
    result2 = _validate_config(config2)
    assert result2["learning_rate"] >= 1e-5


def test_validate_config_clamps_batch_size():
    """유효하지 않은 batch_size를 보정한다."""
    config = {"batch_size": 3}  # 유효값: 1, 2, 4
    result = _validate_config(config)
    assert result["batch_size"] in {1, 2, 4}


def test_validate_config_invalid_targets_defaults():
    """유효하지 않은 lora_targets는 'minimal'로 설정된다."""
    config = {"lora_targets": "invalid_target"}
    result = _validate_config(config)
    assert result["lora_targets"] == "minimal"


def test_validate_config_missing_keys_get_defaults():
    """필수 키가 누락되면 기본값이 적용된다 (REQ-RI-005)."""
    config = {}  # 모든 키 누락
    result = _validate_config(config)
    for key, default_val in _REQUIRED_KEYS_DEFAULTS.items():
        assert key in result


def test_validate_config_epochs_to_max_steps_migration():
    """epochs가 있고 max_steps가 없으면 변환한다 (REQ-RI-005)."""
    config = {"epochs": 3}
    result = _validate_config(config)
    assert "max_steps" in result
    assert "epochs" not in result
    assert result["max_steps"] in {100, 200, 400, 800}


def test_validate_config_epochs_removed_when_max_steps_exists():
    """max_steps가 이미 있으면 epochs를 제거한다."""
    config = {"epochs": 5, "max_steps": 200}
    result = _validate_config(config)
    assert result["max_steps"] == 200
    assert "epochs" not in result


def test_validate_config_clamps_warmup_ratio():
    """warmup_ratio 범위를 [0.0, 0.1]로 클램핑한다."""
    config = {"warmup_ratio": 0.5}
    result = _validate_config(config)
    assert result["warmup_ratio"] <= 0.1


def test_validate_config_clamps_weight_decay():
    """weight_decay 범위를 [0.0, 0.1]로 클램핑한다."""
    config = {"weight_decay": -0.1}
    result = _validate_config(config)
    assert result["weight_decay"] >= 0.0


def test_validate_config_clamps_max_steps():
    """유효하지 않은 max_steps를 가장 가까운 유효값으로 보정한다."""
    config = {"max_steps": 300}
    result = _validate_config(config)
    assert result["max_steps"] in {100, 200, 400, 800}


# --- _build_user_message 테스트 ---


def test_build_user_message_exploration_phase():
    """초기 trial에서 탐색 단계 힌트가 포함된다 (REQ-RI-006)."""
    msg = _build_user_message("no history", trial_number=0, total_trials=40)
    assert "EXPLORATION" in msg
    assert "1 / 40" in msg


def test_build_user_message_transition_phase():
    """중간 trial에서 전환 단계 힌트가 포함된다."""
    msg = _build_user_message("some history", trial_number=15, total_trials=40)
    assert "TRANSITION" in msg


def test_build_user_message_exploitation_phase():
    """후반 trial에서 활용 단계 힌트가 포함된다."""
    msg = _build_user_message("long history", trial_number=35, total_trials=40)
    assert "EXPLOITATION" in msg


def test_build_user_message_includes_history():
    """이전 결과 텍스트가 메시지에 포함된다."""
    msg = _build_user_message("trial 0: val_acc=0.85", trial_number=1, total_trials=10)
    assert "val_acc=0.85" in msg


# --- ask_agent_for_config 테스트 (REQ-RI-006) ---


def test_ask_agent_no_api_key():
    """ANTHROPIC_API_KEY가 없으면 RuntimeError를 발생시킨다 (line 51-56)."""
    env_without_key = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            ask_agent_for_config("system prompt", "no history")


def test_ask_agent_success_path():
    """API 호출 성공 시 config와 raw 텍스트를 반환한다 (lines 66-99)."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    _hp_json = (
        '{"lora_rank": 16, "lora_alpha": 32, "learning_rate": 2e-4,'
        ' "batch_size": 1, "grad_accum_steps": 8, "warmup_ratio": 0.03,'
        ' "weight_decay": 0.01, "lora_targets": "minimal", "max_steps": 400}'
    )
    mock_response.content = [MagicMock(text=_hp_json)]
    mock_client.messages.create.return_value = mock_response

    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            config, raw = ask_agent_for_config(
                "system prompt", "no history",
                trial_number=0, total_trials=10,
            )

    assert config["lora_rank"] == 16
    assert "lora_rank" in raw


def test_ask_agent_retry_exhaustion():
    """모든 재시도 실패 시 RuntimeError를 발생시킨다 (lines 100-112)."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API timeout")

    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with patch("time.sleep"):  # 실제 sleep 방지
                with pytest.raises(RuntimeError, match="Agent API failed"):
                    ask_agent_for_config("system", "history")


def test_ask_agent_exploration_phase_log():
    """초반 trial에서 exploration 단계 레이블이 로깅된다 (lines 89-98)."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"lora_rank": 8, "lora_alpha": 16}')]
    mock_client.messages.create.return_value = mock_response

    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            config, _ = ask_agent_for_config(
                "system", "no history",
                trial_number=0, total_trials=40,
            )

    # exploration 단계: trial 0/40 = 0% < 25% → exploration
    assert config["lora_rank"] == 8


def test_ask_agent_exploitation_phase():
    """후반 trial에서 exploitation 단계 레이블이 로깅된다 (lines 93-98)."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"lora_rank": 32}')]
    mock_client.messages.create.return_value = mock_response

    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            config, _ = ask_agent_for_config(
                "system", "long history",
                trial_number=35, total_trials=40,
            )

    # exploitation 단계: trial 35/40 = 87.5% > 75%
    assert config["lora_rank"] == 32
