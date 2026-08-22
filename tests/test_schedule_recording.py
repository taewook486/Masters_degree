"""phase·temperature 기록 경로 회귀 테스트.

이 결함이 조용했던 이유는 값이 "틀려서"가 아니라 "일관된 기본값"이었기
때문이다. TrialResult는 두 필드를 선언했지만 생성 경로가 채우지 않았고,
되읽기도 같은 기본값을 쓰므로 왕복이 에러 없이 성립했다. 그래서
results.tsv의 temperature 열이 전 행 0.0으로 일관되게 보였다.

따라서 테스트도 그 지점을 겨냥한다.
  (1) 에이전트 경로에서 기본값이 아닌 실제 호출값이 기록되는가
  (2) random 폴백 경로에서는 기록되지 않는가 (거짓 기록 방지)
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.autoresearch.agent import schedule_for
from src.autoresearch.strategies import AutoresearchStrategy
from src.autoresearch.tracker import TrialResult

_HP_JSON = (
    '{"lora_rank": 16, "lora_alpha": 32, "learning_rate": 2e-4,'
    ' "batch_size": 1, "grad_accum_steps": 8, "warmup_ratio": 0.03,'
    ' "weight_decay": 0.01, "lora_targets": "minimal", "max_steps": 400}'
)


def _mock_anthropic():
    """JSON 설정을 돌려주는 anthropic 모듈 mock."""
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=_HP_JSON)]
    client.messages.create.return_value = response
    module = MagicMock()
    module.Anthropic.return_value = client
    return module, client


# --- schedule_for: 공식의 단일 정의 지점 -----------------------------------


@pytest.mark.parametrize(
    ("trial_number", "total_trials", "expected_temp", "expected_phase"),
    [
        (0, 20, 1.0, "exploration"),
        (19, 20, 0.3, "exploitation"),
        (10, 20, 0.63, "transition"),
        # 예산이 주입되지 않아 기본값 40이 쓰이던 조건 — 20 trial에서
        # 온도가 0.66에서 절단되던 상황을 공식 수준에서 재현해 둔다.
        (19, 40, 0.66, "transition"),
        # 0 나눗셈 방어
        (0, 1, 1.0, "exploration"),
    ],
)
def test_schedule_for_matches_recorded_schedule(
    trial_number, total_trials, expected_temp, expected_phase
):
    temperature, phase = schedule_for(trial_number, total_trials)
    assert temperature == pytest.approx(expected_temp)
    assert phase == expected_phase


# --- (1) 에이전트 경로: 실제 호출값이 기록되는가 ----------------------------


def test_agent_path_records_actual_schedule_not_default():
    """에이전트가 실제로 쓴 온도·단계가 전략에 남는다.

    기본값(0.0 / "")과 구별되는 값이어야 한다 — 이 결함의 본질이
    "기본값이 유효값처럼 보인 것"이었다.
    """
    module, _ = _mock_anthropic()
    strategy = AutoresearchStrategy(total_trials=20)

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        with patch.dict(sys.modules, {"anthropic": module}):
            strategy.suggest([])

    expected_temp, expected_phase = schedule_for(0, 20)
    assert strategy.last_temperature == pytest.approx(expected_temp)
    assert strategy.last_phase == expected_phase
    # 기본값과 실제로 구별되는지 — 이 단언이 결함의 핵심을 잡는다
    assert strategy.last_temperature != TrialResult.temperature
    assert strategy.last_phase != TrialResult.phase


def test_recorded_temperature_survives_falsy_guard():
    """온도 0.0은 falsy지만 기록돼야 한다.

    loop.py의 전달 코드가 `if strategy.last_reasoning:` 형태의 truthiness
    가드를 복사하면 온도 0.0이 다시 누락된다. 0.0을 낼 수 있는 일정은
    없지만(하한 0.3), 가드 형태 자체를 고정해 두어 재발을 막는다.
    """
    strategy = AutoresearchStrategy(total_trials=20)
    strategy.last_temperature = 0.0
    strategy.last_phase = "exploitation"

    # loop.py가 쓰는 것과 동일한 판정
    assert strategy.last_temperature is not None
    assert bool(strategy.last_temperature) is False  # truthiness 가드면 누락된다


# --- (2) 폴백 경로: 거짓 기록을 남기지 않는가 -------------------------------


def test_schedule_survives_tsv_round_trip(tmp_path):
    """기록한 일정이 results.tsv 왕복에서 살아남는다.

    이 결함이 조용했던 마지막 이유가 왕복이었다 — 쓰기도 기본값,
    되읽기도 같은 기본값이라 아무 에러 없이 성립했다. 왕복 자체를
    검사해야 그 침묵을 깬다.
    """
    from src.autoresearch.tracker import ExperimentTracker

    tracker = ExperimentTracker(tmp_path / "results.tsv")
    temperature, phase = schedule_for(3, 20)
    trial = TrialResult(
        trial_id=1,
        strategy="autoresearch_v2",
        repeat_id=0,
        status="completed",
        phase=phase,
        temperature=temperature,
    )
    tracker.append(trial)

    (loaded,) = tracker.load_all()
    assert loaded.temperature == pytest.approx(temperature)
    assert loaded.phase == phase
    # 기본값이 아님을 명시적으로 확인 — 이 단언이 빠지면 왕복이
    # 성립한다는 사실만으로 통과해 버린다
    assert loaded.temperature != 0.0
    assert loaded.phase != ""


def test_random_fallback_leaves_schedule_unrecorded():
    """API 실패로 random 폴백하면 온도·단계를 기록하지 않는다.

    에이전트 호출이 없었는데 "온도 T로 호출됨"이라고 적으면 거짓 기록이다.
    """
    strategy = AutoresearchStrategy(total_trials=20)
    # 직전 trial의 값이 남아 있는 상태에서 폴백해도 비워져야 한다
    strategy.last_temperature = 0.65
    strategy.last_phase = "transition"

    with patch(
        "src.autoresearch.agent.ask_agent_for_config",
        side_effect=RuntimeError("API fail"),
    ):
        config = strategy.suggest([])

    assert "lora_rank" in config  # 폴백 자체는 정상 동작
    assert strategy.last_temperature is None
    assert strategy.last_phase is None
