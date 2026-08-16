"""HPO strategy implementations for Phase 3.

Four strategies compared (THESIS_PROPOSAL.md Section 4.5):
  1. Manual      - researcher's default config (1 trial)
  2. Random      - uniform random sampling from search space
  3. Optuna TPE  - Bayesian optimization (Tree-structured Parzen Estimator)
  4. Autoresearch - LLM agent proposes next config based on history
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from src.autoresearch.tracker import TrialResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search space definition (THESIS_PROPOSAL.md Table)
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "lora_rank": [4, 8, 16, 32, 64],
    "lora_alpha_ratio": [1, 2, 4],  # alpha = rank * ratio
    "learning_rate": (1e-5, 5e-4),  # continuous, log-scale
    "batch_size": [1, 2, 4],
    "grad_accum_steps": [4, 8, 16],
    "warmup_ratio": (0.0, 0.1),  # continuous
    "weight_decay": (0.0, 0.1),  # continuous
    "lora_targets": ["minimal", "medium", "full"],
}

# v0.11: max_steps는 탐색 대상이 아니라 전 trial 공통 고정값이다(설계서 §4.5 "고정
# 조건" — trial 간 동일 학습 step 수 보장). v0.2~v0.10 구간에서는 SEARCH_SPACE에
# 남아 있어 탐색 공간 표(탐색 가능)와 고정 조건 문단(200 고정)이 서로 모순됐고,
# RandomSearch/Optuna가 실제로 {100,200,400,800} 중에서 표본을 뽑고 있었다.
PHASE3_FIXED_MAX_STEPS = 200


def config_to_dict(trial: TrialResult) -> dict:
    """Extract hyperparameter dict from a TrialResult."""
    return {
        "lora_rank": trial.lora_rank,
        "lora_alpha": trial.lora_alpha,
        "learning_rate": trial.learning_rate,
        "batch_size": trial.batch_size,
        "grad_accum_steps": trial.grad_accum_steps,
        "warmup_ratio": trial.warmup_ratio,
        "weight_decay": trial.weight_decay,
        "lora_targets": trial.lora_targets,
        "max_steps": trial.max_steps,
    }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class HPOStrategy(ABC):
    """Base class for HPO strategies."""

    name: str
    last_reasoning: str = ""  # Raw LLM reasoning (autoresearch only)

    @abstractmethod
    def suggest(self, history: list[TrialResult]) -> dict:
        """Suggest next hyperparameter configuration.

        Args:
            history: List of all trials (completed + failed) for this strategy+repeat.

        Returns:
            Dict of hyperparameters.
        """
        ...


# ---------------------------------------------------------------------------
# 1. Manual strategy
# ---------------------------------------------------------------------------

class ManualStrategy(HPOStrategy):
    """Researcher's hand-picked default configuration."""

    name = "manual"

    def suggest(self, history: list[TrialResult]) -> dict:
        return {
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "batch_size": 1,
            "grad_accum_steps": 8,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "lora_targets": "minimal",
            "max_steps": PHASE3_FIXED_MAX_STEPS,
        }


# ---------------------------------------------------------------------------
# 2. Random Search
# ---------------------------------------------------------------------------

class RandomSearchStrategy(HPOStrategy):
    """Uniform random sampling from the search space."""

    name = "random"

    def suggest(self, history: list[TrialResult]) -> dict:
        rank = random.choice(SEARCH_SPACE["lora_rank"])
        alpha_ratio = random.choice(SEARCH_SPACE["lora_alpha_ratio"])

        lr_lo, lr_hi = SEARCH_SPACE["learning_rate"]
        lr = np.exp(random.uniform(np.log(lr_lo), np.log(lr_hi)))

        wu_lo, wu_hi = SEARCH_SPACE["warmup_ratio"]
        wd_lo, wd_hi = SEARCH_SPACE["weight_decay"]

        return {
            "lora_rank": rank,
            "lora_alpha": rank * alpha_ratio,
            "learning_rate": round(float(lr), 6),
            "batch_size": random.choice(SEARCH_SPACE["batch_size"]),
            "grad_accum_steps": random.choice(SEARCH_SPACE["grad_accum_steps"]),
            "warmup_ratio": round(random.uniform(wu_lo, wu_hi), 4),
            "weight_decay": round(random.uniform(wd_lo, wd_hi), 4),
            "lora_targets": random.choice(SEARCH_SPACE["lora_targets"]),
            "max_steps": PHASE3_FIXED_MAX_STEPS,
        }


# ---------------------------------------------------------------------------
# 3. Optuna TPE
# ---------------------------------------------------------------------------

class OptunaTPEStrategy(HPOStrategy):
    """Bayesian optimization using Optuna's Tree-structured Parzen Estimator.

    Creates one Optuna study per (strategy, repeat_id) combination.
    The study is created fresh each time suggest() is called,
    with completed trials added to warm-start the TPE sampler.
    """

    name = "optuna"

    def __init__(self) -> None:
        self._study: Any = None

    def _ensure_study(self, history: list[TrialResult]) -> Any:
        """Create/recreate study and register past trials."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
        )

        # Warm-start with completed trials
        for t in history:
            if t.status != "completed":
                continue
            params = {
                "lora_rank": t.lora_rank,
                "lora_alpha_ratio": t.lora_alpha // t.lora_rank if t.lora_rank > 0 else 2,
                "log_learning_rate": float(np.log(t.learning_rate)),
                "batch_size": t.batch_size,
                "grad_accum_steps": t.grad_accum_steps,
                "warmup_ratio": t.warmup_ratio,
                "weight_decay": t.weight_decay,
                "lora_targets": t.lora_targets,
            }
            study.add_trial(
                optuna.trial.create_trial(
                    params=params,
                    distributions=self._distributions(),
                    values=[t.val_accuracy],
                )
            )

        self._study = study
        return study

    @staticmethod
    def _distributions() -> dict:
        import optuna

        lr_lo, lr_hi = SEARCH_SPACE["learning_rate"]
        wu_lo, wu_hi = SEARCH_SPACE["warmup_ratio"]
        wd_lo, wd_hi = SEARCH_SPACE["weight_decay"]

        return {
            "lora_rank": optuna.distributions.CategoricalDistribution(SEARCH_SPACE["lora_rank"]),
            "lora_alpha_ratio": optuna.distributions.CategoricalDistribution(SEARCH_SPACE["lora_alpha_ratio"]),
            "log_learning_rate": optuna.distributions.FloatDistribution(float(np.log(lr_lo)), float(np.log(lr_hi))),
            "batch_size": optuna.distributions.CategoricalDistribution(SEARCH_SPACE["batch_size"]),
            "grad_accum_steps": optuna.distributions.CategoricalDistribution(SEARCH_SPACE["grad_accum_steps"]),
            "warmup_ratio": optuna.distributions.FloatDistribution(wu_lo, wu_hi),
            "weight_decay": optuna.distributions.FloatDistribution(wd_lo, wd_hi),
            "lora_targets": optuna.distributions.CategoricalDistribution(SEARCH_SPACE["lora_targets"]),
        }

    def suggest(self, history: list[TrialResult]) -> dict:
        study = self._ensure_study(history)
        trial = study.ask(self._distributions())

        rank = trial.params["lora_rank"]
        alpha_ratio = trial.params["lora_alpha_ratio"]
        lr = np.exp(trial.params["log_learning_rate"])

        return {
            "lora_rank": rank,
            "lora_alpha": rank * alpha_ratio,
            "learning_rate": round(float(lr), 6),
            "batch_size": trial.params["batch_size"],
            "grad_accum_steps": trial.params["grad_accum_steps"],
            "warmup_ratio": round(trial.params["warmup_ratio"], 4),
            "weight_decay": round(trial.params["weight_decay"], 4),
            "lora_targets": trial.params["lora_targets"],
            "max_steps": PHASE3_FIXED_MAX_STEPS,
        }


# ---------------------------------------------------------------------------
# 4. Autoresearch (LLM Agent)
# ---------------------------------------------------------------------------

class AutoresearchStrategy(HPOStrategy):
    """LLM agent-based autonomous HPO.

    The agent reads the experiment history summary and proposes the next config.
    Uses the Anthropic API (Claude) to generate suggestions.
    Falls back to random search if the API call fails.
    """

    name = "autoresearch"

    def __init__(
        self,
        program_md_path: str = "configs/autoresearch/program.md",
        total_trials: int = 40,
        max_tokens: int = 512,
    ):
        self.program_md_path = program_md_path
        self.total_trials = total_trials
        self.max_tokens = max_tokens
        self._program: str | None = None

    def _load_program(self) -> str:
        if self._program is None:
            path = Path(self.program_md_path)
            if path.exists():
                self._program = path.read_text(encoding="utf-8")
            else:
                logger.warning(f"program.md not found at {path}, using default")
                self._program = _DEFAULT_PROGRAM
        return self._program

    def _is_duplicate(self, config: dict, history: list[TrialResult]) -> bool:
        """이전 완료 trial과 동일한 하이퍼파라미터 설정인지 확인한다."""
        # 비교 대상 키 (연속형 값은 반올림하여 비교)
        compare_keys = [
            "lora_rank", "lora_alpha", "batch_size",
            "grad_accum_steps", "lora_targets", "max_steps",
        ]
        for trial in history:
            if trial.status != "completed":
                continue
            existing = config_to_dict(trial)
            match = True
            for key in compare_keys:
                if config.get(key) != existing.get(key):
                    match = False
                    break
            if match:
                # 연속형 값도 비교 (learning_rate, warmup_ratio, weight_decay)
                lr_match = abs(config.get("learning_rate", 0) - existing.get("learning_rate", 0)) < 1e-5
                wu_match = abs(config.get("warmup_ratio", 0) - existing.get("warmup_ratio", 0)) < 1e-4
                wd_match = abs(config.get("weight_decay", 0) - existing.get("weight_decay", 0)) < 1e-4
                if lr_match and wu_match and wd_match:
                    return True
        return False

    def suggest(self, history: list[TrialResult]) -> dict:
        from src.autoresearch.agent import ask_agent_for_config

        program = self._load_program()
        completed = [t for t in history if t.status == "completed"]
        failed = [t for t in history if t.status == "failed"]

        # Build history summary for the agent
        if not history:
            history_text = "No previous trials. Start with an exploratory configuration."
        elif not completed:
            history_text = "No completed trials yet."
        else:
            # Chronological order for sequential reasoning
            lines = ["Previous experiment results (chronological order):", ""]
            lines.append(
                "trial | rank | alpha | lr       | bs | ga | targets | steps | val_acc | loss"
            )
            lines.append("-" * 90)
            for t in sorted(completed, key=lambda x: x.trial_id)[-20:]:
                lines.append(
                    f"{t.trial_id:5d} | {t.lora_rank:4d} | {t.lora_alpha:5d} | "
                    f"{t.learning_rate:.1e} | {t.batch_size:2d} | {t.grad_accum_steps:2d} | "
                    f"{t.lora_targets:7s} | {t.max_steps:5d} | {t.val_accuracy:.4f} | "
                    f"{t.train_loss:.4f}"
                )
            best = max(completed, key=lambda x: x.val_accuracy)
            lines.append(f"\nBest so far: trial {best.trial_id}, val_acc={best.val_accuracy:.4f}")
            lines.append(f"Total completed: {len(completed)}")
            history_text = "\n".join(lines)

        # Append failed trials so agent avoids re-suggesting OOM configs
        if failed:
            fail_lines = [f"\n## Failed Trials ({len(failed)} total - AVOID these configs):"]
            for t in sorted(failed, key=lambda x: x.trial_id)[-10:]:
                fail_lines.append(
                    f"  trial {t.trial_id}: {t.notes} | "
                    f"rank={t.lora_rank}, alpha={t.lora_alpha}, "
                    f"targets={t.lora_targets}, bs={t.batch_size}"
                )
            history_text += "\n".join(fail_lines)

        trial_number = len(completed)

        # REQ-RI-004: 중복 설정 감지 및 재시도 (최대 3회)
        _MAX_DUPLICATE_RETRIES = 3
        try:
            config = None
            for attempt in range(_MAX_DUPLICATE_RETRIES):
                config, reasoning = ask_agent_for_config(
                    program,
                    history_text,
                    trial_number=trial_number,
                    total_trials=self.total_trials,
                    max_tokens=self.max_tokens,
                )
                if not self._is_duplicate(config, history):
                    break
                logger.warning(
                    f"[Autoresearch] Duplicate config detected (attempt {attempt + 1}/{_MAX_DUPLICATE_RETRIES}), "
                    f"requesting re-proposal"
                )
                # 재시도 시 history_text에 중복 경고 추가
                history_text += (
                    f"\n\n## WARNING: Your last suggestion was a duplicate of a previous trial. "
                    f"Please suggest a DIFFERENT configuration."
                )
            self.last_reasoning = reasoning
            logger.info(f"[Autoresearch] Agent suggested: {config}")
            return config
        except Exception as e:
            logger.warning(f"[Autoresearch] Agent failed ({e}), falling back to random")
            self.last_reasoning = ""
            return RandomSearchStrategy().suggest(history)


class AutoresearchV2Strategy(AutoresearchStrategy):
    """설정 불일치를 제거한 autoresearch 재실험 조건.

    원본 AutoresearchStrategy는 최초 실행 결과의 재현을 위해 그대로 두고,
    프롬프트와 코드가 어긋나 있던 지점만 맞춘 별도 조건으로 분리한다.
    탐색 공간 자체는 원본과 동일하므로 random/optuna와의 비교는 유지된다.

    원본 대비 차이 5건:
      1. epochs 제거 — agent.py가 config.pop("epochs")로 값을 버리고
         max_steps를 PHASE3_FIXED_MAX_STEPS(200)로 고정하므로 무효였다.
      2. 단계 일정을 절대 trial 번호에서 예산 비율로 변경 — agent.py의
         _build_user_message가 주입하는 힌트가 비율 기준이라 충돌했다.
      3. 근거 산출 허용 — 원본의 "JSON only" 제약이 RQ3의 둘째 요건인
         해석 가능한 탐색 근거를 측정 이전에 배제했다.
      4. 중복 금지 명시 — _is_duplicate가 중복을 거부하고 재제안을
         요구하는데 원본 프롬프트는 이를 알리지 않고 중복을 유도했다.
      5. total_trials를 실제 예산으로 주입 — 호출부가 넘기지 않아 기본값
         40이 쓰였고, 예산 20에서는 단계·온도 일정이 절반에서 멈췄다.
         (주입은 run_phase3.py / run_one_repeat.py 호출부에서 수행)

    max_tokens는 근거 산문이 앞에 붙어도 JSON이 잘리지 않도록 올린다.
    """

    name = "autoresearch_v2"

    def __init__(
        self,
        program_md_path: str = "configs/autoresearch/program_v2.md",
        total_trials: int = 40,
        max_tokens: int = 1024,
    ):
        super().__init__(
            program_md_path=program_md_path,
            total_trials=total_trials,
            max_tokens=max_tokens,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[HPOStrategy]] = {
    "manual": ManualStrategy,
    "random": RandomSearchStrategy,
    "optuna": OptunaTPEStrategy,
    "autoresearch": AutoresearchStrategy,
    "autoresearch_v2": AutoresearchV2Strategy,
}


def get_strategy(name: str, **kwargs) -> HPOStrategy:
    """Create an HPO strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)


# ---------------------------------------------------------------------------
# Default program.md content (fallback)
# ---------------------------------------------------------------------------

_DEFAULT_PROGRAM = """
You are an autonomous hyperparameter optimization agent for medical VQA fine-tuning.

Your task: Given the history of previous experiments, suggest the NEXT hyperparameter
configuration that is most likely to improve validation accuracy.

Search space:
- lora_rank: {4, 8, 16, 32, 64}
- lora_alpha: rank * {1, 2, 4}
- learning_rate: [1e-5, 5e-4] (log-scale)
- batch_size: {1, 2, 4}
- grad_accum_steps: {4, 8, 16}
- warmup_ratio: [0.0, 0.1]
- weight_decay: [0.0, 0.1]
- lora_targets: {"minimal", "medium", "full"}

Note: max_steps is fixed at 200 for every trial (not tunable) so that all
trials are trained on the same step budget for a fair comparison. Do not
suggest a value for it.

Strategy guidelines:
1. Early trials (0-5): Explore diverse configurations
2. Mid trials (5-20): Exploit promising regions, vary 1-2 params from best
3. Late trials (20+): Fine-tune around the best configuration

Respond with ONLY a JSON object, no other text.
"""
