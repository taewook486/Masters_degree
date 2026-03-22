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
    "epochs": [1, 2, 3, 5],
}


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
        "epochs": trial.epochs,
    }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class HPOStrategy(ABC):
    """Base class for HPO strategies."""

    name: str

    @abstractmethod
    def suggest(self, history: list[TrialResult]) -> dict:
        """Suggest next hyperparameter configuration.

        Args:
            history: List of completed trials (for this strategy+repeat).

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
            "epochs": 3,
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
            "epochs": random.choice(SEARCH_SPACE["epochs"]),
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
                "epochs": t.epochs,
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
            "epochs": optuna.distributions.CategoricalDistribution(SEARCH_SPACE["epochs"]),
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
            "epochs": trial.params["epochs"],
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

    def __init__(self, program_md_path: str = "configs/autoresearch/program.md"):
        self.program_md_path = program_md_path
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

    def suggest(self, history: list[TrialResult]) -> dict:
        from src.autoresearch.agent import ask_agent_for_config

        program = self._load_program()

        # Build history summary for the agent
        if not history:
            history_text = "No previous trials. Start with an exploratory configuration."
        else:
            completed = [t for t in history if t.status == "completed"]
            if not completed:
                history_text = "No completed trials yet. Start with an exploratory configuration."
            else:
                lines = ["Previous experiment results (sorted by val_accuracy desc):", ""]
                lines.append(
                    "trial | rank | alpha | lr       | bs | ga | targets | epochs | val_acc | loss"
                )
                lines.append("-" * 90)
                for t in sorted(completed, key=lambda x: -x.val_accuracy)[:20]:
                    lines.append(
                        f"{t.trial_id:5d} | {t.lora_rank:4d} | {t.lora_alpha:5d} | "
                        f"{t.learning_rate:.1e} | {t.batch_size:2d} | {t.grad_accum_steps:2d} | "
                        f"{t.lora_targets:7s} | {t.epochs:6d} | {t.val_accuracy:.4f} | "
                        f"{t.train_loss:.4f}"
                    )
                best = max(completed, key=lambda x: x.val_accuracy)
                lines.append(f"\nBest so far: trial {best.trial_id}, val_acc={best.val_accuracy:.4f}")
                lines.append(f"Total completed: {len(completed)}")
                history_text = "\n".join(lines)

        try:
            config = ask_agent_for_config(program, history_text)
            logger.info(f"[Autoresearch] Agent suggested: {config}")
            return config
        except Exception as e:
            logger.warning(f"[Autoresearch] Agent failed ({e}), falling back to random")
            return RandomSearchStrategy().suggest(history)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[HPOStrategy]] = {
    "manual": ManualStrategy,
    "random": RandomSearchStrategy,
    "optuna": OptunaTPEStrategy,
    "autoresearch": AutoresearchStrategy,
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
- epochs: {1, 2, 3, 5}

Strategy guidelines:
1. Early trials (0-5): Explore diverse configurations
2. Mid trials (5-20): Exploit promising regions, vary 1-2 params from best
3. Late trials (20+): Fine-tune around the best configuration

Respond with ONLY a JSON object, no other text.
"""
