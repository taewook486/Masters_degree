"""Experiment result tracker for Phase 3 HPO.

Manages results.tsv: each row is one HPO trial with config + metrics.
Provides read/write/query operations for all 4 HPO strategies.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """A single HPO trial record."""

    trial_id: int
    strategy: str  # "manual", "random", "optuna", "autoresearch"
    repeat_id: int  # Which independent repeat (0-4)

    # Hyperparameters
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_accum_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lora_targets: str = "minimal"  # "minimal", "medium", "full"
    epochs: int = 3

    # Results
    val_accuracy: float = 0.0
    val_closed_acc: float = 0.0
    val_open_acc: float = 0.0
    train_loss: float = 0.0
    train_time_min: float = 0.0
    peak_vram_mb: float = 0.0

    # Meta
    status: str = "pending"  # "pending", "running", "completed", "failed"
    notes: str = ""


TSV_COLUMNS = [f.name for f in fields(TrialResult)]


class ExperimentTracker:
    """Read/write experiment results to a TSV file."""

    def __init__(self, results_path: str | Path):
        self.path = Path(results_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            self._write_header()

    def _write_header(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
            writer.writeheader()

    def append(self, trial: TrialResult) -> None:
        """Append a single trial result."""
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
            writer.writerow(asdict(trial))
        logger.info(
            f"Trial {trial.trial_id} recorded: val_acc={trial.val_accuracy:.4f} "
            f"({trial.strategy}, repeat={trial.repeat_id})"
        )

    def load_all(self) -> list[TrialResult]:
        """Load all trial results from TSV."""
        if not self.path.exists():
            return []

        results = []
        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                trial = TrialResult(
                    trial_id=int(row["trial_id"]),
                    strategy=row["strategy"],
                    repeat_id=int(row["repeat_id"]),
                    lora_rank=int(row["lora_rank"]),
                    lora_alpha=int(row["lora_alpha"]),
                    learning_rate=float(row["learning_rate"]),
                    batch_size=int(row["batch_size"]),
                    grad_accum_steps=int(row["grad_accum_steps"]),
                    warmup_ratio=float(row["warmup_ratio"]),
                    weight_decay=float(row["weight_decay"]),
                    lora_targets=row["lora_targets"],
                    epochs=int(row["epochs"]),
                    val_accuracy=float(row["val_accuracy"]),
                    val_closed_acc=float(row["val_closed_acc"]),
                    val_open_acc=float(row["val_open_acc"]),
                    train_loss=float(row["train_loss"]),
                    train_time_min=float(row["train_time_min"]),
                    peak_vram_mb=float(row["peak_vram_mb"]),
                    status=row["status"],
                    notes=row.get("notes", ""),
                )
                results.append(trial)
        return results

    def load_by_strategy(self, strategy: str, repeat_id: int | None = None) -> list[TrialResult]:
        """Load trials filtered by strategy and optionally repeat_id."""
        all_trials = self.load_all()
        filtered = [t for t in all_trials if t.strategy == strategy]
        if repeat_id is not None:
            filtered = [t for t in filtered if t.repeat_id == repeat_id]
        return filtered

    def next_trial_id(self) -> int:
        """Get the next available trial_id."""
        trials = self.load_all()
        if not trials:
            return 0
        return max(t.trial_id for t in trials) + 1

    def best_trial(self, strategy: str, repeat_id: int | None = None) -> TrialResult | None:
        """Get the best trial by val_accuracy for a strategy."""
        trials = self.load_by_strategy(strategy, repeat_id)
        completed = [t for t in trials if t.status == "completed"]
        if not completed:
            return None
        return max(completed, key=lambda t: t.val_accuracy)

    def summary_text(self, strategy: str, repeat_id: int) -> str:
        """Generate a human-readable summary of trials for the LLM agent."""
        trials = self.load_by_strategy(strategy, repeat_id)
        completed = [t for t in trials if t.status == "completed"]

        if not completed:
            return "No completed trials yet."

        lines = [
            f"Strategy: {strategy}, Repeat: {repeat_id}",
            f"Completed trials: {len(completed)}",
            f"Best val_accuracy: {max(t.val_accuracy for t in completed):.4f}",
            "",
            "trial_id | rank | alpha | lr       | bs | ga | targets | epochs | val_acc | loss",
            "-" * 95,
        ]
        for t in sorted(completed, key=lambda x: -x.val_accuracy):
            lines.append(
                f"{t.trial_id:8d} | {t.lora_rank:4d} | {t.lora_alpha:5d} | "
                f"{t.learning_rate:.1e} | {t.batch_size:2d} | {t.grad_accum_steps:2d} | "
                f"{t.lora_targets:7s} | {t.epochs:6d} | {t.val_accuracy:.4f} | {t.train_loss:.4f}"
            )
        return "\n".join(lines)
