"""Experiment result tracker for Phase 3 HPO.

Manages results.tsv: each row is one HPO trial with config + metrics.
Provides read/write/query operations for all 4 HPO strategies.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
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
    max_steps: int = 400  # v0.2: max_steps replaces epochs for Phase 3

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
    agent_reasoning: str = ""  # Raw LLM reasoning (autoresearch only)

    # REQ-RI-006: 탐색/활용 전환 메타데이터
    phase: str = ""  # "exploration", "transition", "exploitation"
    temperature: float = 0.0  # 에이전트 온도 스케줄링 값


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
                    max_steps=int(row["max_steps"]),
                    val_accuracy=float(row["val_accuracy"]),
                    val_closed_acc=float(row["val_closed_acc"]),
                    val_open_acc=float(row["val_open_acc"]),
                    train_loss=float(row["train_loss"]),
                    train_time_min=float(row["train_time_min"]),
                    peak_vram_mb=float(row["peak_vram_mb"]),
                    status=row["status"],
                    notes=row.get("notes", ""),
                    agent_reasoning=row.get("agent_reasoning", ""),
                    phase=row.get("phase", "unknown"),
                    temperature=float(row.get("temperature", 0.0)),
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

    def export_json(self, output_path: str | Path, strategy: str | None = None) -> Path:
        """결과를 metadata/summary 구조의 JSON으로 내보낸다 (REQ-RI-009).

        Args:
            output_path: JSON 파일 저장 경로.
            strategy: 특정 전략만 내보내기 (None이면 전체).

        Returns:
            저장된 JSON 파일 경로.
        """
        trials = self.load_all()
        if strategy:
            trials = [t for t in trials if t.strategy == strategy]

        completed = [t for t in trials if t.status == "completed"]
        best = max(completed, key=lambda t: t.val_accuracy) if completed else None

        result = {
            "metadata": {
                "source": "autoresearch_hpo",
                "strategy": strategy or "all",
                "total_trials": len(trials),
                "completed_trials": len(completed),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "summary": {
                "best_val_accuracy": best.val_accuracy if best else 0.0,
                "best_trial_id": best.trial_id if best else None,
                "avg_val_accuracy": (
                    round(sum(t.val_accuracy for t in completed) / len(completed), 4)
                    if completed else 0.0
                ),
            },
            "trials": [asdict(t) for t in trials],
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON 내보내기 완료: {out} ({len(trials)} trials)")
        return out

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
            "trial_id | rank | alpha | lr       | bs | ga | targets | steps | val_acc | loss",
            "-" * 95,
        ]
        for t in sorted(completed, key=lambda x: -x.val_accuracy):
            lines.append(
                f"{t.trial_id:8d} | {t.lora_rank:4d} | {t.lora_alpha:5d} | "
                f"{t.learning_rate:.1e} | {t.batch_size:2d} | {t.grad_accum_steps:2d} | "
                f"{t.lora_targets:7s} | {t.max_steps:5d} | {t.val_accuracy:.4f} | {t.train_loss:.4f}"
            )
        return "\n".join(lines)
