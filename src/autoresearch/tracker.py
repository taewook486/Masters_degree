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

from filelock import FileLock

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
        # run_phase3.py --max_parallel로 여러 프로세스가 같은 results.tsv를
        # 공유할 때(REQ: (전략,repeat) 단위 병렬 실행), 헤더 작성/append/
        # trial_id 발급이 서로 겹치지 않도록 프로세스 간 파일 락으로 직렬화한다.
        self._lock = FileLock(str(self.path) + ".lock")

        with self._lock:
            if not self.path.exists():
                self._write_header()

    def _write_header(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
            writer.writeheader()

    def append(self, trial: TrialResult) -> None:
        """Append a single trial result."""
        with self._lock:
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
        """다음 trial_id를 예약해서 반환한다.

        --max_parallel로 여러 프로세스가 서로 다른 (전략,repeat) job을
        동시에 돌릴 때, 이 id는 학습이 끝나는 수십 분 뒤에야 append()로
        results.tsv에 실제로 남는다. results.tsv를 스캔해 max+1만 구하면
        그 사이에 다른 프로세스가 같은 id를 또 받아가는 race가 생긴다.
        그래서 예약 시점에 카운터 파일(.counter)에 즉시 다음 값을
        기록해, 아직 append되지 않은 id도 "이미 나간 id"로 잡히게 한다.
        """
        counter_path = Path(str(self.path) + ".counter")
        with self._lock:
            if counter_path.exists():
                next_id = int(counter_path.read_text().strip())
            else:
                trials = self.load_all()
                next_id = max((t.trial_id for t in trials), default=-1) + 1
            counter_path.write_text(str(next_id + 1))
        return next_id

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
