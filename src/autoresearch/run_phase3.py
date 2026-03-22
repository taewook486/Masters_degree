"""Run all Phase 3 HPO experiments.

THESIS_PROPOSAL.md Section 4.5:
  - 4 strategies x 5 independent repeats
  - ~40 trials per strategy per repeat (manual = 1 trial)
  - Fixed model (best from Phase 2) + fixed dataset (PathVQA)

Usage:
    python -m src.autoresearch.run_phase3 \
        --model_config configs/models/qwen25_vl_3b.yaml \
        --output_dir results/phase3_autoresearch \
        --strategies manual random optuna autoresearch \
        --repeats 5 \
        --trials_per_repeat 40
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.autoresearch.loop import run_hpo_loop
from src.autoresearch.strategies import get_strategy
from src.autoresearch.tracker import ExperimentTracker

logger = logging.getLogger(__name__)

DATASET = "pathvqa"  # Fixed dataset for Phase 3


def run_phase3(
    model_config_path: str,
    base_finetune_config: str,
    output_dir: str,
    strategies: list[str],
    n_repeats: int = 5,
    trials_per_repeat: int = 40,
    seed: int = 42,
    data_dir: str = "data",
    time_budget_min: float = 15.0,
) -> pd.DataFrame:
    """Run all Phase 3 HPO experiments.

    Args:
        model_config_path: Best model config from Phase 2.
        base_finetune_config: Base QLoRA config.
        output_dir: Output directory.
        strategies: List of strategy names to run.
        n_repeats: Number of independent repeats per strategy.
        trials_per_repeat: Max trials per repeat.
        seed: Base random seed.
        data_dir: Dataset directory.
        time_budget_min: Training time budget per trial.

    Returns:
        Summary DataFrame.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker(output_path / "results.tsv")

    for strategy_name in strategies:
        for repeat_id in range(n_repeats):
            # Check if this combination already has enough trials
            existing = tracker.load_by_strategy(strategy_name, repeat_id)
            completed = [t for t in existing if t.status == "completed"]

            expected = 1 if strategy_name == "manual" else trials_per_repeat
            if len(completed) >= expected:
                logger.info(
                    f"SKIP: {strategy_name}/repeat{repeat_id} "
                    f"already has {len(completed)}/{expected} completed trials"
                )
                continue

            remaining = expected - len(completed)
            logger.info(
                f"\n{'#'*60}\n"
                f"Strategy: {strategy_name}, Repeat: {repeat_id}, "
                f"Trials to run: {remaining}\n"
                f"{'#'*60}"
            )

            strategy = get_strategy(strategy_name)
            repeat_seed = seed + repeat_id * 1000  # Different seed per repeat

            run_hpo_loop(
                strategy=strategy,
                model_config_path=model_config_path,
                base_finetune_config=base_finetune_config,
                dataset_name=DATASET,
                tracker=tracker,
                repeat_id=repeat_id,
                output_dir=str(output_path / f"{strategy_name}_repeat{repeat_id}"),
                max_trials=remaining,
                seed=repeat_seed,
                data_dir=data_dir,
                time_budget_min=time_budget_min,
            )

    # Build summary
    return _build_phase3_summary(tracker, output_path)


def _build_phase3_summary(
    tracker: ExperimentTracker,
    output_path: Path,
) -> pd.DataFrame:
    """Build Phase 3 summary: best accuracy per strategy per repeat."""
    import numpy as np

    all_trials = tracker.load_all()
    completed = [t for t in all_trials if t.status == "completed"]

    if not completed:
        logger.warning("No completed trials found")
        return pd.DataFrame()

    # Group by strategy
    strategies = sorted(set(t.strategy for t in completed))
    summary_rows = []

    for strategy in strategies:
        st_trials = [t for t in completed if t.strategy == strategy]
        repeats = sorted(set(t.repeat_id for t in st_trials))

        best_accs = []
        total_trials = []
        for r in repeats:
            r_trials = [t for t in st_trials if t.repeat_id == r]
            if r_trials:
                best = max(t.val_accuracy for t in r_trials)
                best_accs.append(best)
                total_trials.append(len(r_trials))

        if best_accs:
            summary_rows.append({
                "strategy": strategy,
                "n_repeats": len(repeats),
                "best_acc_mean": round(float(np.mean(best_accs)), 4),
                "best_acc_std": round(float(np.std(best_accs)), 4),
                "best_acc_max": round(float(np.max(best_accs)), 4),
                "avg_trials": round(float(np.mean(total_trials)), 1),
                "total_trials": sum(total_trials),
            })

    df = pd.DataFrame(summary_rows)

    # Save
    csv_path = output_path / "phase3_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nPhase 3 summary saved to {csv_path}")
    logger.info(f"\n{df.to_string(index=False)}")

    # Also save per-trial details
    detail_rows = []
    for t in completed:
        detail_rows.append({
            "trial_id": t.trial_id,
            "strategy": t.strategy,
            "repeat_id": t.repeat_id,
            "val_accuracy": t.val_accuracy,
            "train_loss": t.train_loss,
            "train_time_min": t.train_time_min,
            "lora_rank": t.lora_rank,
            "learning_rate": t.learning_rate,
            "lora_targets": t.lora_targets,
            "epochs": t.epochs,
        })
    detail_df = pd.DataFrame(detail_rows)
    detail_path = output_path / "phase3_all_trials.csv"
    detail_df.to_csv(detail_path, index=False)
    logger.info(f"Detailed results: {detail_path}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 HPO experiments")
    parser.add_argument(
        "--model_config", required=True,
        help="Path to best model config YAML (from Phase 2 results)",
    )
    parser.add_argument(
        "--finetune_config", default="configs/finetune/base_qlora.yaml",
    )
    parser.add_argument("--output_dir", default="results/phase3_autoresearch")
    parser.add_argument(
        "--strategies", nargs="+",
        default=["manual", "random", "optuna", "autoresearch"],
        choices=["manual", "random", "optuna", "autoresearch"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trials_per_repeat", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument(
        "--time_budget_min", type=float, default=15.0,
        help="Training time budget per trial in minutes (default: 15)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_phase3(
        model_config_path=args.model_config,
        base_finetune_config=args.finetune_config,
        output_dir=args.output_dir,
        strategies=args.strategies,
        n_repeats=args.repeats,
        trials_per_repeat=args.trials_per_repeat,
        seed=args.seed,
        data_dir=args.data_dir,
        time_budget_min=args.time_budget_min,
    )


if __name__ == "__main__":
    main()
