"""Run all Phase 3 HPO experiments.

THESIS_PROPOSAL.md Section 4.5 (v0.5):
  - 4 strategies x 10 independent repeats (v0.4: 5 -> 10 for statistical power)
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
import concurrent.futures
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.autoresearch.loop import run_hpo_loop
from src.autoresearch.strategies import get_strategy
from src.autoresearch.tracker import ExperimentTracker
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

DATASET = "pathvqa"  # Fixed dataset for Phase 3


def _run_repeat_subprocess(
    job: dict,
    gpu_slot: int,
    model_config_path: str,
    base_finetune_config: str,
    output_dir: str,
    data_dir: str,
    time_budget_min: float,
    max_test_samples: int | None,
) -> None:
    """(전략, repeat) 조합 하나를 GPU 1장에 고정한 서브프로세스로 실행한다.

    HF_DATASETS_CACHE를 GPU별로 분리하는 이유: 2026-08-01 스모크 테스트에서
    실측된 버그(WinError 5/32) — 이 캐시 경로가 GPU 프로세스 간에 겹치면
    Dataset.from_generator()의 내부 빌더 캐시를 여러 프로세스가 동시에 쓰다가
    파일 접근 충돌로 죽는다. run_phase3_smoke_gpu0/1.bat에서 검증된 것과
    동일한 분리 방식을 그대로 따른다.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_slot)
    if platform.system() == "Windows":
        env["HF_DATASETS_CACHE"] = f"D:\\cache\\huggingface_datasets_gpu{gpu_slot}"
    else:
        env["HF_DATASETS_CACHE"] = f"/hf_cache/huggingface_datasets_gpu{gpu_slot}"

    cmd = [
        sys.executable, "-m", "src.autoresearch.run_one_repeat",
        "--model_config_path", model_config_path,
        "--finetune_config_path", base_finetune_config,
        "--output_dir", output_dir,
        "--strategy_name", job["strategy_name"],
        "--repeat_id", str(job["repeat_id"]),
        "--max_trials", str(job["expected"]),
        "--seed", str(job["repeat_seed"]),
        "--data_dir", data_dir,
        "--time_budget_min", str(time_budget_min),
    ]
    if max_test_samples is not None:
        cmd += ["--max_test_samples", str(max_test_samples)]

    logger.info(f"\n{'#'*60}\n{job['log_label']} (GPU {gpu_slot})\n{'#'*60}")
    subprocess.run(cmd, check=True, env=env)


def run_phase3(
    model_config_path: str,
    base_finetune_config: str,
    output_dir: str,
    strategies: list[str],
    n_repeats: int = 10,
    trials_per_repeat: int = 40,
    seed: int = 42,
    data_dir: str = "data",
    time_budget_min: float = 90.0,
    max_test_samples: int | None = None,
    max_parallel: int = 1,
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
        time_budget_min: Wall-clock safety cutoff per trial in minutes (not the
            controlled variable — see PHASE3_FIXED_MAX_STEPS in strategies.py).
        max_test_samples: Cap on final per-trial test-set evaluation samples
            (None = full test set, the paper-default). Reduces wall-clock cost
            without changing max_steps/repeats/trials_per_repeat — the same
            cost lever already used by measure_cross_dataset_cf.py (500
            samples). Does not affect the HPO comparison's statistical design.
        max_parallel: Number of (strategy, repeat) combinations to run
            concurrently, one physical GPU each (via CUDA_VISIBLE_DEVICES on a
            subprocess per combination — see run_one_repeat.py). Trials
            *within* one (strategy, repeat) stay sequential regardless, since
            optuna/autoresearch pick each trial's hyperparameters from the
            previous trials' results. 1 = fully sequential in-process, byte
            -identical to the pre-parallel behavior (no regression).

    Returns:
        Summary DataFrame.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker(output_path / "results.tsv")

    # Build the list of (strategy, repeat) combinations that still need trials.
    # This pre-pass is file-I/O-only (cheap) and stays sequential regardless of
    # max_parallel, mirroring run_phase2.py's _run_jobs skip_existing pre-pass.
    jobs: list[dict] = []
    for strategy_name in strategies:
        for repeat_id in range(n_repeats):
            existing = tracker.load_by_strategy(strategy_name, repeat_id)
            completed = [t for t in existing if t.status == "completed"]

            expected = 1 if strategy_name == "manual" else trials_per_repeat
            if len(completed) >= expected:
                logger.info(
                    f"SKIP: {strategy_name}/repeat{repeat_id} "
                    f"already has {len(completed)}/{expected} completed trials"
                )
                continue

            jobs.append({
                "strategy_name": strategy_name,
                "repeat_id": repeat_id,
                "expected": expected,
                "repeat_seed": seed + repeat_id * 1000,
                "log_label": f"{strategy_name}/repeat{repeat_id} "
                             f"({expected - len(completed)} trials to run)",
            })

    if max_parallel <= 1:
        # Byte-identical to the pre-parallel behavior: in-process, sequential.
        for job in jobs:
            logger.info(f"\n{'#'*60}\n{job['log_label']}\n{'#'*60}")
            # autoresearch_v2만 실제 예산을 받는다. 다른 전략 생성자는
            # 인자를 받지 않고, 원본 autoresearch는 기존 동작을 보존한다.
            extra = (
                {"total_trials": job["expected"]}
                if job["strategy_name"] == "autoresearch_v2"
                else {}
            )
            strategy = get_strategy(job["strategy_name"], **extra)
            run_hpo_loop(
                strategy=strategy,
                model_config_path=model_config_path,
                base_finetune_config=base_finetune_config,
                dataset_name=DATASET,
                tracker=tracker,
                repeat_id=job["repeat_id"],
                output_dir=str(
                    output_path / f"{job['strategy_name']}_repeat{job['repeat_id']}"
                ),
                max_trials=job["expected"],
                seed=job["repeat_seed"],
                data_dir=data_dir,
                time_budget_min=time_budget_min,
                max_test_samples=max_test_samples,
            )
    else:
        for batch_start in range(0, len(jobs), max_parallel):
            batch = jobs[batch_start : batch_start + max_parallel]
            executor_cls = concurrent.futures.ThreadPoolExecutor
            with executor_cls(max_workers=len(batch)) as executor:
                future_to_job = {
                    executor.submit(
                        _run_repeat_subprocess,
                        job=job,
                        gpu_slot=slot,
                        model_config_path=model_config_path,
                        base_finetune_config=base_finetune_config,
                        output_dir=output_dir,
                        data_dir=data_dir,
                        time_budget_min=time_budget_min,
                        max_test_samples=max_test_samples,
                    ): job
                    for slot, job in enumerate(batch)
                }
                for future in concurrent.futures.as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"FAILED: {job['log_label']}: {e}")

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
            "max_steps": t.max_steps,
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
        choices=["manual", "random", "optuna", "autoresearch", "autoresearch_v2"],
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials_per_repeat", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument(
        "--time_budget_min", type=float, default=90.0,
        help=(
            "Wall-clock safety cutoff per trial in minutes (default: 90). "
            "This is NOT the controlled variable — max_steps is fixed at "
            "PHASE3_FIXED_MAX_STEPS(=200) for every trial. 90min gives enough "
            "headroom for the slowest hyperparameter combo (effective_batch up "
            "to 64) to reach 200 steps based on Phase 2 main's measured "
            "throughput (~13s/step at effective_batch=8)."
        ),
    )
    parser.add_argument(
        "--max_test_samples", type=int, default=None,
        help=(
            "Cap on final per-trial test-set evaluation samples (default: "
            "None = full test set, use for the paper-final run). Cuts "
            "wall-clock cost without touching max_steps/repeats/"
            "trials_per_repeat — same lever as measure_cross_dataset_cf.py's "
            "--max_samples 500."
        ),
    )
    parser.add_argument(
        "--max_parallel", type=int, default=1,
        help=(
            "Number of (strategy, repeat) combinations to run concurrently, "
            "one physical GPU each (default: 1 = fully sequential, same as "
            "before). Each combination runs in its own subprocess pinned to "
            "a GPU via CUDA_VISIBLE_DEVICES; trials within one combination "
            "stay sequential since optuna/autoresearch pick each trial from "
            "the previous trials' results."
        ),
    )
    args = parser.parse_args()

    setup_logging(log_dir=args.output_dir, experiment_name="run_phase3")

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
        max_test_samples=args.max_test_samples,
        max_parallel=args.max_parallel,
    )


if __name__ == "__main__":
    main()
