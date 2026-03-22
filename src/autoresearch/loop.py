"""Core HPO experiment loop for Phase 3.

Each trial:
  1. Strategy suggests hyperparameter config
  2. Write temporary finetune YAML
  3. Train for fixed time budget (15 min) or fixed epochs
  4. Evaluate on validation set
  5. Record result in tracker
  6. Repeat

Supports time-boxed training (stop after N minutes regardless of epoch completion).
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import torch
import yaml

from src.autoresearch.strategies import HPOStrategy
from src.autoresearch.tracker import ExperimentTracker, TrialResult
from src.utils.seed import set_seed
from src.utils.vram_monitor import reset_peak_stats

logger = logging.getLogger(__name__)


def _write_trial_config(
    base_config_path: str,
    hp: dict,
    output_path: str,
) -> str:
    """Write a temporary finetune config YAML for this trial."""
    with open(base_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Map target preset names
    target_map = {
        "minimal": ["q_proj", "v_proj"],
        "medium": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "full": ["all_linear"],
    }

    config["lora"]["rank"] = hp["lora_rank"]
    config["lora"]["alpha"] = hp["lora_alpha"]
    config["lora"]["dropout"] = 0.05
    config["lora"]["target_modules"] = target_map.get(hp["lora_targets"], ["q_proj", "v_proj"])

    config["training"]["learning_rate"] = hp["learning_rate"]
    config["training"]["per_device_train_batch_size"] = hp["batch_size"]
    config["training"]["gradient_accumulation_steps"] = hp["grad_accum_steps"]
    config["training"]["warmup_ratio"] = hp["warmup_ratio"]
    config["training"]["weight_decay"] = hp["weight_decay"]
    config["training"]["num_train_epochs"] = hp["epochs"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_path


def run_single_trial(
    model_config_path: str,
    base_finetune_config: str,
    dataset_name: str,
    hp: dict,
    trial_id: int,
    strategy_name: str,
    repeat_id: int,
    output_dir: str,
    seed: int = 42,
    data_dir: str = "data",
    time_budget_min: float = 15.0,
) -> TrialResult:
    """Run a single HPO trial with the given hyperparameters.

    Args:
        model_config_path: Path to model config YAML.
        base_finetune_config: Base finetune config to modify.
        dataset_name: Dataset name (e.g., "pathvqa").
        hp: Hyperparameter dict from strategy.suggest().
        trial_id: Unique trial identifier.
        strategy_name: Strategy that generated this config.
        repeat_id: Independent repeat index.
        output_dir: Base output directory for this trial.
        seed: Random seed.
        data_dir: Dataset directory.
        time_budget_min: Max training time in minutes.

    Returns:
        Completed TrialResult.
    """
    from src.finetune.train_qlora import train_qlora

    set_seed(seed)
    reset_peak_stats()

    trial_dir = Path(output_dir) / f"trial_{trial_id:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Write trial-specific config
    trial_config_path = str(trial_dir / "finetune_config.yaml")
    _write_trial_config(base_finetune_config, hp, trial_config_path)

    trial = TrialResult(
        trial_id=trial_id,
        strategy=strategy_name,
        repeat_id=repeat_id,
        status="running",
        **hp,
    )

    logger.info(
        f"Trial {trial_id} [{strategy_name}]: "
        f"rank={hp['lora_rank']}, alpha={hp['lora_alpha']}, "
        f"lr={hp['learning_rate']:.1e}, targets={hp['lora_targets']}, "
        f"epochs={hp['epochs']}"
    )

    train_start = time.time()

    try:
        result = train_qlora(
            model_config_path=model_config_path,
            finetune_config_path=trial_config_path,
            dataset_name=dataset_name,
            output_dir=str(trial_dir),
            seed=seed,
            data_dir=data_dir,
            eval_after_training=True,
        )

        elapsed_min = (time.time() - train_start) / 60

        eval_summary = result.get("eval_summary", {})
        training = result.get("training", {})

        trial.val_accuracy = eval_summary.get("overall_accuracy", 0.0)
        trial.val_closed_acc = eval_summary.get("closed_accuracy", 0.0)
        trial.val_open_acc = eval_summary.get("open_accuracy", 0.0)
        trial.train_loss = training.get("train_loss", 0.0) or 0.0
        trial.train_time_min = round(elapsed_min, 1)
        trial.peak_vram_mb = training.get("peak_vram_mb", 0.0)
        trial.status = "completed"

        logger.info(
            f"Trial {trial_id} COMPLETED: "
            f"val_acc={trial.val_accuracy:.4f}, "
            f"loss={trial.train_loss:.4f}, "
            f"time={trial.train_time_min:.1f}min"
        )

    except torch.cuda.OutOfMemoryError:
        trial.status = "failed"
        trial.notes = "OOM"
        trial.train_time_min = round((time.time() - train_start) / 60, 1)
        logger.error(f"Trial {trial_id} FAILED: OOM")
        torch.cuda.empty_cache()

    except Exception as e:
        trial.status = "failed"
        trial.notes = str(e)[:200]
        trial.train_time_min = round((time.time() - train_start) / 60, 1)
        logger.error(f"Trial {trial_id} FAILED: {e}")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return trial


def run_hpo_loop(
    strategy: HPOStrategy,
    model_config_path: str,
    base_finetune_config: str,
    dataset_name: str,
    tracker: ExperimentTracker,
    repeat_id: int,
    output_dir: str,
    max_trials: int = 40,
    seed: int = 42,
    data_dir: str = "data",
    time_budget_min: float = 15.0,
) -> list[TrialResult]:
    """Run the full HPO loop for one strategy + one repeat.

    Args:
        strategy: HPO strategy instance.
        model_config_path: Path to model config YAML.
        base_finetune_config: Base finetune config YAML.
        dataset_name: Dataset name (default: "pathvqa").
        tracker: ExperimentTracker for recording results.
        repeat_id: Independent repeat index (0-4).
        output_dir: Output directory.
        max_trials: Maximum number of trials.
        seed: Base random seed (offset by trial_id).
        data_dir: Dataset directory.
        time_budget_min: Max training time per trial in minutes.

    Returns:
        List of TrialResult for this run.
    """
    strategy_name = strategy.name
    logger.info(
        f"\n{'='*60}\n"
        f"HPO Loop: strategy={strategy_name}, repeat={repeat_id}, "
        f"max_trials={max_trials}\n"
        f"{'='*60}"
    )

    # For manual strategy, only 1 trial
    if strategy_name == "manual":
        max_trials = 1

    results: list[TrialResult] = []

    for i in range(max_trials):
        # Get history for this strategy+repeat
        history = tracker.load_by_strategy(strategy_name, repeat_id)
        completed = [t for t in history if t.status == "completed"]

        # Suggest next config
        hp = strategy.suggest(completed)

        trial_id = tracker.next_trial_id()
        trial_seed = seed + i  # Vary seed per trial for diversity

        trial = run_single_trial(
            model_config_path=model_config_path,
            base_finetune_config=base_finetune_config,
            dataset_name=dataset_name,
            hp=hp,
            trial_id=trial_id,
            strategy_name=strategy_name,
            repeat_id=repeat_id,
            output_dir=output_dir,
            seed=trial_seed,
            data_dir=data_dir,
            time_budget_min=time_budget_min,
        )

        tracker.append(trial)
        results.append(trial)

        if trial.status == "completed":
            best = tracker.best_trial(strategy_name, repeat_id)
            if best:
                logger.info(
                    f"[{strategy_name}/repeat{repeat_id}] "
                    f"Trial {i+1}/{max_trials} done. "
                    f"Current best: {best.val_accuracy:.4f} (trial {best.trial_id})"
                )

    return results
