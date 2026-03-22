"""Run all Phase 1 zero-shot evaluation conditions (3 models x 3 datasets x N seeds).

Key optimization: each model is loaded ONCE and reused across all dataset+seed
conditions before being unloaded. This eliminates the 30-60s load/unload
overhead that would otherwise occur 9x per model.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.baseline.evaluate_zero_shot import evaluate_with_loaded_model
from src.baseline.model_loader import load_config, load_model, unload_model

logger = logging.getLogger(__name__)

DATASETS = ["pathvqa", "slake", "vqa_rad"]


def _load_existing_result(output_dir: str, model_name: str, dataset_name: str, seed: int) -> dict | None:
    """Load existing result JSON if it exists and was a full GPU run."""
    result_file = Path(output_dir) / f"{model_name}_{dataset_name}_seed{seed}.json"
    if not result_file.exists():
        return None
    with open(result_file, encoding="utf-8") as f:
        data = json.load(f)
    num_samples = data.get("metadata", {}).get("num_samples", 0)
    peak_vram = data.get("summary", {}).get("peak_vram_mb", 0)
    if num_samples > 10 and peak_vram > 0:
        return data["summary"]
    return None


def run_all_conditions(
    config_dir: str,
    output_dir: str,
    seeds: list[int],
    data_dir: str = "data",
    max_samples: int | None = None,
    skip_existing: bool = True,
    batch_size: int = 4,
    use_torch_compile: bool = False,
    single_seed_first: bool = False,
) -> pd.DataFrame:
    """Run all model-dataset-seed combinations.

    Each model is loaded once and shared across all dataset+seed conditions
    before being unloaded. This eliminates repeated 30-60s load/unload overhead.

    Args:
        config_dir: Directory containing model config YAMLs.
        output_dir: Directory for saving results.
        seeds: List of random seeds.
        data_dir: Base directory for datasets.
        max_samples: Limit samples per condition (for debugging).
        skip_existing: Skip conditions where a full GPU result already exists.
        batch_size: Inference batch size (1=per-sample, >1=batched GPU throughput).
        use_torch_compile: Apply torch.compile(mode='reduce-overhead') to model.
        single_seed_first: Run only seeds[0] for a fast representative pass
            (saves ~3x time; re-run without this flag to add remaining seeds).

    Returns:
        DataFrame with summary across all conditions.
    """
    config_dir_path = Path(config_dir)
    config_files = sorted(config_dir_path.glob("*.yaml"))

    if not config_files:
        raise FileNotFoundError(f"No YAML configs found in {config_dir}")

    # 1-seed strategy: run only the first seed for a fast representative evaluation
    active_seeds = [seeds[0]] if single_seed_first else seeds

    logger.info(f"Found {len(config_files)} model configs")
    logger.info(f"Datasets: {DATASETS}")
    logger.info(f"Seeds: {active_seeds}")
    logger.info(f"Batch size: {batch_size}")
    if single_seed_first and len(seeds) > 1:
        logger.info(f"  (single_seed_first: seeds {seeds[1:]} skipped for now)")

    all_results = []
    condition_idx = 0

    for config_path in config_files:
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw.get("enabled", True):
            logger.info(f"Skipping disabled config: {config_path.name}")
            continue

        config = load_config(str(config_path))
        model_name = config.model_name
        n_conditions = len(DATASETS) * len(active_seeds)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Loading model: {model_name}  ({n_conditions} conditions ahead)")

        # Fix 4: Load model ONCE for all dataset+seed combos
        model, processor = load_model(config)

        if use_torch_compile:
            logger.info("Applying torch.compile(mode='reduce-overhead')...")
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("torch.compile applied successfully")
            except Exception as e:
                logger.warning(f"torch.compile failed ({e}), continuing without it")

        for dataset_name in DATASETS:
            seed_results = []

            for seed in active_seeds:
                condition_idx += 1
                logger.info(
                    f"\n[{condition_idx}] {model_name} / {dataset_name} / seed={seed}"
                )

                if skip_existing and max_samples is None:
                    existing = _load_existing_result(output_dir, model_name, dataset_name, seed)
                    if existing is not None:
                        logger.info(
                            f"  -> SKIPPING (result exists, overall={existing['overall_accuracy']:.4f})"
                        )
                        seed_results.append(existing)
                        continue

                try:
                    summary = evaluate_with_loaded_model(
                        model=model,
                        processor=processor,
                        config=config,
                        dataset_name=dataset_name,
                        output_dir=output_dir,
                        seed=seed,
                        data_dir=data_dir,
                        max_samples=max_samples,
                        batch_size=batch_size,
                    )
                    seed_results.append(summary)
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"OOM: {model_name}/{dataset_name}/seed={seed}")
                    torch.cuda.empty_cache()
                    seed_results.append(None)
                except Exception as e:
                    logger.error(f"FAILED: {model_name}/{dataset_name}/seed={seed}: {e}")
                    seed_results.append(None)

            valid_results = [r for r in seed_results if r is not None]
            if valid_results:
                agg = _aggregate_seed_results(model_name, dataset_name, valid_results)
                all_results.append(agg)
                _save_intermediate(all_results, output_dir)

        # Fix 4: Unload model ONCE after all conditions for this model
        logger.info(f"Unloading model: {model_name}")
        unload_model(model, processor)
        del model, processor  # Drop references so GPU memory is freed before next model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(all_results)


def _aggregate_seed_results(
    model_name: str,
    dataset_name: str,
    results: list[dict],
) -> dict:
    """Compute mean +/- std across seed repetitions."""
    import numpy as np

    def _stat(key: str) -> tuple[float, float]:
        values = [r[key] for r in results if key in r]
        if not values:
            return 0.0, 0.0
        return round(float(np.mean(values)), 4), round(float(np.std(values)), 4)

    closed_mean, closed_std = _stat("closed_accuracy")
    open_mean, open_std = _stat("open_accuracy")
    overall_mean, overall_std = _stat("overall_accuracy")
    time_mean, time_std = _stat("avg_time_ms")
    vram_values = [r.get("peak_vram_mb", 0) for r in results]
    peak_vram = max(vram_values) if vram_values else 0

    return {
        "model_name": model_name,
        "dataset": dataset_name,
        "num_seeds": len(results),
        "closed_acc_mean": closed_mean,
        "closed_acc_std": closed_std,
        "open_acc_mean": open_mean,
        "open_acc_std": open_std,
        "overall_acc_mean": overall_mean,
        "overall_acc_std": overall_std,
        "avg_time_ms_mean": time_mean,
        "avg_time_ms_std": time_std,
        "peak_vram_mb": peak_vram,
    }


def _save_intermediate(results: list[dict], output_dir: str) -> None:
    """Save intermediate summary for crash recovery."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    recovery_file = output_path / "phase1_intermediate.json"
    with open(recovery_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def generate_summary_csv(df: pd.DataFrame, output_dir: str) -> Path:
    """Save final summary as CSV."""
    output_path = Path(output_dir) / "phase1_summary.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Summary saved to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all Phase 1 zero-shot evaluations"
    )
    parser.add_argument("--config_dir", default="configs/models")
    parser.add_argument("--output_dir", default="results/phase1_baseline")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--no_skip_existing", action="store_true",
        help="Re-run all conditions even if result files already exist",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Inference batch size (default: 4; use 1 for per-sample)",
    )
    parser.add_argument(
        "--torch_compile", action="store_true",
        help="Apply torch.compile(mode='reduce-overhead') for ~15-30%% additional speedup",
    )
    parser.add_argument(
        "--single_seed_first", action="store_true",
        help="Run only seeds[0] for a fast representative pass (~3x faster)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    df = run_all_conditions(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        skip_existing=not args.no_skip_existing,
        batch_size=args.batch_size,
        use_torch_compile=args.torch_compile,
        single_seed_first=args.single_seed_first,
    )

    csv_path = generate_summary_csv(df, args.output_dir)
    logger.info("\n=== Phase 1 Zero-Shot Baseline Results ===")
    logger.info(f"\n{df.to_string(index=False)}")
    logger.info(f"\nFull results: {csv_path}")


if __name__ == "__main__":
    main()
