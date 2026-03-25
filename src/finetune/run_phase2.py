"""Run all Phase 2 QLoRA fine-tuning conditions.

Experiment matrix (THESIS_PROPOSAL.md Section 4.4):
  - Main: 3 models x 3 datasets x 3 seeds = 27 conditions
  - Ablation A (data size): 5 ratios x 3 seeds = 15 conditions
  - Ablation B (LoRA rank): 5 ranks x 3 seeds = 15 conditions
  - Ablation C (target modules): 3 configs x 3 seeds = 9 conditions

Ablations use a single "best" model + PathVQA, determined from Phase 1 results.
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

from src.finetune.train_qlora import train_qlora

logger = logging.getLogger(__name__)

DATASETS = ["pathvqa", "slake", "vqa_rad"]
ABLATION_DATASET = "pathvqa"

# Cache for base model VQAv2 results (keyed by model_name)
_BASE_VQAV2_CACHE: dict[str, dict] = {}


def _load_existing_result(output_dir: str) -> dict | None:
    """Load existing train_result.json if it exists and has eval results."""
    result_file = Path(output_dir) / "train_result.json"
    if not result_file.exists():
        return None
    with open(result_file, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("eval_summary") and data["training"].get("peak_vram_mb", 0) > 0:
        return data
    return None


def _get_base_vqav2_result(
    model_config_path: str,
    model_name: str,
    data_dir: str,
    output_dir: str,
) -> dict | None:
    """Get or compute base model VQAv2 result (cached per model)."""
    # Check cache
    if model_name in _BASE_VQAV2_CACHE:
        return _BASE_VQAV2_CACHE[model_name]

    # Check saved file
    cache_file = Path(output_dir) / f"{model_name}_base_vqav2.json"
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            result = json.load(f)
        _BASE_VQAV2_CACHE[model_name] = result
        logger.info(f"[CF] Loaded cached base VQAv2 for {model_name}")
        return result

    # Compute: load base model, evaluate on VQAv2
    try:
        from src.baseline.model_loader import load_model, unload_model
        from src.evaluate.catastrophic_forgetting import evaluate_on_vqav2

        logger.info(f"[CF] Computing base VQAv2 for {model_name}...")
        model, processor, config = load_model(model_config_path)

        result = evaluate_on_vqav2(model, processor, config, data_dir=data_dir)

        unload_model(model, processor)

        # Save cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        _BASE_VQAV2_CACHE[model_name] = result
        logger.info(
            f"[CF] Base VQAv2 for {model_name}: "
            f"overall={result['overall_accuracy']:.4f}"
        )
        return result

    except FileNotFoundError:
        logger.warning(
            "[CF] VQAv2 subset not found. "
            "Run: python -m src.data.general_vqa --download"
        )
        return None
    except Exception as e:
        logger.error(f"[CF] Failed to compute base VQAv2 for {model_name}: {e}")
        return None


def run_main_conditions(
    config_dir: str,
    finetune_config: str,
    output_dir: str,
    seeds: list[int],
    data_dir: str = "data",
    skip_existing: bool = True,
    max_train_samples: int | None = None,
    measure_cf: bool = True,
) -> list[dict]:
    """Run main Phase 2 conditions: all models x all datasets x all seeds.

    Args:
        config_dir: Directory with model config YAMLs.
        finetune_config: Path to base QLoRA config.
        output_dir: Output directory.
        seeds: List of random seeds.
        data_dir: Dataset directory.
        skip_existing: Skip if results already exist.
        max_train_samples: Limit training samples (debugging).
        measure_cf: If True, measure Catastrophic Forgetting on VQAv2.
    """
    config_files = sorted(Path(config_dir).glob("*.yaml"))
    results = []

    for config_path in config_files:
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw.get("enabled", True):
            logger.info(f"Skipping disabled: {config_path.name}")
            continue

        model_name = raw.get("model_name", config_path.stem)

        # Pre-compute base VQAv2 for CF measurement (once per model)
        base_vqav2 = None
        if measure_cf:
            base_vqav2 = _get_base_vqav2_result(
                str(config_path), model_name, data_dir, output_dir,
            )

        for dataset_name in DATASETS:
            for seed in seeds:
                run_dir = Path(output_dir) / f"{model_name}_{dataset_name}_seed{seed}"

                logger.info(f"\n=== {model_name} / {dataset_name} / seed={seed} ===")

                if skip_existing and max_train_samples is None:
                    existing = _load_existing_result(str(run_dir))
                    if existing is not None:
                        logger.info("  -> SKIPPING (result exists)")
                        results.append(existing)
                        continue

                try:
                    result = train_qlora(
                        model_config_path=str(config_path),
                        finetune_config_path=finetune_config,
                        dataset_name=dataset_name,
                        output_dir=str(run_dir),
                        seed=seed,
                        data_dir=data_dir,
                        max_train_samples=max_train_samples,
                        measure_cf=measure_cf,
                        base_vqav2_result=base_vqav2,
                    )
                    results.append(result)
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"OOM: {model_name}/{dataset_name}/seed={seed}")
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"FAILED: {model_name}/{dataset_name}/seed={seed}: {e}")

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return results


def run_ablation_a(
    model_config_path: str,
    finetune_config: str,
    output_dir: str,
    seeds: list[int],
    data_dir: str = "data",
    skip_existing: bool = True,
) -> list[dict]:
    """Ablation A: Training data size impact (5%, 10%, 25%, 50%, 100%)."""
    ratios = [0.05, 0.10, 0.25, 0.50, 1.0]
    results = []

    with open(model_config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    model_name = raw.get("model_name", Path(model_config_path).stem)

    for ratio in ratios:
        for seed in seeds:
            run_dir = (
                Path(output_dir)
                / f"ablation_a_{model_name}_{ABLATION_DATASET}_ratio{ratio}_seed{seed}"
            )
            logger.info(f"\n=== Ablation A: ratio={ratio}, seed={seed} ===")

            if skip_existing:
                existing = _load_existing_result(str(run_dir))
                if existing is not None:
                    logger.info("  -> SKIPPING")
                    results.append(existing)
                    continue

            try:
                result = train_qlora(
                    model_config_path=model_config_path,
                    finetune_config_path=finetune_config,
                    dataset_name=ABLATION_DATASET,
                    output_dir=str(run_dir),
                    seed=seed,
                    data_dir=data_dir,
                    subset_ratio=ratio,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"FAILED: ablation_a ratio={ratio}/seed={seed}: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def run_ablation_b(
    model_config_path: str,
    base_finetune_config: str,
    output_dir: str,
    seeds: list[int],
    data_dir: str = "data",
    skip_existing: bool = True,
) -> list[dict]:
    """Ablation B: LoRA rank impact (4, 8, 16, 32, 64)."""
    ranks = [4, 8, 16, 32, 64]
    results = []

    with open(model_config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    model_name = raw.get("model_name", Path(model_config_path).stem)

    for rank in ranks:
        # Build per-rank config path
        ablation_config = Path(output_dir) / f"_config_rank{rank}.yaml"
        _write_ablation_config(base_finetune_config, str(ablation_config), rank=rank)

        for seed in seeds:
            run_dir = (
                Path(output_dir)
                / f"ablation_b_{model_name}_{ABLATION_DATASET}_rank{rank}_seed{seed}"
            )
            logger.info(f"\n=== Ablation B: rank={rank}, seed={seed} ===")

            if skip_existing:
                existing = _load_existing_result(str(run_dir))
                if existing is not None:
                    logger.info("  -> SKIPPING")
                    results.append(existing)
                    continue

            try:
                result = train_qlora(
                    model_config_path=model_config_path,
                    finetune_config_path=str(ablation_config),
                    dataset_name=ABLATION_DATASET,
                    output_dir=str(run_dir),
                    seed=seed,
                    data_dir=data_dir,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"FAILED: ablation_b rank={rank}/seed={seed}: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def run_ablation_c(
    model_config_path: str,
    output_dir: str,
    seeds: list[int],
    data_dir: str = "data",
    skip_existing: bool = True,
) -> list[dict]:
    """Ablation C: Target modules impact (minimal, medium, full)."""
    configs_dir = Path("configs/finetune/ablation")
    module_configs = {
        "minimal": str(configs_dir / "target_minimal.yaml"),
        "medium": str(configs_dir / "target_medium.yaml"),
        "full": str(configs_dir / "target_full.yaml"),
    }
    results = []

    with open(model_config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    model_name = raw.get("model_name", Path(model_config_path).stem)

    for label, config_path in module_configs.items():
        if not Path(config_path).exists():
            logger.warning(f"Ablation C config not found: {config_path}")
            continue

        for seed in seeds:
            run_dir = (
                Path(output_dir)
                / f"ablation_c_{model_name}_{ABLATION_DATASET}_{label}_seed{seed}"
            )
            logger.info(f"\n=== Ablation C: targets={label}, seed={seed} ===")

            if skip_existing:
                existing = _load_existing_result(str(run_dir))
                if existing is not None:
                    logger.info("  -> SKIPPING")
                    results.append(existing)
                    continue

            try:
                result = train_qlora(
                    model_config_path=model_config_path,
                    finetune_config_path=config_path,
                    dataset_name=ABLATION_DATASET,
                    output_dir=str(run_dir),
                    seed=seed,
                    data_dir=data_dir,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"FAILED: ablation_c {label}/seed={seed}: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def _write_ablation_config(
    base_config_path: str,
    output_path: str,
    rank: int | None = None,
    target_modules: list[str] | None = None,
) -> None:
    """Create a modified finetune config for ablation studies."""
    with open(base_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if rank is not None:
        config["lora"]["rank"] = rank
        config["lora"]["alpha"] = rank * 2  # Keep alpha/rank ratio = 2
    if target_modules is not None:
        config["lora"]["target_modules"] = target_modules

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


def build_summary(results: list[dict], label: str) -> pd.DataFrame:
    """Build summary DataFrame from list of result dicts."""
    rows = []
    for r in results:
        if r is None:
            continue
        meta = r.get("metadata", {})
        train = r.get("training", {})
        lora = r.get("lora_config", {})
        eval_s = r.get("eval_summary", {})
        rows.append({
            "experiment": label,
            "model": meta.get("model_name"),
            "dataset": meta.get("dataset"),
            "seed": meta.get("seed"),
            "subset_ratio": meta.get("subset_ratio"),
            "lora_rank": lora.get("rank"),
            "lora_targets": str(lora.get("target_modules")),
            "train_samples": train.get("train_samples"),
            "train_loss": train.get("train_loss"),
            "train_time_min": train.get("train_time_min"),
            "trainable_pct": train.get("trainable_pct"),
            "peak_vram_mb": train.get("peak_vram_mb"),
            "closed_acc": eval_s.get("closed_accuracy"),
            "open_acc": eval_s.get("open_accuracy"),
            "open_bertscore_f1": eval_s.get("open_bertscore_f1"),
            "overall_acc": eval_s.get("overall_accuracy"),
            # CF metrics (v0.2)
            "cf_base_overall": cf.get("base_overall_accuracy") if (cf := r.get("catastrophic_forgetting")) else None,
            "cf_ft_overall": cf.get("finetuned_overall_accuracy") if (cf := r.get("catastrophic_forgetting")) else None,
            "cf_degradation_pct": cf.get("degradation_overall_accuracy_pct") if (cf := r.get("catastrophic_forgetting")) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Phase 2 experiments")
    parser.add_argument("--config_dir", default="configs/models")
    parser.add_argument("--finetune_config", default="configs/finetune/base_qlora.yaml")
    parser.add_argument("--output_dir", default="results/phase2_finetune")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--no_skip_existing", action="store_true")
    parser.add_argument(
        "--best_model_config", default=None,
        help="Path to best model config YAML (for ablations). "
             "If not provided, ablations are skipped.",
    )
    parser.add_argument(
        "--ablation", nargs="*", choices=["a", "b", "c", "all"],
        default=None,
        help="Which ablation studies to run (default: none, use 'all' for all)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    skip = not args.no_skip_existing
    all_dfs = []

    # Main conditions
    logger.info("=" * 60)
    logger.info("Phase 2: Main QLoRA Fine-Tuning Conditions")
    logger.info("=" * 60)

    main_results = run_main_conditions(
        config_dir=args.config_dir,
        finetune_config=args.finetune_config,
        output_dir=args.output_dir,
        seeds=args.seeds,
        data_dir=args.data_dir,
        skip_existing=skip,
        max_train_samples=args.max_train_samples,
    )
    if main_results:
        all_dfs.append(build_summary(main_results, "main"))

    # Ablation studies
    ablations = set(args.ablation or [])
    if "all" in ablations:
        ablations = {"a", "b", "c"}

    if ablations and args.best_model_config:
        if "a" in ablations:
            logger.info("\n" + "=" * 60)
            logger.info("Ablation A: Training Data Size")
            logger.info("=" * 60)
            ab_a = run_ablation_a(
                args.best_model_config, args.finetune_config,
                args.output_dir, args.seeds, args.data_dir, skip,
            )
            if ab_a:
                all_dfs.append(build_summary(ab_a, "ablation_a"))

        if "b" in ablations:
            logger.info("\n" + "=" * 60)
            logger.info("Ablation B: LoRA Rank")
            logger.info("=" * 60)
            ab_b = run_ablation_b(
                args.best_model_config, args.finetune_config,
                args.output_dir, args.seeds, args.data_dir, skip,
            )
            if ab_b:
                all_dfs.append(build_summary(ab_b, "ablation_b"))

        if "c" in ablations:
            logger.info("\n" + "=" * 60)
            logger.info("Ablation C: Target Modules")
            logger.info("=" * 60)
            ab_c = run_ablation_c(
                args.best_model_config, args.output_dir,
                args.seeds, args.data_dir, skip,
            )
            if ab_c:
                all_dfs.append(build_summary(ab_c, "ablation_c"))

    elif ablations:
        logger.warning("--best_model_config required for ablation studies")

    # Save combined summary
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        summary_path = Path(args.output_dir) / "phase2_summary.csv"
        combined.to_csv(summary_path, index=False)
        logger.info(f"\nPhase 2 summary saved to {summary_path}")
        logger.info(f"\n{combined.to_string(index=False)}")


if __name__ == "__main__":
    main()
