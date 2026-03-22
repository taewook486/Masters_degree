"""Zero-shot evaluation of a single VLM on a single medical VQA dataset."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.baseline.model_loader import (
    generate_answer,
    generate_answers_batch,
    load_config,
    load_model,
    unload_model,
)
from src.data.dataset import load_medical_vqa_dataset
from src.evaluate.metrics import _extract_yes_no, compute_overall_accuracy, preprocess_answer
from src.utils.seed import set_seed
from src.utils.vram_monitor import get_vram_usage, reset_peak_stats

logger = logging.getLogger(__name__)


def evaluate_with_loaded_model(
    model: Any,
    processor: Any,
    config: DictConfig,
    dataset_name: str,
    output_dir: str,
    seed: int = 42,
    data_dir: str = "data",
    max_samples: int | None = None,
    batch_size: int = 4,
) -> dict:
    """Run zero-shot evaluation with a pre-loaded model (no load/unload).

    Use this when running multiple conditions with the same model to avoid
    the 30-60s model load/unload overhead per condition.

    Args:
        model: Pre-loaded VLM model.
        processor: Pre-loaded processor.
        config: Model DictConfig from load_config().
        dataset_name: One of "pathvqa", "slake", "vqa_rad".
        output_dir: Directory to save results JSON.
        seed: Random seed for data shuffle reproducibility.
        data_dir: Base directory where datasets are stored.
        max_samples: Limit number of samples (for debugging).
        batch_size: Inference batch size (1 = per-sample, >1 = batched).

    Returns:
        Dictionary with summary metrics.
    """
    set_seed(seed)
    model_name = config.model_name

    logger.info(f"=== Evaluating {model_name} on {dataset_name} (seed={seed}) ===")
    reset_peak_stats()

    samples = load_medical_vqa_dataset(dataset_name, split="test", data_dir=data_dir)
    if max_samples is not None:
        samples = samples[:max_samples]
        logger.info(f"Limited to {max_samples} samples (debug mode)")

    per_sample_results: list[dict] = []
    predictions: list[str] = []
    gold_answers: list[str] = []
    question_types: list[str] = []

    if batch_size > 1:
        _infer_batch(
            model, processor, config, samples, batch_size,
            model_name, dataset_name,
            per_sample_results, predictions, gold_answers, question_types,
        )
    else:
        _infer_single(
            model, processor, config, samples,
            model_name, dataset_name,
            per_sample_results, predictions, gold_answers, question_types,
        )

    # Aggregate metrics
    vram = get_vram_usage()
    metrics = compute_overall_accuracy(predictions, gold_answers, question_types)
    time_values = [r["time_ms"] for r in per_sample_results if r["time_ms"] > 0]
    avg_time_ms = sum(time_values) / len(time_values) if time_values else 0.0

    summary = {
        **metrics,
        "avg_time_ms": round(avg_time_ms, 1),
        "peak_vram_mb": vram["peak_mb"],
    }

    result = {
        "metadata": {
            "model_name": model_name,
            "model_id": config.model_id,
            "dataset": dataset_name,
            "split": "test",
            "seed": seed,
            "num_samples": len(samples),
            "batch_size": batch_size,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "per_sample": per_sample_results,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"{model_name}_{dataset_name}_seed{seed}.json"

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {result_file}")
    logger.info(
        f"Summary: closed={summary['closed_accuracy']:.4f}, "
        f"open={summary['open_accuracy']:.4f}, "
        f"overall={summary['overall_accuracy']:.4f}, "
        f"avg_time={summary['avg_time_ms']:.1f}ms, "
        f"peak_vram={summary['peak_vram_mb']:.0f}MB"
    )

    return summary


def _is_correct(pred: str, gold: str, question_type: str) -> bool:
    """Determine correctness for a single sample."""
    pred_clean = preprocess_answer(pred)
    gold_clean = preprocess_answer(gold)
    if question_type == "closed":
        return _extract_yes_no(pred_clean) == _extract_yes_no(gold_clean)
    return pred_clean == gold_clean or gold_clean in pred_clean


def _infer_batch(
    model: Any,
    processor: Any,
    config: DictConfig,
    samples: list,
    batch_size: int,
    model_name: str,
    dataset_name: str,
    per_sample_results: list,
    predictions: list,
    gold_answers: list,
    question_types: list,
) -> None:
    """Run batched inference, appending results to the output lists."""
    n_batches = (len(samples) + batch_size - 1) // batch_size
    progress = tqdm(range(n_batches), desc=f"{model_name}/{dataset_name}", unit="batch")

    for bi in progress:
        start_idx = bi * batch_size
        end_idx = min(start_idx + batch_size, len(samples))
        batch_samples = samples[start_idx:end_idx]
        batch_imgs = [s.image for s in batch_samples]
        batch_qs = [s.question for s in batch_samples]

        t0 = time.perf_counter()
        try:
            batch_preds = generate_answers_batch(model, processor, batch_imgs, batch_qs, config)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at batch {bi} (size={len(batch_samples)}), falling back to single")
            torch.cuda.empty_cache()
            batch_preds = []
            for img, q in zip(batch_imgs, batch_qs):
                try:
                    pred = generate_answer(model, processor, img, q, config)
                except Exception as e:
                    logger.warning(f"Error: {e}")
                    pred = ""
                batch_preds.append(pred)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_sample_ms = elapsed_ms / len(batch_samples)

        for sample, pred in zip(batch_samples, batch_preds):
            correct = _is_correct(pred, sample.answer, sample.question_type)
            predictions.append(pred)
            gold_answers.append(sample.answer)
            question_types.append(sample.question_type)
            per_sample_results.append({
                "index": len(per_sample_results),
                "question": sample.question,
                "gold_answer": sample.answer,
                "predicted_answer": pred,
                "question_type": sample.question_type,
                "correct": correct,
                "time_ms": round(per_sample_ms, 1),
            })

        running_correct = sum(r["correct"] for r in per_sample_results)
        running_acc = running_correct / len(per_sample_results)
        progress.set_postfix(acc=f"{running_acc:.3f}", n=len(per_sample_results))


def _infer_single(
    model: Any,
    processor: Any,
    config: DictConfig,
    samples: list,
    model_name: str,
    dataset_name: str,
    per_sample_results: list,
    predictions: list,
    gold_answers: list,
    question_types: list,
) -> None:
    """Run per-sample inference, appending results to the output lists."""
    progress = tqdm(samples, desc=f"{model_name}/{dataset_name}", unit="sample")
    for i, sample in enumerate(progress):
        t0 = time.perf_counter()
        try:
            pred = generate_answer(model, processor, sample.image, sample.question, config)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at sample {i}, skipping")
            torch.cuda.empty_cache()
            pred = ""
        except Exception as e:
            logger.warning(f"Error at sample {i}: {e}")
            pred = ""

        elapsed_ms = (time.perf_counter() - t0) * 1000
        correct = _is_correct(pred, sample.answer, sample.question_type)
        predictions.append(pred)
        gold_answers.append(sample.answer)
        question_types.append(sample.question_type)
        per_sample_results.append({
            "index": i,
            "question": sample.question,
            "gold_answer": sample.answer,
            "predicted_answer": pred,
            "question_type": sample.question_type,
            "correct": correct,
            "time_ms": round(elapsed_ms, 1),
        })

        running_correct = sum(r["correct"] for r in per_sample_results)
        progress.set_postfix(acc=f"{running_correct / len(per_sample_results):.3f}")


def evaluate_single_condition(
    model_config_path: str,
    dataset_name: str,
    output_dir: str,
    seed: int = 42,
    data_dir: str = "data",
    max_samples: int | None = None,
    batch_size: int = 4,
) -> dict:
    """Run zero-shot evaluation for one model on one dataset.

    Loads the model, evaluates, then unloads. For running multiple conditions
    with the same model, use evaluate_with_loaded_model() to avoid repeated
    30-60s load/unload overhead.
    """
    config = load_config(model_config_path)
    model, processor = load_model(config)
    try:
        return evaluate_with_loaded_model(
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
    finally:
        unload_model(model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of a VLM on a medical VQA dataset"
    )
    parser.add_argument("--model_config", required=True, help="Path to model config YAML")
    parser.add_argument(
        "--dataset", required=True, choices=["pathvqa", "slake", "vqa_rad"]
    )
    parser.add_argument("--output_dir", default="results/phase1_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Inference batch size (default: 4, use 1 for per-sample)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    evaluate_single_condition(
        model_config_path=args.model_config,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        seed=args.seed,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
