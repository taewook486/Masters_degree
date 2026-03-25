"""Catastrophic Forgetting measurement for Phase 2.

THESIS_PROPOSAL v0.2:
  - Control group: VQAv2 validation subset (2,000 samples, balanced)
  - Measurement: Base model accuracy vs Fine-tuned model accuracy on VQAv2
  - Metric: Degradation rate = (Base - Fine-tuned) / Base * 100%
  - Scope: All 9 main conditions (3 models x 3 datasets)
  - Phase 3: Only the final best config
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.data.general_vqa import load_vqav2_subset
from src.evaluate.metrics import (
    compute_closed_accuracy,
    compute_open_accuracy,
)

logger = logging.getLogger(__name__)


def evaluate_on_vqav2(
    model,
    processor,
    config,
    data_dir: str = "data",
    max_samples: int | None = None,
    batch_size: int = 4,
) -> dict:
    """Evaluate a model on the VQAv2 subset.

    Uses the same inference pipeline as zero-shot evaluation.

    Args:
        model: Loaded model (base or fine-tuned).
        processor: Model processor/tokenizer.
        config: Model config (OmegaConf).
        data_dir: Base data directory.
        max_samples: Limit samples for debugging.
        batch_size: Inference batch size.

    Returns:
        Dict with closed_accuracy, open_accuracy, overall_accuracy.
    """
    from src.baseline.model_loader import generate_answer

    samples = load_vqav2_subset(data_dir=data_dir, max_samples=max_samples)

    predictions = []
    gold_answers = []
    question_types = []

    start = time.time()

    for sample in samples:
        prompt = (
            "Look at this image and answer the following question.\n"
            f"Question: {sample.question}\n"
            "Answer concisely."
        )

        try:
            pred = generate_answer(
                model=model,
                processor=processor,
                config=config,
                image=sample.image,
                prompt=prompt,
            )
        except Exception as e:
            logger.warning(f"Inference failed: {e}")
            pred = ""

        predictions.append(pred)
        gold_answers.append(sample.answer)
        question_types.append(sample.question_type)

    elapsed = time.time() - start

    # Compute metrics by type
    closed_preds = [p for p, qt in zip(predictions, question_types) if qt == "closed"]
    closed_golds = [g for g, qt in zip(gold_answers, question_types) if qt == "closed"]
    open_preds = [p for p, qt in zip(predictions, question_types) if qt == "open"]
    open_golds = [g for g, qt in zip(gold_answers, question_types) if qt == "open"]

    closed_acc = compute_closed_accuracy(closed_preds, closed_golds)
    open_acc = compute_open_accuracy(open_preds, open_golds)

    total = len(predictions)
    total_correct = 0
    if closed_preds:
        total_correct += round(closed_acc * len(closed_preds))
    if open_preds:
        total_correct += round(open_acc * len(open_preds))
    overall_acc = total_correct / total if total > 0 else 0.0

    return {
        "closed_accuracy": round(closed_acc, 4),
        "open_accuracy": round(open_acc, 4),
        "overall_accuracy": round(overall_acc, 4),
        "closed_count": len(closed_preds),
        "open_count": len(open_preds),
        "total_count": total,
        "eval_time_sec": round(elapsed, 1),
    }


def measure_catastrophic_forgetting(
    base_vqav2_result: dict,
    finetuned_vqav2_result: dict,
) -> dict:
    """Compute catastrophic forgetting metrics.

    Args:
        base_vqav2_result: VQAv2 eval result from base (pre-finetune) model.
        finetuned_vqav2_result: VQAv2 eval result from fine-tuned model.

    Returns:
        Dict with degradation rates for each metric.
    """
    metrics = {}
    for key in ["closed_accuracy", "open_accuracy", "overall_accuracy"]:
        base_val = base_vqav2_result.get(key, 0.0)
        ft_val = finetuned_vqav2_result.get(key, 0.0)

        if base_val > 0:
            degradation = (base_val - ft_val) / base_val * 100
        else:
            degradation = 0.0

        metrics[f"base_{key}"] = base_val
        metrics[f"finetuned_{key}"] = ft_val
        metrics[f"degradation_{key}_pct"] = round(degradation, 2)

    return metrics


def run_cf_measurement(
    model,
    processor,
    config,
    base_vqav2_result: dict | None,
    output_dir: str,
    model_name: str,
    dataset_name: str,
    seed: int,
    data_dir: str = "data",
    max_samples: int | None = None,
) -> dict:
    """Run full CF measurement for one fine-tuned model.

    If base_vqav2_result is None, evaluates the base model first
    (but the caller should cache and reuse base results across seeds).

    Args:
        model: Fine-tuned model (already loaded).
        processor: Model processor.
        config: Model config.
        base_vqav2_result: Pre-computed base model VQAv2 results (or None).
        output_dir: Directory to save CF results.
        model_name: Model name for logging.
        dataset_name: Training dataset name.
        seed: Random seed.
        data_dir: Dataset directory.
        max_samples: Limit VQAv2 samples for debugging.

    Returns:
        CF measurement dict with base, finetuned, and degradation metrics.
    """
    logger.info(f"[CF] Evaluating fine-tuned {model_name} (trained on {dataset_name}) on VQAv2...")
    ft_result = evaluate_on_vqav2(
        model, processor, config,
        data_dir=data_dir, max_samples=max_samples,
    )

    if base_vqav2_result is None:
        logger.warning("[CF] No base VQAv2 result provided, cannot compute degradation")
        return {"finetuned_vqav2": ft_result}

    cf_metrics = measure_catastrophic_forgetting(base_vqav2_result, ft_result)

    result = {
        "model_name": model_name,
        "train_dataset": dataset_name,
        "seed": seed,
        "base_vqav2": base_vqav2_result,
        "finetuned_vqav2": ft_result,
        "catastrophic_forgetting": cf_metrics,
    }

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cf_file = output_path / "cf_result.json"
    with open(cf_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    deg = cf_metrics["degradation_overall_accuracy_pct"]
    logger.info(
        f"[CF] {model_name}/{dataset_name}/seed={seed}: "
        f"base={base_vqav2_result['overall_accuracy']:.4f} -> "
        f"ft={ft_result['overall_accuracy']:.4f} "
        f"(degradation={deg:.1f}%)"
    )

    return result
