"""Cross-dataset Catastrophic Forgetting 측정 (THESIS §4.4 Table 4.2b-B).

의료 도메인 내 cross-dataset 일반화: 한 데이터셋으로 파인튜닝한 모델을
훈련에 쓰지 않은 나머지 2개 의료 데이터셋으로 평가해, zero-shot(Phase 1)
대비 정확도 변화율을 측정한다. Phase 2가 이미 저장해 둔 LoRA 어댑터
(results/phase2_finetune/<조건>/adapter/)를 재사용하므로 재학습이 필요 없다.

실험 범위: 4모델 x 3훈련데이터셋 x 3시드 x 2평가데이터셋 = 72회 평가.

사용법:
    python scripts/measure_cross_dataset_cf.py \\
        --config_dir configs/models \\
        --phase2_dir results/phase2_finetune \\
        --phase1_summary results/phase1_baseline/phase1_summary.csv \\
        --seeds 42 123 456

출력:
    results/phase2_finetune/<조건>/cross_cf_<train>_to_<eval>.json (72개)
    results/phase2_finetune/cross_dataset_cf_summary.csv (집계)
    results/phase2_finetune/cross_dataset_cf_summary.md (사람이 읽는 리포트)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from src.baseline.model_loader import load_config, load_model, unload_model
from src.evaluate.catastrophic_forgetting import run_cross_dataset_cf_measurement
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

DATASETS = ["pathvqa", "slake", "vqa_rad"]


def _load_phase1_baselines(summary_csv: str) -> dict[tuple[str, str], float]:
    """phase1_summary.csv에서 (model_name, dataset) -> overall_acc_mean 매핑."""
    baselines: dict[tuple[str, str], float] = {}
    with open(summary_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["model_name"], row["dataset"])
            baselines[key] = float(row["overall_acc_mean"])
    return baselines


def _load_existing_cross_cf(condition_dir: Path, train_dataset: str, eval_dataset: str) -> dict | None:
    """이미 저장된 cross_cf 결과가 있으면 반환 (재개 시 재평가 방지)."""
    f = condition_dir / f"cross_cf_{train_dataset}_to_{eval_dataset}.json"
    if not f.exists():
        return None
    try:
        with open(f, encoding="utf-8") as fp:
            return json.load(fp)
    except (json.JSONDecodeError, OSError):
        return None


def _discover_model_configs(config_dir: str) -> list[Path]:
    configs = []
    for path in sorted(Path(config_dir).glob("*.yaml")):
        if path.stem == "_template":
            continue
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw.get("enabled", True):
            continue
        configs.append(path)
    return configs


def run_all_cross_dataset_cf(
    config_dir: str,
    phase2_dir: str,
    phase1_summary: str,
    seeds: list[int],
    data_dir: str = "data",
    max_samples: int | None = None,
) -> list[dict]:
    baselines = _load_phase1_baselines(phase1_summary)
    all_results: list[dict] = []

    for config_path in _discover_model_configs(config_dir):
        config = load_config(str(config_path))
        model_name = config.model_name

        base_model, processor = load_model(config)
        logger.info(f"Loaded base model: {model_name}")

        for train_dataset in DATASETS:
            for seed in seeds:
                condition_dir = Path(phase2_dir) / f"{model_name}_{train_dataset}_seed{seed}"
                adapter_path = condition_dir / "adapter"
                if not adapter_path.exists():
                    logger.warning(f"  [skip] adapter not found: {adapter_path}")
                    continue

                eval_datasets = [d for d in DATASETS if d != train_dataset]

                # 이미 다 끝난 조건이면 어댑터 로드조차 건너뛴다 (재개 시 GPU 낭비 방지).
                pending = [d for d in eval_datasets if _load_existing_cross_cf(condition_dir, train_dataset, d) is None]
                for d in eval_datasets:
                    if d not in pending:
                        existing = _load_existing_cross_cf(condition_dir, train_dataset, d)
                        logger.info(f"  [skip] {model_name}/{train_dataset}->{d}/seed={seed} (이미 존재)")
                        all_results.append(existing)
                if not pending:
                    continue

                # PEFT wraps the base model without mutating its weights (no merge),
                # so the same base_model can be re-wrapped for every condition below.
                peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
                peft_model.eval()

                for eval_dataset in pending:
                    base_acc = baselines.get((model_name, eval_dataset))
                    if base_acc is None:
                        logger.warning(
                            f"  [skip] no Phase 1 baseline for {model_name}/{eval_dataset}"
                        )
                        continue
                    result = run_cross_dataset_cf_measurement(
                        model=peft_model,
                        processor=processor,
                        config=config,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        base_accuracy=base_acc,
                        output_dir=str(condition_dir),
                        model_name=model_name,
                        seed=seed,
                        data_dir=data_dir,
                        max_samples=max_samples,
                    )
                    all_results.append(result)

                del peft_model
                torch.cuda.empty_cache()

        unload_model(base_model, processor)

    return all_results


def _render_summary(results: list[dict], output_dir: str) -> None:
    csv_path = Path(output_dir) / "cross_dataset_cf_summary.csv"
    md_path = Path(output_dir) / "cross_dataset_cf_summary.md"

    fieldnames = [
        "model_name", "train_dataset", "eval_dataset", "seed",
        "base_accuracy", "finetuned_accuracy", "change_pct",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            meta, summ = r["metadata"], r["summary"]
            writer.writerow({
                "model_name": meta["model_name"],
                "train_dataset": meta["train_dataset"],
                "eval_dataset": meta["eval_dataset"],
                "seed": meta["seed"],
                "base_accuracy": summ["base_accuracy"],
                "finetuned_accuracy": summ["finetuned_accuracy"],
                "change_pct": summ["change_pct"],
            })

    lines = [
        "# Cross-dataset Catastrophic Forgetting 측정 (Table 4.2b-B)",
        "",
        "| 모델 | 훈련 | 평가 | seed | Base Acc. | Fine-tuned Acc. | 변화율(%) |",
        "|------|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        meta, summ = r["metadata"], r["summary"]
        lines.append(
            f"| {meta['model_name']} | {meta['train_dataset']} | {meta['eval_dataset']} | "
            f"{meta['seed']} | {summ['base_accuracy']:.4f} | {summ['finetuned_accuracy']:.4f} | "
            f"{summ['change_pct']:+.2f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"요약 CSV: {csv_path}")
    print(f"요약 MD:  {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset CF 측정 (Table 4.2b-B)")
    parser.add_argument("--config_dir", default="configs/models")
    parser.add_argument("--phase2_dir", default="results/phase2_finetune")
    parser.add_argument("--phase1_summary", default="results/phase1_baseline/phase1_summary.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    setup_logging(log_dir=args.phase2_dir, experiment_name="cross_dataset_cf")

    results = run_all_cross_dataset_cf(
        config_dir=args.config_dir,
        phase2_dir=args.phase2_dir,
        phase1_summary=args.phase1_summary,
        seeds=args.seeds,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
    )

    logger.info(f"Completed {len(results)} cross-dataset CF measurements")
    _render_summary(results, args.phase2_dir)


if __name__ == "__main__":
    main()
