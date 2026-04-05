"""Phase 1 Gemma 4 E2B 전용 실행 스크립트.

모델을 1회만 로드하고 9개 조건(3 데이터셋 × 3 시드)을 순차 실행.
RunPod 또는 로컬에서 사용 가능.
"""

from __future__ import annotations

import argparse
import gc
import logging
import time

import torch

from src.baseline.evaluate_zero_shot import evaluate_with_loaded_model
from src.baseline.model_loader import load_config, load_model, unload_model

logger = logging.getLogger(__name__)

DATASETS = ["pathvqa", "slake", "vqa_rad"]
SEEDS = [42, 123, 456]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Gemma 4 E2B evaluation")
    parser.add_argument("--output_dir", default="results/phase1_baseline")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = "configs/models/gemma4_e2b.yaml"
    config = load_config(config_path)

    logger.info("=" * 60)
    logger.info(f"Loading model: {config.model_name} (1회 로드, 9개 조건 실행)")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    model, processor = load_model(config)
    load_time = time.perf_counter() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    total_conditions = len(DATASETS) * len(SEEDS)
    completed = 0

    for dataset_name in DATASETS:
        for seed in SEEDS:
            completed += 1
            logger.info(f"\n[{completed}/{total_conditions}] {config.model_name} / {dataset_name} / seed={seed}")

            try:
                summary = evaluate_with_loaded_model(
                    model=model,
                    processor=processor,
                    config=config,
                    dataset_name=dataset_name,
                    output_dir=args.output_dir,
                    seed=seed,
                    data_dir=args.data_dir,
                    max_samples=args.max_samples,
                    batch_size=args.batch_size,
                )
                logger.info(
                    f"  -> closed={summary['closed_accuracy']:.4f}, "
                    f"open={summary['open_accuracy']:.4f}, "
                    f"overall={summary['overall_accuracy']:.4f}"
                )
            except torch.cuda.OutOfMemoryError:
                logger.error(f"  -> OOM! batch_size={args.batch_size}를 줄여보세요.")
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"  -> FAILED: {e}")

    logger.info(f"\nUnloading model: {config.model_name}")
    unload_model(model, processor)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Phase 1 Gemma 4 E2B 완료! ({total_conditions}개 조건)")
    logger.info(f"결과: {args.output_dir}/gemma4-e2b_*.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
