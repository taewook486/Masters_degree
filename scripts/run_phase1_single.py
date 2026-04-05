"""Phase 1: 단일 모델 제로샷 평가 스크립트 (범용).

모델을 1회만 로드하고 9개 조건(3 데이터셋 x 3 시드)을 순차 실행.
RunPod 또는 로컬에서 사용 가능.

사용법:
    # 특정 모델 평가
    python scripts/run_phase1_single.py --config configs/models/gemma4_e2b.yaml

    # 배치 크기 조정
    python scripts/run_phase1_single.py --config configs/models/gemma4_e2b.yaml --batch_size 8

    # 디버그 모드 (샘플 수 제한)
    python scripts/run_phase1_single.py --config configs/models/gemma4_e2b.yaml --max_samples 10

    # 기존 결과 건너뛰기
    python scripts/run_phase1_single.py --config configs/models/gemma4_e2b.yaml --skip_existing
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import torch

from src.baseline.evaluate_zero_shot import evaluate_with_loaded_model
from src.baseline.model_loader import load_config, load_model, unload_model

logger = logging.getLogger(__name__)

DATASETS = ["pathvqa", "slake", "vqa_rad"]
SEEDS = [42, 123, 456]


def result_exists(output_dir: str, model_name: str, dataset: str, seed: int) -> bool:
    """기존 결과 파일이 존재하는지 확인."""
    result_file = Path(output_dir) / f"{model_name}_{dataset}_seed{seed}.json"
    return result_file.exists()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: 단일 모델 제로샷 평가 (범용)",
    )
    parser.add_argument(
        "--config", required=True,
        help="모델 config YAML 경로 (예: configs/models/gemma4_e2b.yaml)",
    )
    parser.add_argument("--output_dir", default="results/phase1_baseline")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="기존 결과 파일이 있으면 해당 조건 건너뛰기",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_config(args.config)
    model_name = config.model_name

    # 실행할 조건 계산
    conditions = []
    skipped = 0
    for dataset_name in DATASETS:
        for seed in SEEDS:
            if args.skip_existing and result_exists(args.output_dir, model_name, dataset_name, seed):
                skipped += 1
                logger.info(f"  [SKIP] {model_name}/{dataset_name}/seed{seed} (결과 존재)")
            else:
                conditions.append((dataset_name, seed))

    if not conditions:
        logger.info(f"모든 조건이 이미 완료되었습니다. (총 {skipped}개 건너뜀)")
        return

    logger.info("=" * 60)
    logger.info(f"Loading model: {model_name}")
    logger.info(f"  실행 조건: {len(conditions)}개 (건너뜀: {skipped}개)")
    logger.info("=" * 60)

    t0 = time.perf_counter()
    model, processor = load_model(config)
    load_time = time.perf_counter() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    completed = 0
    for dataset_name, seed in conditions:
        completed += 1
        logger.info(
            f"\n[{completed}/{len(conditions)}] {model_name} / {dataset_name} / seed={seed}"
        )

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

    logger.info(f"\nUnloading model: {model_name}")
    unload_model(model, processor)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Phase 1 {model_name} 완료! ({completed}개 조건)")
    logger.info(f"결과: {args.output_dir}/{model_name}_*.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
