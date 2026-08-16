"""단일 (전략, repeat) 조합 실행 진입점 (서브프로세스 격리용).

Phase 3의 trial들은 Phase 2의 (model, dataset, seed) 조건과 달리 같은
(전략, repeat) 안에서는 순차적이다 — optuna는 베이지안 탐색으로, autoresearch는
LLM이 이전 trial 이력(실패 포함)을 보고 다음 하이퍼파라미터를 정하기 때문에
trial 단위 병렬화는 할 수 없다. 대신 서로 다른 (전략, repeat) 조합끼리는
완전히 독립적이라, 이 단위를 병렬화 단위로 삼는다.

run_phase3.py --max_parallel > 1일 때, 이 스크립트를 (전략, repeat) 조합당
하나씩 서브프로세스로 띄운다(CUDA_VISIBLE_DEVICES로 GPU 1장씩 배정). results.tsv는
공유하며, tracker.py의 파일 락으로 trial_id 발급/append를 프로세스 간 직렬화한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.autoresearch.loop import run_hpo_loop
from src.autoresearch.strategies import get_strategy
from src.autoresearch.tracker import ExperimentTracker
from src.utils.logging_config import setup_logging

DATASET = "pathvqa"  # run_phase3.py와 동일 (Phase 3 고정 데이터셋)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one (strategy, repeat) combination (subprocess-isolated)"
    )
    parser.add_argument("--model_config_path", required=True)
    parser.add_argument("--finetune_config_path", required=True)
    parser.add_argument("--output_dir", required=True, help="Phase 3 shared output dir")
    parser.add_argument("--strategy_name", required=True)
    parser.add_argument("--repeat_id", type=int, required=True)
    parser.add_argument("--max_trials", type=int, required=True)
    parser.add_argument(
        "--seed", type=int, required=True, help="Already repeat-adjusted seed"
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--time_budget_min", type=float, default=90.0)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    setup_logging(
        log_dir=args.output_dir,
        experiment_name=f"run_phase3_{args.strategy_name}_repeat{args.repeat_id}",
    )

    tracker = ExperimentTracker(output_path / "results.tsv")
    # autoresearch_v2만 실제 예산을 받는다. 다른 전략 생성자는 인자를
    # 받지 않고, 원본 autoresearch는 기존 동작을 보존한다.
    extra = (
        {"total_trials": args.max_trials}
        if args.strategy_name == "autoresearch_v2"
        else {}
    )
    strategy = get_strategy(args.strategy_name, **extra)

    run_hpo_loop(
        strategy=strategy,
        model_config_path=args.model_config_path,
        base_finetune_config=args.finetune_config_path,
        dataset_name=DATASET,
        tracker=tracker,
        repeat_id=args.repeat_id,
        output_dir=str(output_path / f"{args.strategy_name}_repeat{args.repeat_id}"),
        max_trials=args.max_trials,
        seed=args.seed,
        data_dir=args.data_dir,
        time_budget_min=args.time_budget_min,
        max_test_samples=args.max_test_samples,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
