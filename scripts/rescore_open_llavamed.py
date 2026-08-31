"""저장된 예측에 LLaVA-Med와 동일한 open-ended 채점 기준(recall)을 적용한다.

논문 4.4.6은 LLaVA-Med와 주관식 성능을 직접 비교할 수 없다고 서술한다. 채점
기준이 다르기 때문인데, 예측 원본(per_sample)이 전부 남아 있으므로 모델을 다시
돌리지 않고도 동일 기준 수치를 산출할 수 있다. 이 스크립트가 그 일을 한다.

대상은 Phase 2 main 조건의 Qwen3-VL-2B × 3데이터셋 × 3시드 = 9개 평가 결과이며,
4.4.6의 Table 4.4a가 LLaVA-Med와 비교하는 조건과 동일하다. GPU는 필요 없다.

사용:
    python3 scripts/rescore_open_llavamed.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate.llavamed_recall import llavamed_recall  # noqa: E402

MODEL = "qwen3-vl-2b"
DATASETS = ["pathvqa", "slake", "vqa_rad"]
SEEDS = [42, 123, 456]

# LLaVA-Med 논문 보고값 (Li et al., 2023). open은 recall, closed는 accuracy.
LLAVAMED_REPORTED = {
    "pathvqa": {"open": 37.95, "closed": 91.21},
    "slake": {"open": 83.08, "closed": 85.34},
    "vqa_rad": {"open": 61.52, "closed": 84.19},
}


def score_one(path: Path) -> dict:
    """평가 결과 1건에서 open 표본만 뽑아 recall 평균을 낸다."""
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    samples = [
        s for s in data.get("per_sample", []) if s.get("question_type") != "closed"
    ]

    recalls = [
        llavamed_recall(s.get("predicted_answer", ""), s.get("gold_answer", ""))
        for s in samples
    ]

    return {
        "path": str(path),
        "n_open": len(samples),
        "llavamed_recall": round(statistics.mean(recalls) * 100, 2) if recalls else 0.0,
        # 논문이 현재 보고 중인 주관식 지표(정답 일치 또는 정답 문자열 포함)
        "open_accuracy": round(summary.get("open_accuracy", 0.0) * 100, 2),
        "closed_accuracy": round(summary.get("closed_accuracy", 0.0) * 100, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/phase2_finetune")
    ap.add_argument(
        "--out", default="results/phase2_finetune/llavamed_recall_rescore.json"
    )
    args = ap.parse_args()

    base = Path(args.results_dir)
    per_run = []
    for dataset in DATASETS:
        for seed in SEEDS:
            name = f"{MODEL}_{dataset}_seed{seed}"
            path = base / name / f"{name}.json"
            if not path.exists():
                print(f"[WARN] 없음: {path}")
                continue
            row = score_one(path)
            row.update({"dataset": dataset, "seed": seed})
            per_run.append(row)

    aggregated = {}
    for dataset in DATASETS:
        rows = [r for r in per_run if r["dataset"] == dataset]
        if not rows:
            continue
        recalls = [r["llavamed_recall"] for r in rows]
        opens = [r["open_accuracy"] for r in rows]
        aggregated[dataset] = {
            "n_seeds": len(rows),
            "n_open": rows[0]["n_open"],
            "recall_mean": round(statistics.mean(recalls), 2),
            "recall_sd": (
                round(statistics.stdev(recalls), 2) if len(recalls) > 1 else 0.0
            ),
            "open_accuracy_mean": round(statistics.mean(opens), 2),
            "closed_accuracy_mean": round(
                statistics.mean([r["closed_accuracy"] for r in rows]), 2
            ),
            "llavamed_open_reported": LLAVAMED_REPORTED[dataset]["open"],
            "llavamed_closed_reported": LLAVAMED_REPORTED[dataset]["closed"],
        }

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(
            {"per_run": per_run, "aggregated": aggregated},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=== 동일 기준(LLaVA-Med recall) 주관식 재채점 ===")
    header = (
        f"{'dataset':>8} {'n_open':>7} {'본연구 recall':>14} {'SD':>5} "
        f"{'LLaVA-Med':>10} {'격차':>7} {'현행 open':>10}"
    )
    print(header)
    for dataset, a in aggregated.items():
        gap = round(a["recall_mean"] - a["llavamed_open_reported"], 2)
        print(
            f"{dataset:>8} {a['n_open']:>7} {a['recall_mean']:>14} {a['recall_sd']:>5} "
            f"{a['llavamed_open_reported']:>10} {gap:>+7} {a['open_accuracy_mean']:>10}"
        )
    print(f"\n기록: {out_path}")


if __name__ == "__main__":
    main()
