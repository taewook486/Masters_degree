"""Phase 1 재채점 — 저장된 예측으로 정오/요약 재계산 (GPU 불필요).

매처 개선(_extract_yes_no v0.6: 문장 속 yes/no 추출 + 회피 문구 비확답 처리)을
기존 Phase 1 결과에 반영한다. 모델을 다시 돌리지 않고, 저장된 predicted_answer를
같은 로직으로 재채점하므로 4개 모델 전체에 '동일한' 매처가 일관 적용된다.

원본(results/phase1_baseline_3seed_debug)은 보존하고, 재채점 결과를 별도 디렉터리에 쓴다.
(주의: 재채점 결과 디렉터리는 이후 results/phase1_baseline으로 이름이 바뀌었다 —
논문 산출물은 1시드/12조건 구조이고, 3시드 원본은 phase1_baseline_3seed_debug로 이동됨)

사용법:
    python scripts/rescore_phase1.py \
        --results_dir results/phase1_baseline_3seed_debug \
        --output_dir results/phase1_baseline --seed 42

이후 재채점 결과로 RQ1 분석:
    python scripts/analyze_phase1.py \
        --results_dir results/phase1_baseline --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluate.metrics import (
    _extract_yes_no,
    compute_overall_accuracy,
    preprocess_answer,
)
from src.evaluate.statistics import bootstrap_accuracy_ci

DATASETS = ["pathvqa", "slake", "vqa_rad"]

# run_all._aggregate_seed_results 와 동일한 요약 컬럼 순서 (하위 호환)
SUMMARY_COLUMNS = [
    "model_name", "dataset", "num_seeds",
    "closed_acc_mean", "closed_acc_std", "closed_acc_ci_low", "closed_acc_ci_high",
    "open_acc_mean", "open_acc_std", "open_acc_ci_low", "open_acc_ci_high",
    "overall_acc_mean", "overall_acc_std", "overall_acc_ci_low", "overall_acc_ci_high",
    "open_bertscore_f1_mean", "open_bertscore_f1_std",
    "open_bertscore_accuracy_mean", "open_bertscore_accuracy_std",
    "avg_time_ms_mean", "avg_time_ms_std", "peak_vram_mb",
]


def _is_correct(pred: str, gold: str, question_type: str) -> bool:
    """evaluate_zero_shot._is_correct 와 동일 로직 (torch import 회피용 로컬 복제)."""
    pred_clean = preprocess_answer(pred)
    gold_clean = preprocess_answer(gold)
    if question_type == "closed":
        return _extract_yes_no(pred_clean) == _extract_yes_no(gold_clean)
    return pred_clean == gold_clean or gold_clean in pred_clean


def _rescore_condition(src_file: Path, dst_file: Path) -> dict:
    """한 조건 JSON을 재채점하여 dst에 쓰고, 요약 행(dict)을 반환."""
    with open(src_file, encoding="utf-8") as fp:
        data = json.load(fp)

    per_sample = data["per_sample"]
    preds = [r["predicted_answer"] for r in per_sample]
    golds = [r["gold_answer"] for r in per_sample]
    qtypes = [r["question_type"] for r in per_sample]

    # per-sample 정오 재계산 (개선 매처 반영)
    correctness = [
        1 if _is_correct(p, g, qt) else 0 for p, g, qt in zip(preds, golds, qtypes)
    ]
    for r, c in zip(per_sample, correctness):
        r["correct"] = bool(c)

    orig = data.get("summary", {})
    meta = data.get("metadata", {})

    # BERTScore는 매처와 무관하므로 원본에 이미 있으면 재계산을 생략한다.
    # 단, bertscore 도입 이전 백업에서 복원된 조건처럼 원본에 아예 없는
    # 경우엔 생략하면 빈 값으로 남으므로 그때만 새로 계산한다.
    needs_bertscore = orig.get("open_bertscore_f1") is None
    metrics = compute_overall_accuracy(
        preds, golds, qtypes, compute_bertscore=needs_bertscore
    )

    closed_flags = [c for c, qt in zip(correctness, qtypes) if qt == "closed"]
    open_flags = [c for c, qt in zip(correctness, qtypes) if qt != "closed"]
    overall_ci = bootstrap_accuracy_ci(correctness)
    closed_ci = bootstrap_accuracy_ci(closed_flags)
    open_ci = bootstrap_accuracy_ci(open_flags)

    bertscore_f1 = metrics.get("open_bertscore_f1", orig.get("open_bertscore_f1"))
    bertscore_acc = metrics.get(
        "open_bertscore_accuracy", orig.get("open_bertscore_accuracy")
    )

    new_summary = {
        **metrics,
        "overall_acc_ci_low": overall_ci[0],
        "overall_acc_ci_high": overall_ci[1],
        "closed_acc_ci_low": closed_ci[0],
        "closed_acc_ci_high": closed_ci[1],
        "open_acc_ci_low": open_ci[0],
        "open_acc_ci_high": open_ci[1],
        "open_bertscore_f1": bertscore_f1,
        "open_bertscore_accuracy": bertscore_acc,
        "avg_time_ms": orig.get("avg_time_ms"),
        "peak_vram_mb": orig.get("peak_vram_mb"),
        "rescored": True,
    }
    data["summary"] = new_summary

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_file, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)

    return {
        "model_name": meta.get("model_name"),
        "dataset": meta.get("dataset"),
        "num_seeds": 1,
        "closed_acc_mean": metrics["closed_accuracy"],
        "closed_acc_std": 0.0,
        "closed_acc_ci_low": closed_ci[0],
        "closed_acc_ci_high": closed_ci[1],
        "open_acc_mean": metrics["open_accuracy"],
        "open_acc_std": 0.0,
        "open_acc_ci_low": open_ci[0],
        "open_acc_ci_high": open_ci[1],
        "overall_acc_mean": metrics["overall_accuracy"],
        "overall_acc_std": 0.0,
        "overall_acc_ci_low": overall_ci[0],
        "overall_acc_ci_high": overall_ci[1],
        "open_bertscore_f1_mean": bertscore_f1,
        "open_bertscore_f1_std": 0.0,
        "open_bertscore_accuracy_mean": bertscore_acc,
        "open_bertscore_accuracy_std": 0.0,
        "avg_time_ms_mean": orig.get("avg_time_ms"),
        "avg_time_ms_std": 0.0,
        "peak_vram_mb": orig.get("peak_vram_mb"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 재채점 (개선 매처, GPU 불필요)"
    )
    parser.add_argument("--results_dir", default="results/phase1_baseline_3seed_debug")
    parser.add_argument("--output_dir", default="results/phase1_baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src_dir = Path(args.results_dir)
    dst_dir = Path(args.output_dir)
    files = sorted(src_dir.glob(f"*_seed{args.seed}.json"))
    if not files:
        raise SystemExit(f"재채점할 결과가 없습니다: {src_dir}/*_seed{args.seed}.json")

    rows = []
    print(f"재채점: {len(files)}개 조건  ({src_dir} → {dst_dir})")
    print(f"{'조건':<28} {'overall(원본→재채점)':<28} {'closed(원본→재채점)'}")
    for src_file in files:
        with open(src_file, encoding="utf-8") as fp:
            before = json.load(fp).get("summary", {})
        row = _rescore_condition(src_file, dst_dir / src_file.name)
        b_ov = before.get("overall_accuracy")
        b_cl = before.get("closed_accuracy")
        name = f"{row['model_name']}/{row['dataset']}"
        print(
            f"{name:<28} {b_ov} → {row['overall_acc_mean']:<20} "
            f"{b_cl} → {row['closed_acc_mean']}"
        )
        rows.append(row)

    # 요약 CSV
    csv_path = dst_dir / "phase1_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n재채점 완료. 요약: {csv_path}")
    print(
        f"다음: python scripts/analyze_phase1.py "
        f"--results_dir {dst_dir} --seed {args.seed}"
    )


if __name__ == "__main__":
    main()
