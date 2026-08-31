"""Phase 2 학습 시간 재집계 — train_time_min(버그) 대신 train_runtime_sec 기준.

배경: `train_qlora.py`가 기록한 `train_time_min`은 감싸는 스크립트의
wall-clock이라, (모델, 데이터셋) 전처리 캐시를 처음 만드는 조건(대개 seed=42)에서
1회성 캐시 생성 비용이 합산돼 과대 측정된다. Trainer 내부 지표인
`train_runtime_sec`는 오염되지 않았다.

이 스크립트는 각 조건의 train_result.json을 직접 읽어 두 값을 나란히 집계하고,
논문 §4.2.2에 넣을 Ablation A 요약표를 산출한다. GPU 재실행은 필요 없다.

사용:
    python3 scripts/recompute_phase2_train_time.py
    python3 scripts/recompute_phase2_train_time.py \
        --out results/phase2_finetune/train_time_corrected.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

RESULTS_DIR = Path("results/phase2_finetune")


def load_runs(results_dir: Path) -> list[dict]:
    """train_result.json을 전수 읽어 조건별 레코드로 정규화한다."""
    runs = []
    for path in sorted(results_dir.glob("*/train_result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        train = data.get("training", {})

        runtime_sec = train.get("train_runtime_sec")
        if runtime_sec is None:
            # 누락은 추정하지 않고 그대로 드러낸다.
            print(f"[WARN] train_runtime_sec 누락: {path}")

        runs.append(
            {
                "condition": path.parent.name,
                "model": meta.get("model_name"),
                "dataset": meta.get("dataset"),
                "seed": meta.get("seed"),
                "subset_ratio": meta.get("subset_ratio"),
                "train_samples": train.get("train_samples"),
                "reported_min": train.get("train_time_min"),
                "runtime_sec": runtime_sec,
                "corrected_min": (
                    round(runtime_sec / 60, 1) if runtime_sec else None
                ),
            }
        )
    return runs


def ablation_a_table(runs: list[dict]) -> list[dict]:
    """Ablation A(데이터 비율) 조건만 뽑아 비율별 3시드 평균을 낸다."""
    rows = []
    ablation = [r for r in runs if r["condition"].startswith("ablation_a_")]
    ratios = sorted({r["subset_ratio"] for r in ablation})

    for ratio in ratios:
        group = [r for r in ablation if r["subset_ratio"] == ratio]
        corrected = [
            r["corrected_min"] for r in group if r["corrected_min"] is not None
        ]
        reported = [
            r["reported_min"] for r in group if r["reported_min"] is not None
        ]
        rows.append(
            {
                "subset_ratio": ratio,
                "train_samples": group[0]["train_samples"],
                "n_seeds": len(group),
                "corrected_mean_min": round(statistics.mean(corrected), 1),
                "corrected_sd_min": (
                    round(statistics.stdev(corrected), 2)
                    if len(corrected) > 1
                    else 0.0
                ),
                "corrected_min_min": min(corrected),
                "corrected_max_min": max(corrected),
                "reported_mean_min": round(statistics.mean(reported), 1),
            }
        )
    return rows


def contaminated_runs(runs: list[dict], tolerance_min: float = 2.0) -> list[dict]:
    """보고값이 실측보다 tolerance 이상 큰 조건 = 캐시 생성 비용이 섞인 조건."""
    out = []
    for r in runs:
        if r["reported_min"] is None or r["corrected_min"] is None:
            continue
        if r["reported_min"] - r["corrected_min"] > tolerance_min:
            inflation = round(r["reported_min"] - r["corrected_min"], 1)
            out.append({**r, "inflation_min": inflation})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out", default="results/phase2_finetune/train_time_corrected.csv")
    args = ap.parse_args()

    runs = load_runs(Path(args.results_dir))
    print(f"조건 {len(runs)}개 로드")

    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        writer.writeheader()
        writer.writerows(runs)
    print(f"전체 조건 CSV 기록: {out_path}")

    print("\n=== Ablation A: 데이터 비율별 학습 시간 (3시드) ===")
    header = (
        f"{'ratio':>6} {'samples':>8} {'n':>2} {'보정평균':>9} "
        f"{'SD':>6} {'범위':>15} {'버그평균':>9}"
    )
    print(header)
    for row in ablation_a_table(runs):
        rng = f"{row['corrected_min_min']}-{row['corrected_max_min']}"
        print(
            f"{row['subset_ratio']:>6} {row['train_samples']:>8} {row['n_seeds']:>2} "
            f"{row['corrected_mean_min']:>9} {row['corrected_sd_min']:>6} "
            f"{rng:>15} {row['reported_mean_min']:>9}"
        )

    contaminated = contaminated_runs(runs)
    print(f"\n=== 오염 조건: {len(contaminated)}/{len(runs)} ===")
    for r in sorted(contaminated, key=lambda x: -x["inflation_min"]):
        print(
            f"  {r['condition']:<55} 보고 {r['reported_min']:>6}분 / "
            f"실측 {r['corrected_min']:>5}분 "
            f"(+{r['inflation_min']}분)"
        )

    seeds = sorted({r["seed"] for r in contaminated})
    print(f"\n오염 조건의 seed 분포: {seeds}")


if __name__ == "__main__":
    main()
