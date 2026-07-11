"""Phase 1.5 오염 측정 통합 분석 — Min-K% 결과 해석.

measure_contamination.py가 산출한 조건별(모델×데이터셋) Min-K% 점수를 통합해:
  1. 조건별 분포 요약 (mean/median/std/p90/p95)
  2. 데이터셋 내 상대 이상치 플래그 — Min-K%가 상위 tail(가장 높은 = 모델이 정답을
     과하게 확신 = 사전훈련 노출 의심)인 샘플을 식별
  3. 의심 sample_id 목록 저장 (Phase 1 재계산 sub-analysis용; contamination의 sample_id는
     load_medical_vqa_dataset test 순서라 Phase 1 per_sample index와 정렬됨)

[방법론 한계] 외부 비-멤버 calibration set이 없으므로, 절대 임계값이 아니라
'데이터셋 내 상대 이상치'로 의심 샘플을 식별한다. 논문에 이 한계를 명시할 것.
참조: Shi et al., "Detecting Pretraining Data from Large Language Models", NAACL 2024.

사용법:
    python scripts/analyze_contamination.py \
        --results_dir results/contamination --k_percent 20 --top_pct 5
출력:
    results/contamination/contamination_analysis.md
    results/contamination/contamination_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _percentile(sorted_vals: list[float], p: float) -> float:
    """선형 보간 백분위수 (p: 0~100)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def _load_conditions(results_dir: Path, k: int) -> dict:
    conds = {}
    for f in sorted(results_dir.glob(f"*_minK{k}.json")):
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        s = d["summary"]
        scores = [
            (r["sample_id"], r["minK_score"])
            for r in d.get("per_sample", [])
            if r.get("num_tokens", 0) > 0
        ]
        conds[(s["model_name"], s["dataset"])] = {"summary": s, "scores": scores}
    return conds


def analyze(results_dir: Path, k: int, top_pct: float) -> dict:
    conds = _load_conditions(results_dir, k)
    if not conds:
        raise SystemExit(f"오염 결과가 없습니다: {results_dir}/*_minK{k}.json")

    report: dict = {"k_percent": k, "outlier_top_pct": top_pct, "conditions": {}}
    for (model, ds), c in sorted(conds.items()):
        vals = sorted(v for _, v in c["scores"])
        n = len(vals)
        if n == 0:
            continue
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if n > 1 else 0.0
        # 이상치 임계: 상위 top_pct% (Min-K% 가장 높은 값)
        thr = _percentile(vals, 100.0 - top_pct)
        suspected = sorted(
            [(sid, sc) for sid, sc in c["scores"] if sc >= thr],
            key=lambda x: -x[1],
        )
        report["conditions"][f"{model}/{ds}"] = {
            "model": model,
            "dataset": ds,
            "n": n,
            "mean": round(mean, 4),
            "median": round(statistics.median(vals), 4),
            "std": round(std, 4),
            "p90": round(_percentile(vals, 90), 4),
            "p95": round(_percentile(vals, 95), 4),
            "outlier_threshold": round(thr, 4),
            "n_suspected": len(suspected),
            "suspected_ratio": round(len(suspected) / n, 4),
            "suspected_sample_ids": [sid for sid, _ in suspected],
        }
    return report


def _render_markdown(report: dict) -> str:
    lines = ["# Phase 1.5 데이터 오염 분석 (Min-K% Probability)", ""]
    lines.append(f"- Min-K% 파라미터: K={report['k_percent']}%")
    lines.append(
        f"- 이상치 기준: 데이터셋 내 Min-K% 상위 {report['outlier_top_pct']}% "
        f"(모델이 정답을 과확신 → 사전훈련 노출 의심)"
    )
    lines.append("- 참조: Shi et al., NAACL 2024")
    lines.append("")
    lines.append("> **[한계]** 외부 비-멤버 calibration set이 없어 절대 임계값이 아닌 "
                 "'데이터셋 내 상대 이상치'로 의심 샘플을 식별한다. 의심 = 확정 오염이 "
                 "아니며, 논문에 이 한계를 명시한다.")
    lines.append("")
    lines.append("## 조건별 Min-K% 분포 및 의심 샘플")
    lines.append("")
    lines.append("| 모델/데이터셋 | n | mean | median | std | p95 | 의심수 | 의심비율 |")
    lines.append("|---------------|--:|-----:|-------:|----:|----:|------:|--------:|")
    for name, c in report["conditions"].items():
        lines.append(
            f"| {name} | {c['n']} | {c['mean']} | {c['median']} | {c['std']} | "
            f"{c['p95']} | {c['n_suspected']} | {c['suspected_ratio']:.1%} |"
        )
    lines.append("")
    lines.append("## 해석 가이드")
    lines.append("")
    lines.append("- **mean_minK가 데이터셋별로 높을수록**(0에 가까울수록) 해당 모델이 그 "
                 "데이터셋 정답에 높은 확률을 부여 → 노출 가능성 상대적으로 큼.")
    lines.append("- **의심 샘플(상위 tail)**: 각 조건에서 Min-K%가 유독 높은 샘플. "
                 "`suspected_sample_ids`로 저장됨. Phase 1 재계산 sub-analysis에서 이 "
                 "샘플을 제거하고 정확도를 재산출해 결론이 유지되는지 확인한다.")
    lines.append("- sample_id는 test split 순서라 Phase 1 per_sample index와 정렬된다.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.5 오염 측정 통합 분석")
    parser.add_argument("--results_dir", default="results/contamination")
    parser.add_argument("--k_percent", type=int, default=20)
    parser.add_argument("--top_pct", type=float, default=5.0,
                        help="이상치로 플래그할 상위 백분율 (기본 5%%)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report = analyze(results_dir, args.k_percent, args.top_pct)

    md = _render_markdown(report)
    md_path = results_dir / "contamination_analysis.md"
    json_path = results_dir / "contamination_analysis.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + md)
    print(f"\n리포트 저장: {md_path}")
    print(f"JSON 저장:   {json_path}")


if __name__ == "__main__":
    main()
