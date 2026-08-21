"""Phase 3 RQ3 분석: HPO 전략 비교 run-level 통계 검정 (설계서 §4.5).

`src/evaluate/statistics.py`의 Kruskal-Wallis + Mann-Whitney + Bootstrap CI
로직은 구현돼 있으나 실행 진입점이 없었다. 이 스크립트가 그 진입점이다.

분석 단위: run-level (전략 × 10회 독립 반복 -> 반복별 최고 val_accuracy 10개).
Autoresearch/Optuna는 sequential optimization 특성상 동일 run 내 trial 간
의존성이 있어 trial을 독립 관측치로 취급할 수 없다 (설계서 §4.5, v0.5 변경).
trial-level 데이터는 탐색 궤적 시각화 전용이며 본 비교 검정에는 쓰지 않는다.

데이터 소스: results/phase3_autoresearch/results.tsv (ExperimentTracker)
  각 (strategy, repeat_id) 조합에서 완료 trial 중 최고 val_accuracy = run-level 값.
  manual은 반복당 trial 1개뿐이므로 그 1개 값이 곧 run-level 값이다.

  autoresearch_v2는 별도 실행이라 results.tsv가 따로 있다. --extra_results로
  추가 tsv 경로를 넘기면 여러 파일의 trial을 합쳐 5조건으로 분석한다.
  인자를 주지 않으면 기존과 동일하게 4조건만 분석한다.

수행 내용:
  1. 전략별 run-level 값에 percentile Bootstrap 95% CI
  2. Kruskal-Wallis: 완료 데이터가 있는 전략 전체 비교
  3. Mann-Whitney U: RQ3 핵심 쌍별 비교
     - autoresearch vs optuna (원본 RQ3)
     - autoresearch_v2 vs optuna (v2가 기존 최강 baseline을 넘는가)
     - autoresearch_v2 vs autoresearch (v2 개선이 실제 효과가 있는가)

사용법:
    python scripts/analyze_phase3.py --results_dir results/phase3_autoresearch
    python scripts/analyze_phase3.py --results_dir results/phase3_autoresearch \\
        --extra_results results_pod_backup/phase3_autoresearch_v2/results.tsv
출력:
    results/phase3_autoresearch/phase3_rq3_analysis.md  (사람이 읽는 리포트)
    results/phase3_autoresearch/phase3_rq3_analysis.json (기계 판독)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.autoresearch.tracker import ExperimentTracker
from src.evaluate.statistics import (
    bootstrap_accuracy_ci,
    run_kruskal_wallis,
    run_mann_whitney,
)

STRATEGIES = ["manual", "random", "optuna", "autoresearch", "autoresearch_v2"]

# RQ3 쌍별 비교 (Mann-Whitney U). (기준, 비교대상) 순서는 rank-biserial r의
# 부호 규약과 맞물린다 — r > 0이면 첫 번째가 우세하는 경향.
PAIRWISE = [
    ("autoresearch", "optuna"),
    ("autoresearch_v2", "optuna"),
    ("autoresearch_v2", "autoresearch"),
]


def _run_level_accuracies(results_tsvs: list[Path]) -> dict[str, list[float]]:
    """전략별 run-level(반복별 최고) val_accuracy 목록을 반환한다.

    여러 tsv를 받아 완료 trial을 합쳐서 집계한다. 전략별 실행이 서로 다른
    파일에 나뉘어 있어도 하나의 비교표로 묶기 위함이다.
    """
    completed = []
    for path in results_tsvs:
        tracker = ExperimentTracker(path)
        completed.extend(t for t in tracker.load_all() if t.status == "completed")

    by_strategy: dict[str, list[float]] = {s: [] for s in STRATEGIES}
    for strategy in STRATEGIES:
        st_trials = [t for t in completed if t.strategy == strategy]
        for repeat_id in sorted({t.repeat_id for t in st_trials}):
            r_trials = [t for t in st_trials if t.repeat_id == repeat_id]
            if r_trials:
                by_strategy[strategy].append(max(t.val_accuracy for t in r_trials))
    return by_strategy


def analyze(results_dir: Path, extra_results: list[Path] | None = None) -> dict:
    results_tsv = results_dir / "results.tsv"
    sources = [results_tsv, *(extra_results or [])]
    run_level = _run_level_accuracies(sources)

    present = {s: vals for s, vals in run_level.items() if vals}
    missing = [s for s in STRATEGIES if not run_level.get(s)]
    if len(present) < 2:
        raise SystemExit(
            "Kruskal-Wallis에는 최소 2개 전략의 완료 데이터가 필요합니다. "
            f"현재 데이터 있는 전략: {list(present.keys())} "
            f"({', '.join(str(p) for p in sources)})"
        )

    report: dict = {
        "sources": [str(p) for p in sources],
        "n_repeats": {s: len(v) for s, v in run_level.items()},
        "missing_strategies": missing,
        "bootstrap_ci": {},
        "kruskal_wallis": None,
        "mann_whitney_autoresearch_vs_optuna": None,
        "pairwise": {},
        "run_level_values": run_level,
    }

    # 1) 전략별 percentile Bootstrap 95% CI
    for s, vals in present.items():
        ci_low, ci_high = bootstrap_accuracy_ci(vals)
        report["bootstrap_ci"][s] = {
            "mean": round(sum(vals) / len(vals), 4),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": len(vals),
        }

    # 2) Kruskal-Wallis (완료 데이터가 있는 전략 전체)
    report["kruskal_wallis"] = run_kruskal_wallis(present)

    # 3) Mann-Whitney: RQ3 쌍별 비교
    for base, other in PAIRWISE:
        if base in present and other in present:
            report["pairwise"][f"{base}_vs_{other}"] = run_mann_whitney(
                present[base], present[other]
            )

    # 기존 리포트 소비처와의 호환을 위해 원본 RQ3 쌍은 최상위 키로도 남긴다.
    report["mann_whitney_autoresearch_vs_optuna"] = report["pairwise"].get(
        "autoresearch_vs_optuna"
    )

    return report


def _render_markdown(report: dict) -> str:
    lines = [
        "# Phase 3 RQ3 분석 — HPO 전략 비교 (run-level)",
        "",
        "- 분석 단위: run-level (전략 x 독립 반복 -> 반복별 최고 val_accuracy)",
        "- trial-level 데이터는 탐색 궤적 시각화 전용이며 본 비교 검정에는"
        " 사용하지 않음"
        " (설계서 §4.5 v0.5 변경 — sequential optimization의 trial 간 의존성)",
        "",
    ]
    if report.get("sources"):
        lines.append(f"- 데이터 소스: {', '.join(report['sources'])}")
        lines.append("")
    if report["missing_strategies"]:
        missing = ", ".join(report["missing_strategies"])
        lines.append(f"> **주의**: 완료 데이터가 없는 전략: {missing}")
        lines.append("")

    lines.append("## 전략별 run-level 성능 (Bootstrap 95% CI)")
    lines.append("")
    lines.append("| 전략 | n | mean | 95% CI |")
    lines.append("|------|:-:|:----:|:------:|")
    for s, ci in report["bootstrap_ci"].items():
        lines.append(
            f"| {s} | {ci['n']} | {ci['mean']:.4f} | "
            f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] |"
        )
    lines.append("")

    kw = report["kruskal_wallis"]
    lines.append("## Kruskal-Wallis (전략 간 비교)")
    lines.append("")
    if kw:
        sig = "유의함 (p < 0.05)" if kw["significant"] else "유의하지 않음"
        lines.append(f"- H = {kw['h_stat']:.4f}, p = {kw['p_value']:.4g} -> {sig}")
    lines.append("")

    lines.append("## Mann-Whitney U — 쌍별 비교 (RQ3)")
    lines.append("")
    pairwise = report.get("pairwise") or {}
    if pairwise:
        lines.append("| 비교 (기준 vs 대상) | U | p | rank-biserial r | 판정 |")
        lines.append("|---|:-:|:-:|:-:|:-:|")
        for base, other in PAIRWISE:
            mw = pairwise.get(f"{base}_vs_{other}")
            if not mw:
                continue
            sig = "유의함" if mw["significant"] else "유의하지 않음"
            lines.append(
                f"| {base} vs {other} | {mw['u_stat']:.4f} | "
                f"{mw['p_value']:.4g} | {mw['rank_biserial_r']:.4f} | {sig} |"
            )
        lines.append("")
        lines.append(
            "- r > 0: 기준 전략이 대상보다 확률적으로 우세하는 경향 "
            "(부호 규약: `src/evaluate/statistics.py` run_mann_whitney 참조)"
        )
    else:
        lines.append("> 쌍별 비교에 필요한 완료 데이터 부족으로 생략")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 RQ3 분석 "
            "(Kruskal-Wallis + Mann-Whitney + Bootstrap CI, run-level)"
        )
    )
    parser.add_argument("--results_dir", default="results/phase3_autoresearch")
    parser.add_argument(
        "--extra_results",
        nargs="*",
        default=[],
        metavar="TSV",
        help=(
            "추가로 합쳐 읽을 results.tsv 경로 "
            "(예: 별도 실행된 autoresearch_v2). 생략하면 4조건만 분석한다."
        ),
    )
    parser.add_argument(
        "--output_suffix",
        default="",
        metavar="SUFFIX",
        help=(
            "출력 파일명 접미사. 조건 수가 다른 분석 결과가 서로 덮어쓰지 "
            "않도록 분리할 때 쓴다 (예: _5cond)."
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report = analyze(results_dir, [Path(p) for p in args.extra_results])

    stem = f"phase3_rq3_analysis{args.output_suffix}"
    md_path = results_dir / f"{stem}.md"
    json_path = results_dir / f"{stem}.json"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"리포트 저장: {md_path}")
    print(f"리포트 저장: {json_path}")


if __name__ == "__main__":
    main()
