"""Phase 3 anytime performance 곡선 생성 (논문 §4.3.3).

전략별로 "trial이 진행됨에 따라 그때까지의 최고 val_accuracy가 어떻게
올라가는가"를 그린다. 탐색 효율(같은 예산에서 얼마나 빨리 좋은 설정을
찾는가)을 비교하는 그림이다.

분석 단위 주의 (설계서 §4.5 / 논문 §3.7):
  이 스크립트가 쓰는 trial-level 데이터는 **탐색 궤적 묘사 전용**이며
  전략 간 우열의 통계적 근거로 삼지 않는다. 검정은 run-level에서만
  수행하며 그쪽은 `scripts/analyze_phase3.py`가 담당한다.

데이터 읽기 주의:
  results.tsv의 `agent_reasoning` 컬럼에는 줄바꿈이 포함된 인용 문자열이
  들어 있다. `pandas.read_csv(..., on_bad_lines="skip")`로 읽으면 유효한
  trial이 조용히 누락될 수 있으므로, 여기서는 `analyze_phase3.py`와 동일하게
  csv 모듈 기반의 `ExperimentTracker`로 읽는다.

사용법:
    python scripts/plot_phase3_anytime.py --results_dir results/phase3_autoresearch
출력:
    <results_dir>/phase3_anytime.png / .pdf        (그림)
    <results_dir>/phase3_anytime_curve.csv         (그림의 원본 수치)
    <results_dir>/phase3_anytime_summary.md        (최고 성능 도달 trial 요약표)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# pod/CI 등 디스플레이 없는 환경에서 실행되므로, pyplot을 끌어오는
# visualize 모듈보다 먼저 non-interactive 백엔드를 지정해야 한다.
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.autoresearch.tracker import ExperimentTracker  # noqa: E402
from src.evaluate.visualize import plot_anytime_performance  # noqa: E402

STRATEGIES = ["manual", "random", "optuna", "autoresearch"]


def _runs_by_strategy(results_tsv: Path) -> dict[str, list[list[float]]]:
    """전략별로 '반복별 누적 최고 val_accuracy 수열' 목록을 만든다.

    반환값 구조: {전략: [[repeat0의 누적최고...], [repeat1의 ...], ...]}
    각 반복 안에서는 trial_id 오름차순을 실행 순서로 본다
    (tracker가 예약 시점에 증가하는 id를 발급하므로 시간순과 일치).
    """
    tracker = ExperimentTracker(results_tsv)
    completed = [t for t in tracker.load_all() if t.status == "completed"]

    runs: dict[str, list[list[float]]] = {}
    for strategy in STRATEGIES:
        st_trials = [t for t in completed if t.strategy == strategy]
        if not st_trials:
            continue

        strategy_runs: list[list[float]] = []
        for repeat_id in sorted({t.repeat_id for t in st_trials}):
            r_trials = sorted(
                (t for t in st_trials if t.repeat_id == repeat_id),
                key=lambda t: t.trial_id,
            )
            cumulative: list[float] = []
            best = float("-inf")
            for t in r_trials:
                best = max(best, t.val_accuracy)
                cumulative.append(best)
            if cumulative:
                strategy_runs.append(cumulative)

        if strategy_runs:
            runs[strategy] = strategy_runs
    return runs


def _build_curve(runs: dict[str, list[list[float]]]) -> pd.DataFrame:
    """반복들을 trial index별 중앙값 + IQR로 집계한다.

    반복마다 완료 trial 수가 다를 수 있으므로(실패/미완료), 각 index에서
    실제로 데이터가 있는 반복만 모아 집계하고 그 개수를 n으로 함께 남긴다.
    n이 전체 반복 수보다 작은 구간은 그림에서 별도 표시된다.
    """
    rows = []
    for strategy, strategy_runs in runs.items():
        max_len = max(len(r) for r in strategy_runs)
        for i in range(max_len):
            vals = [r[i] for r in strategy_runs if len(r) > i]
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            rows.append(
                {
                    "strategy": strategy,
                    "trial_index": i + 1,
                    "median": round(float(med), 6),
                    "q1": round(float(q1), 6),
                    "q3": round(float(q3), 6),
                    "n": len(vals),
                }
            )
    return pd.DataFrame(rows)


def _build_summary(runs: dict[str, list[list[float]]]) -> pd.DataFrame:
    """전략별 '최종 최고 성능에 처음 도달한 trial 번호' 요약.

    논문 §4.3.3을 그림 없이 표로만 쓸 경우에 쓰는 수치다.
    trial 예산을 다 쓰기 전에 이미 최고점에 도달했다면 탐색이 조기 수렴한
    것이고, 마지막 trial에서야 도달했다면 예산이 부족했다는 신호가 된다.
    """
    rows = []
    for strategy, strategy_runs in runs.items():
        time_to_best = [r.index(max(r)) + 1 for r in strategy_runs]
        final_best = [max(r) for r in strategy_runs]
        rows.append(
            {
                "strategy": strategy,
                "n_runs": len(strategy_runs),
                "trials_per_run(median)": int(
                    np.median([len(r) for r in strategy_runs])
                ),
                "final_best(median)": round(float(np.median(final_best)), 4),
                "time_to_best(median)": float(np.median(time_to_best)),
                "time_to_best(IQR)": (
                    f"[{np.percentile(time_to_best, 25):.1f}, "
                    f"{np.percentile(time_to_best, 75):.1f}]"
                ),
            }
        )
    order = {s: i for i, s in enumerate(STRATEGIES)}
    return pd.DataFrame(rows).sort_values(
        "strategy", key=lambda col: col.map(order)
    )


def _render_summary_md(summary: pd.DataFrame, curve: pd.DataFrame) -> str:
    lines = [
        "# Phase 3 anytime performance 요약 (trial-level, 시각화 전용)",
        "",
        "> 이 표와 그림은 탐색 궤적 묘사 전용이며 전략 간 우열의 통계적 근거가 아니다.",
        "> 검정은 run-level에서만 수행한다 (`scripts/analyze_phase3.py`, 논문 §3.7).",
        "",
        "## 전략별 최고 성능 도달 시점",
        "",
        "| 전략 | 반복 수 | 반복당 trial(중앙값) | 최종 최고(중앙값) "
        "| 최고 도달 trial(중앙값) | 도달 trial IQR |",
        "|------|:------:|:------------------:|:----------------:"
        "|:---------------------:|:-------------:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['n_runs']} | {r['trials_per_run(median)']} | "
            f"{r['final_best(median)']:.4f} | {r['time_to_best(median)']:.1f} | "
            f"{r['time_to_best(IQR)']} |"
        )

    incomplete = curve[
        curve["n"] < curve.groupby("strategy")["n"].transform("max")
    ]
    if not incomplete.empty:
        lines += [
            "",
            "> **주의**: 아직 전 반복이 완주하지 않은 구간이 있다 "
            f"(trial index {int(incomplete['trial_index'].min())}부터). "
            "실행 완료 후 다시 생성할 것.",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 anytime performance 곡선 생성 (논문 §4.3.3)"
    )
    parser.add_argument("--results_dir", default="results/phase3_autoresearch")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = _runs_by_strategy(results_dir / "results.tsv")
    if not runs:
        raise SystemExit(
            f"완료된 trial이 없습니다: {results_dir / 'results.tsv'}"
        )

    curve = _build_curve(runs)
    summary = _build_summary(runs)

    png_path = plot_anytime_performance(curve, str(results_dir))
    curve_path = results_dir / "phase3_anytime_curve.csv"
    summary_path = results_dir / "phase3_anytime_summary.md"
    curve.to_csv(curve_path, index=False)
    summary_path.write_text(_render_summary_md(summary, curve), encoding="utf-8")

    for strategy, strategy_runs in runs.items():
        counts = [len(r) for r in strategy_runs]
        print(
            f"{strategy}: {len(strategy_runs)}개 반복, "
            f"trial 수 {min(counts)}~{max(counts)}"
        )
    print(f"\n그림 저장: {png_path}")
    print(f"곡선 원본 수치 저장: {curve_path}")
    print(f"요약표 저장: {summary_path}")


if __name__ == "__main__":
    main()
