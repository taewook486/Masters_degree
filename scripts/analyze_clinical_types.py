#!/usr/bin/env python3
"""Phase 3 파인튜닝 성능 향상의 임상 유형별 분해 분석 (논문 4.4.4 검증).

논문 4.4.4는 "본 연구가 달성한 정확도 향상의 상당 부분은 임상적 가치가
상대적으로 낮은 유형에서 발생했을 가능성이 있다"고 서술하되, 유형별 분해를
수행하지 않아 검증을 향후 과제로 남겼다. 이 스크립트가 그 검증을 수행한다.

검증이 가능한 이유:
  Phase 1 제로샷 평가는 PathVQA test 전체 6,719건을 평가했고, Phase 3의 810
  trial은 전부 그 앞 500건을 평가했다(질문 집합 완전 일치를 실행 시 재확인함).
  따라서 동일 표본에 대한 대응 비교가 가능하다.

분류 기준:
  src.evaluate.clinical_significance.classify_clinical_type — Phase 1 4.1.3의
  유형별 표를 만든 것과 동일한 함수를 그대로 재사용한다.

사용:
    PYTHONPATH=. python3 scripts/analyze_clinical_types.py
    PYTHONPATH=. python3 scripts/analyze_clinical_types.py --top_k 10
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from src.evaluate.clinical_significance import (
    CLINICAL_WEIGHTS,
    classify_clinical_type,
)

# 조건별 (results.tsv 경로, trial 디렉터리 루트)
CONDITIONS: dict[str, tuple[str, str]] = {
    "manual": (
        "results/phase3_autoresearch/results.tsv",
        "results/phase3_autoresearch",
    ),
    "random": (
        "results/phase3_autoresearch/results.tsv",
        "results/phase3_autoresearch",
    ),
    "optuna": (
        "results/phase3_autoresearch/results.tsv",
        "results/phase3_autoresearch",
    ),
    "autoresearch": (
        "results/phase3_autoresearch/results.tsv",
        "results/phase3_autoresearch",
    ),
    "autoresearch_v2": (
        "results_pod_backup/phase3_autoresearch_v2/results.tsv",
        "results_pod_backup/phase3_autoresearch_v2",
    ),
}

ZEROSHOT_JSON = "results/phase1_baseline/qwen3-vl-2b_pathvqa_seed42.json"

# 유형 출력 순서 — 임상 중요도 가중치 내림차순
TYPE_ORDER = [
    "diagnosis",
    "location",
    "measurement",
    "description",
    "temporal",
    "yes_no",
    "unknown",
]


def load_trials(tsv_path: Path, strategy: str) -> list[dict]:
    """results.tsv에서 특정 전략의 completed trial만 읽는다.

    agent_reasoning 필드에 개행이 들어 있어 csv 모듈의 인용 처리가 필요하다.
    """
    with tsv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [
        r
        for r in rows
        if r.get("strategy") == strategy and r.get("status") == "completed"
    ]


def find_eval_json(
    root: Path, strategy: str, repeat_id: str, trial_id: str
) -> Path | None:
    """trial 디렉터리에서 평가 결과 json을 찾는다.

    파일명에 학습 seed가 들어가 고정할 수 없으므로 글롭으로 찾는다.
    train_result.json은 학습 통계라 제외한다.
    """
    trial_dir = root / f"{strategy}_repeat{repeat_id}" / f"trial_{int(trial_id):04d}"
    if not trial_dir.is_dir():
        return None
    for p in sorted(trial_dir.glob("*_pathvqa_*.json")):
        return p
    return None


def per_type_stats(samples: list[dict]) -> tuple[dict[str, int], dict[str, float]]:
    """샘플 목록을 임상 유형별로 갈라 개수와 정확도를 낸다."""
    hits: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for s in samples:
        ctype = classify_clinical_type(s["question"], s["gold_answer"])
        total[ctype] += 1
        if s["correct"]:
            hits[ctype] += 1
    acc = {t: hits[t] / total[t] for t in total}
    return dict(total), acc


def contribution_shares(
    counts: dict[str, int],
    deltas: dict[str, float],
    n_total: int,
) -> tuple[dict[str, float], dict[str, float], float]:
    """전체 정확도 개선폭에 대한 유형별 기여도를 분해한다.

    4.4.4의 "향상의 상당 부분이 어느 유형에서 발생했는가"는 유형별 개선폭이
    아니라 표본 수로 가중한 기여도를 묻는 질문이다. 유형 t의 기여도는
    (n_t / N) * delta_t 이고, 이 값들의 합이 전체 개선폭과 같다.

    Returns:
        (절대 기여도, 순개선분 대비 점유율, 전체 개선폭)
    """
    absolute = {t: (counts[t] / n_total) * deltas.get(t, 0.0) for t in counts}
    overall = sum(absolute.values())
    # 점유율은 양의 기여만 분모로 삼는다 (음수 기여가 있으면 합이 100%를
    # 넘는 착시가 생기므로, 상승분 내 점유율로 읽는다).
    positive = sum(v for v in absolute.values() if v > 0)
    share = {t: (v / positive if positive else 0.0) for t, v in absolute.items()}
    return absolute, share, overall


def weighted_clinical_accuracy(acc: dict[str, float], counts: dict[str, int]) -> float:
    """관찰된 유형만으로 WCA를 계산한다 (compute_wca와 동일 정의)."""
    num = sum(acc[t] * CLINICAL_WEIGHTS[t] for t in acc if counts.get(t, 0) > 0)
    den = sum(CLINICAL_WEIGHTS[t] for t in acc if counts.get(t, 0) > 0)
    return num / den if den else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    """순위 상관 (동점은 평균 순위)."""

    def ranks(vs: list[float]) -> list[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 향상의 임상 유형별 분해")
    ap.add_argument("--top_k", type=int, default=10,
                    help="조건별 상위 K개 trial 평균으로 안정성을 확인한다 (기본 10)")
    ap.add_argument(
        "--out_prefix",
        default="results/phase3_autoresearch/clinical_type_breakdown",
    )
    args = ap.parse_args()

    # ---- 1. 제로샷 기준선 (Phase 1) ----
    zs = json.loads(Path(ZEROSHOT_JSON).read_text(encoding="utf-8"))
    zs_all = zs["per_sample"]
    zs_500 = zs_all[:500]

    zs_count_full, zs_acc_full = per_type_stats(zs_all)
    zs_count, zs_acc = per_type_stats(zs_500)

    # ---- 2. Phase 3 조건별 trial 적재 ----
    report: dict = {
        "zeroshot": {
            "source": ZEROSHOT_JSON,
            "full_test": {"n_total": len(zs_all), "per_type_count": zs_count_full,
                          "per_type_accuracy": zs_acc_full},
            "eval_500": {
                "n_total": len(zs_500),
                "per_type_count": zs_count,
                "per_type_accuracy": zs_acc,
                "wca": weighted_clinical_accuracy(zs_acc, zs_count),
                "overall_accuracy": (
                    sum(1 for s in zs_500 if s["correct"]) / len(zs_500)
                ),
            },
        },
        "conditions": {},
        "sample_set_identical": None,
    }

    identical_checks: list[bool] = []
    zs_questions = [s["question"] for s in zs_500]

    for cond, (tsv, root) in CONDITIONS.items():
        tsv_path, root_path = Path(tsv), Path(root)
        if not tsv_path.exists():
            print(f"[skip] {cond}: {tsv} 없음", file=sys.stderr)
            continue

        trials = load_trials(tsv_path, cond)
        loaded: list[tuple[float, str, list[dict]]] = []
        for t in trials:
            jp = find_eval_json(root_path, cond, t["repeat_id"], t["trial_id"])
            if jp is None:
                continue
            data = json.loads(jp.read_text(encoding="utf-8"))
            ps = data.get("per_sample")
            if not ps:
                continue
            identical_checks.append([s["question"] for s in ps] == zs_questions)
            loaded.append((float(t["val_accuracy"]), t["trial_id"], ps))

        if not loaded:
            print(f"[skip] {cond}: 평가 json 없음", file=sys.stderr)
            continue

        loaded.sort(key=lambda x: -x[0])
        best_acc, best_id, best_ps = loaded[0]
        best_count, best_type_acc = per_type_stats(best_ps)

        # 상위 K trial의 유형별 정확도 평균 (단일 trial 노이즈 확인용)
        topk = loaded[: args.top_k]
        topk_accs: dict[str, list[float]] = defaultdict(list)
        for _, _, ps in topk:
            _, a = per_type_stats(ps)
            for t_, v in a.items():
                topk_accs[t_].append(v)
        topk_mean = {t_: statistics.mean(v) for t_, v in topk_accs.items()}
        topk_sd = {t_: (statistics.stdev(v) if len(v) > 1 else 0.0)
                   for t_, v in topk_accs.items()}

        best_delta = {
            t_: best_type_acc.get(t_, 0.0) - zs_acc.get(t_, 0.0) for t_ in best_count
        }
        best_abs, best_share, best_overall = contribution_shares(
            best_count, best_delta, len(best_ps)
        )
        topk_delta = {
            t_: topk_mean.get(t_, 0.0) - zs_acc.get(t_, 0.0) for t_ in topk_mean
        }
        topk_abs, topk_share, topk_overall = contribution_shares(
            best_count, topk_delta, len(best_ps)
        )

        report["conditions"][cond] = {
            "n_trials_loaded": len(loaded),
            "best": {
                "trial_id": best_id,
                "val_accuracy": best_acc,
                "per_type_count": best_count,
                "per_type_accuracy": best_type_acc,
                "per_type_delta_vs_zeroshot": best_delta,
                "per_type_contribution": best_abs,
                "per_type_contribution_share": best_share,
                "overall_delta": best_overall,
                "wca": weighted_clinical_accuracy(best_type_acc, best_count),
            },
            f"top{args.top_k}_mean": {
                "per_type_accuracy": topk_mean,
                "per_type_sd": topk_sd,
                "per_type_delta_vs_zeroshot": topk_delta,
                "per_type_contribution": topk_abs,
                "per_type_contribution_share": topk_share,
                "overall_delta": topk_overall,
                "wca": weighted_clinical_accuracy(topk_mean, best_count),
            },
        }

    report["sample_set_identical"] = all(identical_checks) if identical_checks else None
    report["n_sample_set_checks"] = len(identical_checks)

    # ---- 3. 가중치 대 개선폭 상관 (4.4.4의 핵심 주장) ----
    for cond, block in report["conditions"].items():
        for key in ("best", f"top{args.top_k}_mean"):
            deltas = block[key]["per_type_delta_vs_zeroshot"]
            types = [t for t in TYPE_ORDER
                     if t in deltas and zs_count.get(t, 0) >= 10]
            if len(types) >= 3:
                w = [CLINICAL_WEIGHTS[t] for t in types]
                d = [deltas[t] for t in types]
                block[key]["weight_delta_spearman"] = spearman(w, d)
                block[key]["correlation_types"] = types

    # ---- 4. 출력 ----
    out_json = Path(f"{args.out_prefix}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = render_markdown(report, zs_acc, zs_count, args.top_k)
    out_md = Path(f"{args.out_prefix}.md")
    out_md.write_text(md, encoding="utf-8")

    print(md)
    print(f"\n저장: {out_json}\n저장: {out_md}", file=sys.stderr)


def render_markdown(report: dict, zs_acc: dict, zs_count: dict, top_k: int) -> str:
    L: list[str] = ["# Phase 3 정확도 향상의 임상 유형별 분해", ""]
    L.append(
        f"- 평가 표본 동일성 검사: {report['n_sample_set_checks']}개 trial 전부 "
        f"제로샷 앞 500건과 질문 집합 일치 = **{report['sample_set_identical']}**"
    )
    L.append(
        "- 유형 분류: `src/evaluate/clinical_significance.py` 의 "
        "`classify_clinical_type` (Phase 1 4.1.3과 동일 함수)"
    )
    L.append(
        "- 기여도 = (유형 표본수 / 500) × 유형별 개선폭. "
        "이 값들의 합이 전체 정확도 개선폭과 같다."
    )
    L.append("")

    L.append("## 제로샷 기준선 (Qwen3-VL-2B, PathVQA)")
    L.append("")
    ev = report["zeroshot"]["eval_500"]
    L.append(
        f"평가 500건 기준 전체 정확도 {ev['overall_accuracy']:.4f}, "
        f"WCA **{ev['wca']:.4f}**"
    )
    L.append("")
    L.append(
        "| 유형 | 가중치 | 전체 6,719건 n | 전체 정확도 "
        "| 평가 500건 n | 500건 정확도 |"
    )
    L.append("|---|---:|---:|---:|---:|---:|")
    full = report["zeroshot"]["full_test"]
    for t in TYPE_ORDER:
        if t not in zs_count and t not in full["per_type_count"]:
            continue
        L.append(
            f"| {t} | {CLINICAL_WEIGHTS[t]:.1f} "
            f"| {full['per_type_count'].get(t, 0)} "
            f"| {full['per_type_accuracy'].get(t, 0.0):.4f} "
            f"| {zs_count.get(t, 0)} | {zs_acc.get(t, 0.0):.4f} |"
        )
    L.append("")

    for cond, block in report["conditions"].items():
        best = block["best"]
        L.append(
            f"## {cond} — 최고 trial {best['trial_id']} "
            f"(val_acc {best['val_accuracy']:.4f}, WCA {best['wca']:.4f})"
        )
        L.append("")
        L.append(
            "| 유형 | 가중치 | n | 제로샷 | 파인튜닝 "
            "| 개선폭 | 기여도 | 상승분 점유 |"
        )
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for t in TYPE_ORDER:
            if t not in best["per_type_count"]:
                continue
            L.append(
                f"| {t} | {CLINICAL_WEIGHTS[t]:.1f} "
                f"| {best['per_type_count'][t]} "
                f"| {zs_acc.get(t, 0.0):.4f} "
                f"| {best['per_type_accuracy'][t]:.4f} "
                f"| {best['per_type_delta_vs_zeroshot'][t]:+.4f} "
                f"| {best['per_type_contribution'][t]:+.4f} "
                f"| {best['per_type_contribution_share'][t] * 100:+.1f}% |"
            )
        L.append("")
        L.append(f"- 전체 정확도 개선폭 합계: **{best['overall_delta']:+.4f}**")
        if "weight_delta_spearman" in best:
            types = ", ".join(best["correlation_types"])
            L.append(
                f"- 임상 가중치 대 개선폭 Spearman r = "
                f"**{best['weight_delta_spearman']:+.4f}** (유형 {types}; "
                f"표본 10건 이상인 유형만, n={len(best['correlation_types'])})"
            )

        tk = block[f"top{top_k}_mean"]
        L.append("")
        L.append(f"상위 {top_k} trial 평균 — 단일 trial 노이즈 확인 "
                 f"(WCA {tk['wca']:.4f})")
        L.append("")
        L.append(
            "| 유형 | 평균 정확도 | 표준편차 | 제로샷 대비 "
            "| 기여도 | 상승분 점유 |"
        )
        L.append("|---|---:|---:|---:|---:|---:|")
        for t in TYPE_ORDER:
            if t not in tk["per_type_accuracy"]:
                continue
            L.append(
                f"| {t} | {tk['per_type_accuracy'][t]:.4f} "
                f"| {tk['per_type_sd'][t]:.4f} "
                f"| {tk['per_type_delta_vs_zeroshot'][t]:+.4f} "
                f"| {tk['per_type_contribution'][t]:+.4f} "
                f"| {tk['per_type_contribution_share'][t] * 100:+.1f}% |"
            )
        L.append("")
        L.append(f"- 전체 정확도 개선폭 합계: **{tk['overall_delta']:+.4f}**")
        if "weight_delta_spearman" in tk:
            L.append(
                f"- 임상 가중치 대 개선폭 Spearman r = "
                f"**{tk['weight_delta_spearman']:+.4f}**"
            )
        L.append("")

    return "\n".join(L)


if __name__ == "__main__":
    main()
