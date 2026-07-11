"""Phase 1 오염 강건성 재계산 — 의심 샘플 제거 후 결론 유지 확인 (설계서 §4.2.1).

오염 분석(analyze_contamination.py)이 플래그한 '의심 샘플'을 제거하고 Phase 1
정확도·RQ1을 재계산해, best model과 모델 순위가 유지되는지 확인한다. GPU 불필요
(저장된 per-sample 정오 + 의심 sample_id 사용).

공정성: 의심 샘플은 (모델×데이터셋)별로 다르므로, RQ1 비교가 성립하도록 **데이터셋별
합집합**을 제거해 4모델을 '동일한 축소 샘플셋'으로 재평가한다.

정렬: 오염의 sample_id와 Phase 1 per_sample index는 둘 다 test split 순서 → index로 정렬.

사용법:
    python scripts/robustness_phase1.py \
        --phase1_dir results/phase1_baseline_rescored \
        --contamination_json results/contamination/contamination_analysis.json \
        --seed 42
출력:
    results/phase1_baseline_rescored/phase1_robustness.md / .json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from src.evaluate.statistics import bootstrap_accuracy_ci, run_cochran_q, run_mcnemar

DATASETS = ["pathvqa", "slake", "vqa_rad"]


def _load_suspected_union(contam_json: Path) -> dict[str, set[int]]:
    """데이터셋별 의심 sample_id 합집합 (모든 모델 통합)."""
    with open(contam_json, encoding="utf-8") as fp:
        d = json.load(fp)
    union: dict[str, set[int]] = {ds: set() for ds in DATASETS}
    for _, c in d["conditions"].items():
        ds = c["dataset"]
        union.setdefault(ds, set()).update(c.get("suspected_sample_ids", []))
    return union


def _discover_models(phase1_dir: Path, seed: int) -> list[str]:
    models: set[str] = set()
    for f in phase1_dir.glob(f"*_seed{seed}.json"):
        stem = f.stem[: f.stem.rfind(f"_seed{seed}")]
        for ds in DATASETS:
            if stem.endswith(f"_{ds}"):
                models.add(stem[: -(len(ds) + 1)])
                break
    return sorted(models)


def _load_correct_by_index(phase1_dir: Path, model: str, dataset: str, seed: int):
    f = phase1_dir / f"{model}_{dataset}_seed{seed}.json"
    if not f.exists():
        return None
    with open(f, encoding="utf-8") as fp:
        d = json.load(fp)
    return {
        r["index"]: (1 if r["correct"] else 0, r["question_type"])
        for r in d["per_sample"]
    }


def _acc(correct: list[int]) -> dict:
    n = len(correct)
    if n == 0:
        return {"n": 0, "acc": None, "ci": (None, None)}
    lo, hi = bootstrap_accuracy_ci(correct)
    return {"n": n, "acc": round(sum(correct) / n, 4), "ci": (lo, hi)}


def analyze(phase1_dir: Path, contam_json: Path, seed: int) -> dict:
    models = _discover_models(phase1_dir, seed)
    suspected = _load_suspected_union(contam_json)

    report: dict = {
        "seed": seed,
        "models": models,
        "removed_per_dataset": {ds: len(suspected.get(ds, set())) for ds in DATASETS},
        "per_dataset": {},
        "pooled_ranking": {},
    }

    # 데이터셋별: 원본 vs 축소 정확도 + RQ1
    pooled_full = {m: [] for m in models}
    pooled_clean = {m: [] for m in models}

    for ds in DATASETS:
        remove = suspected.get(ds, set())
        cond_full: dict[str, list[int]] = {}
        cond_clean: dict[str, list[int]] = {}
        entry = {"n_removed": len(remove), "models": {}}
        # 공통 index 정렬 (모델 간 동일 index여야 RQ1 성립)
        ref_indices = None
        ok = True
        for m in models:
            cbi = _load_correct_by_index(phase1_dir, m, ds, seed)
            if cbi is None:
                ok = False
                break
            idx_sorted = sorted(cbi.keys())
            if ref_indices is None:
                ref_indices = idx_sorted
            full = [cbi[i][0] for i in ref_indices]
            clean = [cbi[i][0] for i in ref_indices if i not in remove]
            cond_full[m] = full
            cond_clean[m] = clean
            pooled_full[m].extend(full)
            pooled_clean[m].extend(clean)
            entry["models"][m] = {
                "full": _acc(full),
                "clean": _acc(clean),
            }
        if not ok:
            continue
        # RQ1: 축소셋 Cochran's Q
        cq_full = run_cochran_q(cond_full)
        cq_clean = run_cochran_q(cond_clean)
        entry["cochran_q_full"] = {k: cq_full[k] for k in ("q_stat", "p_value", "significant")}
        entry["cochran_q_clean"] = {k: cq_clean[k] for k in ("q_stat", "p_value", "significant")}
        report["per_dataset"][ds] = entry

    # pooled 순위: 원본 vs 축소
    def _rank(pooled: dict[str, list[int]]) -> list[tuple[str, float]]:
        accs = [(m, round(sum(v) / len(v), 4)) for m, v in pooled.items() if v]
        return sorted(accs, key=lambda x: -x[1])

    rank_full = _rank(pooled_full)
    rank_clean = _rank(pooled_clean)
    report["pooled_ranking"] = {
        "full": rank_full,
        "clean": rank_clean,
        "best_full": rank_full[0][0] if rank_full else None,
        "best_clean": rank_clean[0][0] if rank_clean else None,
        "ranking_preserved": [m for m, _ in rank_full] == [m for m, _ in rank_clean],
        "cochran_q_pooled_full": {k: run_cochran_q(pooled_full)[k] for k in ("q_stat", "p_value", "significant")},
        "cochran_q_pooled_clean": {k: run_cochran_q(pooled_clean)[k] for k in ("q_stat", "p_value", "significant")},
    }
    return report


def _render_markdown(report: dict) -> str:
    L = ["# Phase 1 오염 강건성 재계산 (의심 샘플 제거)", ""]
    L.append(f"- seed: {report['seed']}  ·  모델: {', '.join(report['models'])}")
    L.append("- 방식: 데이터셋별 의심 샘플(합집합) 제거 후 4모델을 동일 축소셋으로 재평가")
    rm = report["removed_per_dataset"]
    L.append(f"- 제거 샘플 수: " + ", ".join(f"{ds}={rm[ds]}" for ds in DATASETS))
    L.append("")

    for ds, e in report["per_dataset"].items():
        L.append(f"## {ds} (제거 {e['n_removed']}개)")
        L.append("")
        L.append("| 모델 | 원본 acc | 축소 acc | 원본 n | 축소 n |")
        L.append("|------|:-------:|:-------:|:-----:|:-----:|")
        for m, md in e["models"].items():
            f, c = md["full"], md["clean"]
            L.append(f"| {m} | {f['acc']} | {c['acc']} | {f['n']} | {c['n']} |")
        cf, cc = e["cochran_q_full"], e["cochran_q_clean"]
        L.append("")
        L.append(f"Cochran's Q: 원본 p={cf['p_value']:.4g}({'유의' if cf['significant'] else '비유의'}) "
                 f"→ 축소 p={cc['p_value']:.4g}({'유의' if cc['significant'] else '비유의'})")
        L.append("")

    pr = report["pooled_ranking"]
    L.append("## Pooled 순위 (전체 데이터셋)")
    L.append("")
    L.append("| 순위 | 원본 | 축소 |")
    L.append("|:---:|------|------|")
    for i in range(max(len(pr["full"]), len(pr["clean"]))):
        a = f"{pr['full'][i][0]} ({pr['full'][i][1]})" if i < len(pr["full"]) else ""
        b = f"{pr['clean'][i][0]} ({pr['clean'][i][1]})" if i < len(pr["clean"]) else ""
        L.append(f"| {i+1} | {a} | {b} |")
    L.append("")
    verdict = "유지됨 ✓" if pr["ranking_preserved"] else "변화 발생 ⚠️"
    L.append(f"**Best model**: 원본 `{pr['best_full']}` → 축소 `{pr['best_clean']}`")
    L.append(f"**모델 순위**: **{verdict}**")
    L.append("")
    L.append("## 결론")
    if pr["ranking_preserved"] and pr["best_full"] == pr["best_clean"]:
        L.append("> 오염 의심 샘플을 제거해도 **best model과 모델 순위가 불변**이다. "
                 "즉 Phase 1 결론은 잠재적 데이터 오염에 강건하다.")
    else:
        L.append("> 오염 의심 샘플 제거 시 순위/best model에 변화가 있다. 원인 검토 필요.")
    L.append("")
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 오염 강건성 재계산")
    parser.add_argument("--phase1_dir", default="results/phase1_baseline_rescored")
    parser.add_argument("--contamination_json", default="results/contamination/contamination_analysis.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    phase1_dir = Path(args.phase1_dir)
    report = analyze(phase1_dir, Path(args.contamination_json), args.seed)

    md = _render_markdown(report)
    md_path = phase1_dir / "phase1_robustness.md"
    json_path = phase1_dir / "phase1_robustness.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + md)
    print(f"\n리포트 저장: {md_path}")
    print(f"JSON 저장:   {json_path}")


if __name__ == "__main__":
    main()
