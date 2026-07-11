"""Phase 1 RQ1 분석: 모델 간 zero-shot 성능 차이 검정 (Cochran's Q + McNemar).

zero-shot은 결정적 평가이므로 시드-분산 ANOVA 대신 '공유 테스트셋 짝지은 검정'을 쓴다
(THESIS v0.6). 4개 모델이 동일 샘플로 평가되므로 반복측정 검정이 적절하고 검정력도 높다.

수행 내용:
  1. 데이터셋별: 4모델 Cochran's Q (H0: 정확도 동일) → 유의 시 McNemar 쌍별(Bonferroni)
  2. pooled: 3개 데이터셋을 합쳐 모델 종합 비교 (Cochran's Q + McNemar)
  3. 각 모델×데이터셋 및 pooled: 정확도 + 부트스트랩 95% CI

사용법:
    python scripts/analyze_phase1.py \
        --results_dir results/phase1_baseline --seed 42
출력:
    results/phase1_baseline/phase1_rq1_analysis.md  (사람이 읽는 리포트)
    results/phase1_baseline/phase1_rq1_analysis.json (기계 판독)
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from src.evaluate.statistics import (
    bootstrap_accuracy_ci,
    run_cochran_q,
    run_mcnemar,
)

DATASETS = ["pathvqa", "slake", "vqa_rad"]


def _load_condition(results_dir: Path, model: str, dataset: str, seed: int):
    """조건별 per-sample 정오(0/1)와 정렬 키를 반환. 없으면 (None, None)."""
    f = results_dir / f"{model}_{dataset}_seed{seed}.json"
    if not f.exists():
        return None, None
    with open(f, encoding="utf-8") as fp:
        data = json.load(fp)
    per_sample = data.get("per_sample", [])
    correct = [1 if r.get("correct") else 0 for r in per_sample]
    # 정렬 검증용 키 (index + question). 모델 간 동일 순서여야 짝지은 검정이 성립.
    keys = [(r.get("index"), r.get("question")) for r in per_sample]
    return correct, keys


def _discover_models(results_dir: Path, seed: int) -> list[str]:
    """결과 파일명에서 모델 목록 추출 ({model}_{dataset}_seed{seed}.json)."""
    models: set[str] = set()
    for f in results_dir.glob(f"*_seed{seed}.json"):
        stem = f.stem  # {model}_{dataset}_seed{seed}
        stem = stem[: stem.rfind(f"_seed{seed}")]
        for ds in DATASETS:
            if stem.endswith(f"_{ds}"):
                models.add(stem[: -(len(ds) + 1)])
                break
    return sorted(models)


def _aligned_correctness(
    results_dir: Path, models: list[str], dataset: str, seed: int
) -> dict[str, list[int]] | None:
    """한 데이터셋에서 모든 모델의 정오 벡터를 정렬 검증 후 dict로 반환."""
    vectors: dict[str, list[int]] = {}
    ref_keys = None
    for m in models:
        correct, keys = _load_condition(results_dir, m, dataset, seed)
        if correct is None:
            print(f"  [경고] 누락: {m}/{dataset} → 이 데이터셋 검정에서 제외")
            return None
        if ref_keys is None:
            ref_keys = keys
        elif keys != ref_keys:
            # 순서/샘플이 모델 간 다르면 짝지은 검정 불가
            if len(keys) != len(ref_keys):
                print(f"  [경고] 샘플 수 불일치: {m}/{dataset} ({len(keys)} vs {len(ref_keys)})")
                return None
            print(f"  [경고] 샘플 순서 불일치 의심: {m}/{dataset} (index 기준 강제 정렬 필요)")
        vectors[m] = correct
    return vectors


def _pairwise_mcnemar(vectors: dict[str, list[int]]) -> list[dict]:
    """모든 모델 쌍에 McNemar + Bonferroni 보정."""
    models = list(vectors.keys())
    pairs = list(itertools.combinations(models, 2))
    n_pairs = len(pairs)
    rows = []
    for a, b in pairs:
        res = run_mcnemar(vectors[a], vectors[b])
        p_adj = min(res["p_value"] * n_pairs, 1.0)
        rows.append({
            "model_a": a,
            "model_b": b,
            "b01_a_only": res["b01"],
            "b10_b_only": res["b10"],
            "n_discordant": res["n_discordant"],
            "p_value": round(res["p_value"], 6),
            "p_bonferroni": round(p_adj, 6),
            "significant": bool(p_adj < 0.05),
        })
    return rows


def _acc_ci_table(vectors: dict[str, list[int]]) -> dict[str, dict]:
    """모델별 정확도 + 부트스트랩 95% CI."""
    out = {}
    for m, v in vectors.items():
        acc = sum(v) / len(v) if v else 0.0
        lo, hi = bootstrap_accuracy_ci(v)
        out[m] = {"accuracy": round(acc, 4), "ci_low": lo, "ci_high": hi, "n": len(v)}
    return out


def analyze(results_dir: Path, seed: int) -> dict:
    models = _discover_models(results_dir, seed)
    if len(models) < 2:
        raise SystemExit(f"모델이 2개 미만입니다({models}). Phase 1 결과를 확인하세요.")
    print(f"발견된 모델: {models}")

    report: dict = {"seed": seed, "models": models, "per_dataset": {}, "pooled": {}}

    # 1) 데이터셋별 검정
    pooled_vectors: dict[str, list[int]] = {m: [] for m in models}
    pooled_ok = True
    for ds in DATASETS:
        vectors = _aligned_correctness(results_dir, models, ds, seed)
        if vectors is None:
            pooled_ok = False
            continue
        cq = run_cochran_q(vectors)
        entry = {
            "accuracy_ci": _acc_ci_table(vectors),
            "cochran_q": {k: cq[k] for k in ("q_stat", "p_value", "df", "k", "significant")},
            "mcnemar_pairs": _pairwise_mcnemar(vectors) if cq["significant"] else [],
        }
        report["per_dataset"][ds] = entry
        for m in models:
            pooled_vectors[m].extend(vectors[m])

    # 2) pooled (전체 데이터셋 합산) — 모든 데이터셋이 정렬 가능할 때만
    if pooled_ok and all(len(pooled_vectors[m]) > 0 for m in models):
        cq = run_cochran_q(pooled_vectors)
        report["pooled"] = {
            "accuracy_ci": _acc_ci_table(pooled_vectors),
            "cochran_q": {k: cq[k] for k in ("q_stat", "p_value", "df", "k", "significant")},
            "mcnemar_pairs": _pairwise_mcnemar(pooled_vectors) if cq["significant"] else [],
        }
    else:
        report["pooled"] = {"skipped": "일부 데이터셋 정렬 실패로 pooled 분석 생략"}

    return report


def _render_markdown(report: dict) -> str:
    lines = ["# Phase 1 RQ1 분석 — 모델 간 zero-shot 성능 차이", ""]
    lines.append(f"- seed: {report['seed']}  ·  모델: {', '.join(report['models'])}")
    lines.append("- 검정: Cochran's Q (공유 테스트셋 이진 정오) + McNemar 쌍별(Bonferroni)")
    lines.append("- 불확실성: 정확도 부트스트랩 95% CI  ·  유의수준 alpha=0.05")
    lines.append("")

    def _block(title: str, entry: dict) -> None:
        lines.append(f"## {title}")
        if entry.get("skipped"):
            lines.append(f"> {entry['skipped']}")
            lines.append("")
            return
        lines.append("")
        lines.append("| 모델 | 정확도 | 95% CI | n |")
        lines.append("|------|:------:|:------:|:--:|")
        for m, s in entry["accuracy_ci"].items():
            lines.append(f"| {m} | {s['accuracy']:.4f} | [{s['ci_low']:.4f}, {s['ci_high']:.4f}] | {s['n']} |")
        cq = entry["cochran_q"]
        verdict = "유의 (모델 간 차이 있음)" if cq["significant"] else "비유의"
        lines.append("")
        lines.append(
            f"**Cochran's Q** = {cq['q_stat']:.3f}, df={cq['df']}, "
            f"p = {cq['p_value']:.4g} → **{verdict}**"
        )
        if entry.get("mcnemar_pairs"):
            lines.append("")
            lines.append("McNemar 쌍별 (Bonferroni 보정):")
            lines.append("")
            lines.append("| 모델 A | 모델 B | A만정답 | B만정답 | p(adj) | 유의 |")
            lines.append("|--------|--------|:-------:|:-------:|:------:|:----:|")
            for r in entry["mcnemar_pairs"]:
                sig = "O" if r["significant"] else "-"
                lines.append(
                    f"| {r['model_a']} | {r['model_b']} | {r['b01_a_only']} | "
                    f"{r['b10_b_only']} | {r['p_bonferroni']:.4g} | {sig} |"
                )
        lines.append("")

    for ds, entry in report["per_dataset"].items():
        _block(f"데이터셋: {ds}", entry)
    _block("Pooled (전체 데이터셋 합산)", report["pooled"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 RQ1 분석 (Cochran's Q + McNemar)")
    parser.add_argument("--results_dir", default="results/phase1_baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report = analyze(results_dir, args.seed)

    md = _render_markdown(report)
    md_path = results_dir / "phase1_rq1_analysis.md"
    json_path = results_dir / "phase1_rq1_analysis.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + md)
    print(f"\n리포트 저장: {md_path}")
    print(f"JSON 저장:   {json_path}")


if __name__ == "__main__":
    main()
