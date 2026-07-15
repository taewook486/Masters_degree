"""Phase 2 RQ2 분석: QLoRA 파인튜닝 효과 통계 검정 (설계서 §4.4).

`src/evaluate/robust_statistics.py`의 BCa Bootstrap + Mixed-Effects + Wilcoxon
로직은 구현돼 있으나 실행 진입점이 없었다. 이 스크립트가 그 진입점으로,
zero-shot(base) 대비 파인튜닝(finetuned) 성능 차이를 3중 검증한다.

데이터 소스:
  - base(zero-shot): results/phase1_baseline/{model}_{dataset}_seed42.json → summary.overall_accuracy
  - finetuned:       results/phase2_finetune/{model}_{dataset}_seed{seed}/train_result.json → eval_summary.overall_accuracy

zero-shot은 결정적(1시드)이므로, 파인튜닝 각 시드(42/123/456)의 base 짝은 동일한
seed42 zero-shot 정확도를 재사용한다. 파인튜닝 효과 = finetuned − base.

수행 내용:
  1. 모델별: (dataset × seed) 쌍 최대 9개로 paired 검정 (t-test + BCa Bootstrap CI + Wilcoxon)
  2. 전체: Mixed-Effects Model (accuracy ~ condition + dataset, group=seed)

사용법:
    python scripts/analyze_phase2.py \
        --phase1_dir results/phase1_baseline \
        --phase2_dir results/phase2_finetune \
        --base_seed 42
출력:
    results/phase2_finetune/phase2_rq2_analysis.md  (사람이 읽는 리포트)
    results/phase2_finetune/phase2_rq2_analysis.json (기계 판독)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluate.robust_statistics import (
    analyze_paired_robust,
    mixed_effects_analysis,
)


def _load_base_accuracy(phase1_dir: Path, model: str, dataset: str, seed: int) -> float | None:
    """zero-shot base 정확도(summary.overall_accuracy) 반환. 없으면 None."""
    f = phase1_dir / f"{model}_{dataset}_seed{seed}.json"
    if not f.exists():
        return None
    with open(f, encoding="utf-8") as fp:
        data = json.load(fp)
    return data.get("summary", {}).get("overall_accuracy")


def _discover_conditions(phase2_dir: Path) -> list[dict]:
    """Phase 2 조건별 train_result.json을 파싱해 (model, dataset, seed, acc) 목록 반환.

    ablation_* 디렉터리는 RQ2 메인 비교 대상이 아니므로 제외한다.
    """
    conditions: list[dict] = []
    for result_file in sorted(phase2_dir.glob("*_seed*/train_result.json")):
        run_dir = result_file.parent.name
        if run_dir.startswith("ablation_"):
            continue
        with open(result_file, encoding="utf-8") as fp:
            data = json.load(fp)
        meta = data.get("metadata", {})
        eval_summary = data.get("eval_summary")
        if not eval_summary:
            print(f"  [경고] eval_summary 누락: {run_dir} → 제외")
            continue
        conditions.append({
            "model": meta.get("model_name"),
            "dataset": meta.get("dataset"),
            "seed": meta.get("seed"),
            "finetuned_acc": eval_summary.get("overall_accuracy"),
        })
    return conditions


def analyze(phase1_dir: Path, phase2_dir: Path, base_seed: int) -> dict:
    conditions = _discover_conditions(phase2_dir)
    if not conditions:
        raise SystemExit(
            f"Phase 2 결과가 없습니다. {phase2_dir}/*_seed*/train_result.json 를 확인하세요."
        )

    # base 정확도 짝지기 (동일 model/dataset의 seed42 zero-shot 재사용)
    paired: list[dict] = []
    missing_base: list[str] = []
    for c in conditions:
        base = _load_base_accuracy(phase1_dir, c["model"], c["dataset"], base_seed)
        if base is None or c["finetuned_acc"] is None:
            missing_base.append(f"{c['model']}/{c['dataset']}/seed{c['seed']}")
            continue
        paired.append({**c, "base_acc": base, "delta": round(c["finetuned_acc"] - base, 4)})

    models = sorted({p["model"] for p in paired})
    report: dict = {
        "base_seed": base_seed,
        "n_conditions": len(paired),
        "models": models,
        "missing_base": missing_base,
        "per_model": {},
        "mixed_effects": {},
        "conditions": paired,
    }

    # 1) 모델별 paired robust 검정
    for m in models:
        rows = [p for p in paired if p["model"] == m]
        base_vals = [p["base_acc"] for p in rows]
        ft_vals = [p["finetuned_acc"] for p in rows]
        res = analyze_paired_robust(base_vals, ft_vals)
        report["per_model"][m] = {
            "n": res.n,
            "mean_base": round(sum(base_vals) / len(base_vals), 4) if base_vals else 0.0,
            "mean_finetuned": round(sum(ft_vals) / len(ft_vals), 4) if ft_vals else 0.0,
            "t_statistic": res.t_statistic,
            "t_pvalue": res.t_pvalue,
            "cohens_d": res.cohens_d,
            "cohens_d_ci": [res.cohens_d_ci_lower, res.cohens_d_ci_upper],
            "wilcoxon_pvalue": res.wilcoxon_pvalue,
            "wilcoxon_r": res.wilcoxon_effect_size_r,
        }

    # 2) 전체 Mixed-Effects (base/finetuned 두 조건 행 구성)
    accuracies: list[float] = []
    cond_labels: list[str] = []
    seeds: list[int] = []
    datasets: list[str] = []
    for p in paired:
        accuracies.extend([p["base_acc"], p["finetuned_acc"]])
        cond_labels.extend(["base", "finetuned"])
        seeds.extend([p["seed"], p["seed"]])
        datasets.extend([p["dataset"], p["dataset"]])
    report["mixed_effects"] = mixed_effects_analysis(
        accuracies, cond_labels, seeds, datasets
    )

    return report


def _render_markdown(report: dict) -> str:
    lines = [
        "# Phase 2 RQ2 분석 — QLoRA 파인튜닝 효과",
        "",
        f"- base(zero-shot) seed: {report['base_seed']}  ·  조건 수: {report['n_conditions']}",
        f"- 모델: {', '.join(report['models'])}",
        "- 검정: paired t-test + BCa Bootstrap 95% CI(Cohen's d) + Wilcoxon signed-rank",
        "- 파인튜닝 효과 = finetuned − base (overall_accuracy)",
        "",
    ]
    if report["missing_base"]:
        lines.append(f"> **주의**: base 또는 eval 누락으로 제외된 조건 {len(report['missing_base'])}개: "
                     f"{', '.join(report['missing_base'])}")
        lines.append("")

    lines.append("## 모델별 파인튜닝 효과")
    lines.append("")
    lines.append("| 모델 | n | base | finetuned | Cohen's d | d 95% CI | t p | Wilcoxon p |")
    lines.append("|------|:-:|:----:|:---------:|:---------:|:--------:|:---:|:----------:|")
    for m, s in report["per_model"].items():
        ci = s["cohens_d_ci"]
        lines.append(
            f"| {m} | {s['n']} | {s['mean_base']:.4f} | {s['mean_finetuned']:.4f} | "
            f"{s['cohens_d']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{s['t_pvalue']:.4g} | {s['wilcoxon_pvalue']:.4g} |"
        )
    lines.append("")

    me = report["mixed_effects"]
    lines.append("## Mixed-Effects Model (accuracy ~ condition + dataset, group=seed)")
    lines.append("")
    if me.get("skipped"):
        lines.append("> statsmodels/pandas 미설치 또는 표본 부족으로 생략 "
                     "(`uv pip install statsmodels pandas`).")
    else:
        lines.append(f"- condition[finetuned] 고정효과 계수: **{me['fixed_effect_coef']}** "
                     f"(p = {me['p_value']:.4g})")
        lines.append(f"- ICC(seed): {me['icc_seed']}  ·  잔차분산: {me['residual_var']}  ·  n = {me['n']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 RQ2 분석 (BCa Bootstrap + Mixed-Effects + Wilcoxon)"
    )
    parser.add_argument("--phase1_dir", default="results/phase1_baseline")
    parser.add_argument("--phase2_dir", default="results/phase2_finetune")
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    phase1_dir = Path(args.phase1_dir)
    phase2_dir = Path(args.phase2_dir)
    report = analyze(phase1_dir, phase2_dir, args.base_seed)

    md_path = phase2_dir / "phase2_rq2_analysis.md"
    json_path = phase2_dir / "phase2_rq2_analysis.json"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"리포트 저장: {md_path}")
    print(f"리포트 저장: {json_path}")


if __name__ == "__main__":
    main()
