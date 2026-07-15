"""임상적 의미 분석 (WCA + ECE) — Phase 1/2 결과 후처리 진입점 (설계서 §4.4.5).

`src/evaluate/clinical_significance.py`의 WCA/ECE 로직은 구현돼 있으나 실행 진입점이
없었다. 이 스크립트가 그 진입점으로, 저장된 per-sample 결과(JSON)를 읽어 WCA와
질문 유형별 정확도를 산출한다. BERTScore 재실행이 필요 없도록 **저장된 `correct`
플래그를 그대로 사용**한다(재채점 없음).

- WCA (Weighted Clinical Accuracy): PathVQA 7개 질문 유형 분류 + 임상 중요도 가중치.
  PathVQA만 임상 유형 키워드 매핑이 설계돼 있으므로 기본 대상은 pathvqa.
- ECE (Expected Calibration Error): 모델 confidence 필요. 현재 per-sample 레코드에
  confidence가 저장되지 않으므로 ECE는 산출 불가 → 리포트에 'N/A(미저장)'로 명시한다.

주의(설계서 §5.3): WCA 가중치는 외부 검증 없는 임시 척도이며, 절대적 임상 중요도로
해석될 수 없는 참고용 보조 지표다. primary 지표(정확도, BERTScore)를 대체하지 않는다.

사용법:
    python scripts/analyze_clinical.py \
        --results_dir results/phase1_baseline --dataset pathvqa --seed 42
출력:
    results/phase1_baseline/clinical_analysis_pathvqa.md  (사람이 읽는 리포트)
    results/phase1_baseline/clinical_analysis_pathvqa.json (기계 판독)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluate.clinical_significance import (
    CLINICAL_WEIGHTS,
    classify_clinical_type,
)


def _discover_models(results_dir: Path, dataset: str, seed: int) -> list[str]:
    """결과 파일명에서 모델 목록 추출 ({model}_{dataset}_seed{seed}.json)."""
    suffix = f"_{dataset}_seed{seed}.json"
    models = {
        f.name[: -len(suffix)]
        for f in results_dir.glob(f"*{suffix}")
    }
    return sorted(models)


def _load_per_sample(
    results_dir: Path, model: str, dataset: str, seed: int
) -> list[dict] | None:
    """조건별 per-sample 레코드 반환. 없으면 None."""
    f = results_dir / f"{model}_{dataset}_seed{seed}.json"
    if not f.exists():
        return None
    with open(f, encoding="utf-8") as fp:
        data = json.load(fp)
    return data.get("per_sample", [])


def _compute_wca(per_sample: list[dict]) -> dict:
    """저장된 correct 플래그로 질문 유형별 정확도 + WCA 산출.

    clinical_significance.classify_clinical_type / CLINICAL_WEIGHTS를 재사용한다.
    is_correct_fn 재채점 대신 per-sample의 correct(bool)를 그대로 사용한다.
    """
    type_correct: dict[str, int] = {k: 0 for k in CLINICAL_WEIGHTS}
    type_total: dict[str, int] = {k: 0 for k in CLINICAL_WEIGHTS}

    for r in per_sample:
        question = r.get("question", "")
        gold = r.get("gold_answer", "")
        ctype = classify_clinical_type(question, gold)
        type_total[ctype] += 1
        if r.get("correct"):
            type_correct[ctype] += 1

    per_type_accuracy: dict[str, float] = {}
    per_type_count: dict[str, int] = {}
    for ctype in CLINICAL_WEIGHTS:
        count = type_total[ctype]
        per_type_count[ctype] = count
        per_type_accuracy[ctype] = (
            round(type_correct[ctype] / count, 4) if count > 0 else 0.0
        )

    # WCA = Σ(유형별 정확도 × 가중치) / Σ가중치 (관찰된 유형만)
    weighted_sum = 0.0
    weight_sum = 0.0
    for ctype, weight in CLINICAL_WEIGHTS.items():
        if per_type_count[ctype] > 0:
            weighted_sum += per_type_accuracy[ctype] * weight
            weight_sum += weight
    wca = round(weighted_sum / weight_sum, 4) if weight_sum > 0 else 0.0

    overall = sum(1 for r in per_sample if r.get("correct"))
    overall_acc = round(overall / len(per_sample), 4) if per_sample else 0.0

    return {
        "wca": wca,
        "overall_accuracy": overall_acc,
        "per_type_accuracy": per_type_accuracy,
        "per_type_count": per_type_count,
        "n": len(per_sample),
        # confidence 미저장 → ECE 산출 불가
        "ece": None,
    }


def analyze(results_dir: Path, dataset: str, seed: int) -> dict:
    models = _discover_models(results_dir, dataset, seed)
    if not models:
        raise SystemExit(
            f"{dataset} 결과가 없습니다(seed={seed}). "
            f"{results_dir}/*_{dataset}_seed{seed}.json 를 확인하세요."
        )
    print(f"발견된 모델: {models}")

    report: dict = {
        "dataset": dataset,
        "seed": seed,
        "models": models,
        "weights": CLINICAL_WEIGHTS,
        "per_model": {},
    }
    for m in models:
        per_sample = _load_per_sample(results_dir, m, dataset, seed)
        if not per_sample:
            print(f"  [경고] 누락/빈 결과: {m}/{dataset} → 제외")
            continue
        report["per_model"][m] = _compute_wca(per_sample)
    return report


def _render_markdown(report: dict) -> str:
    lines = [
        f"# 임상적 의미 분석 (WCA) — {report['dataset']} (seed {report['seed']})",
        "",
        "- 지표: Weighted Clinical Accuracy (WCA) + 질문 유형별 정확도",
        f"- 가중치: {', '.join(f'{k}={v}' for k, v in report['weights'].items())}",
        "- ECE: N/A — per-sample confidence 미저장 (산출 불가)",
        "",
        "> **주의(§5.3)**: WCA 가중치는 외부 검증 없는 임시 척도다. 절대적 임상 "
        "중요도로 해석 불가하며, primary 지표(정확도, BERTScore)를 보완하는 참고용이다.",
        "",
        "## 모델별 WCA 요약",
        "",
        "| 모델 | Overall Acc | WCA | n |",
        "|------|:-----------:|:---:|:-:|",
    ]
    for m, s in report["per_model"].items():
        lines.append(f"| {m} | {s['overall_accuracy']:.4f} | {s['wca']:.4f} | {s['n']} |")
    lines.append("")

    for m, s in report["per_model"].items():
        lines.append(f"## {m} — 질문 유형별 정확도")
        lines.append("")
        lines.append("| 유형 | 가중치 | 정확도 | 샘플 수 |")
        lines.append("|------|:------:|:------:|:-------:|")
        for ctype, weight in report["weights"].items():
            acc = s["per_type_accuracy"][ctype]
            cnt = s["per_type_count"][ctype]
            lines.append(f"| {ctype} | {weight} | {acc:.4f} | {cnt} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="임상적 의미 분석 (WCA) — 저장된 per-sample 결과 후처리"
    )
    parser.add_argument("--results_dir", default="results/phase1_baseline")
    parser.add_argument("--dataset", default="pathvqa")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report = analyze(results_dir, args.dataset, args.seed)

    md_path = results_dir / f"clinical_analysis_{args.dataset}.md"
    json_path = results_dir / f"clinical_analysis_{args.dataset}.json"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"리포트 저장: {md_path}")
    print(f"리포트 저장: {json_path}")


if __name__ == "__main__":
    main()
