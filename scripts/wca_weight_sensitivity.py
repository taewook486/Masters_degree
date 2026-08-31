"""WCA 가중치 민감도 분석.

Weighted Clinical Accuracy의 유형별 가중치는 연구자가 부여한 값이므로, 그 값에
결론이 얼마나 의존하는지를 따로 확인해야 한다. 이 스크립트는 4.4.4가 근거로 삼는
주장 — "임상 중요도로 가중해도 파인튜닝의 향상이 상쇄되지 않는다" — 이 특정
가중치 조합에서만 성립하는지, 아니면 가중 방식과 무관하게 성립하는지를 검증한다.

대조하는 가중 체계는 다섯 가지다. 현행 값, 모든 유형을 동일하게 보는 균등 가중,
임상 중요도 대비를 현행보다 키운 조합과 줄인 조합, 그리고 순서를 뒤집은 조합이다.
마지막 역순 조합은 현실적인 가정이 아니라, 결론이 가중치 순서에 의존하는지를
확인하기 위한 반대 방향 점검이다.

입력 수치는 논문 Table 4.4(평가 500건, Qwen3-VL-2B, PathVQA)와 같다.

사용:
    python3 scripts/wca_weight_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

BREAKDOWN = Path("results/phase3_autoresearch/clinical_type_breakdown.json")
CONDITION = "autoresearch_v2"  # Table 4.4의 최고 설정 (4.3.5 재실험)

SCHEMES: dict[str, dict[str, float]] = {
    # 논문 3.8.3의 현행 가중치
    "현행": {
        "diagnosis": 1.0, "location": 0.8, "measurement": 0.7,
        "description": 0.6, "temporal": 0.5, "yes_no": 0.5, "unknown": 0.5,
    },
    # 가중을 두지 않음 → 유형별 정확도의 단순 평균
    "균등": {
        "diagnosis": 1.0, "location": 1.0, "measurement": 1.0,
        "description": 1.0, "temporal": 1.0, "yes_no": 1.0, "unknown": 1.0,
    },
    # 임상 중요도 대비를 현행보다 크게
    "강한대비": {
        "diagnosis": 1.0, "location": 0.9, "measurement": 0.8,
        "description": 0.4, "temporal": 0.3, "yes_no": 0.2, "unknown": 0.2,
    },
    # 대비를 거의 없앰(균등에 가까움)
    "약한대비": {
        "diagnosis": 1.0, "location": 0.95, "measurement": 0.9,
        "description": 0.85, "temporal": 0.8, "yes_no": 0.8, "unknown": 0.8,
    },
    # 순서를 뒤집음 — 현실적 가정이 아니라 반대 방향 점검용
    "역순": {
        "diagnosis": 0.5, "location": 0.5, "measurement": 0.5,
        "description": 0.6, "temporal": 0.7, "yes_no": 0.8, "unknown": 1.0,
    },
}


def load_inputs() -> tuple[dict[str, float], dict[str, float], float, float]:
    """제로샷·최고 설정의 유형별 정확도를 분석 산출물에서 그대로 읽는다.

    논문 표의 반올림값을 옮겨 적으면 전사 오차가 섞이므로 원본을 읽는다.
    """
    data = json.loads(BREAKDOWN.read_text(encoding="utf-8"))
    zs = data["zeroshot"]["eval_500"]
    best = data["conditions"][CONDITION]["best"]

    # 표본이 0인 유형은 WCA에서 제외한다(관찰된 유형만 사용).
    counts = zs["per_type_count"]
    observed = [t for t, n in counts.items() if n > 0]

    zs_acc = {t: zs["per_type_accuracy"][t] for t in observed}
    best_acc = {t: best["per_type_accuracy"][t] for t in observed}
    return zs_acc, best_acc, zs["overall_accuracy"], best["val_accuracy"]


def wca(acc: dict[str, float], weights: dict[str, float]) -> float:
    """관찰된 유형만으로 가중 평균을 낸다 (analyze_clinical_types와 동일 정의)."""
    num = sum(acc[t] * weights[t] for t in acc)
    den = sum(weights[t] for t in acc)
    return num / den if den else 0.0


def main() -> None:
    zs_acc, best_acc, overall_zs, overall_best = load_inputs()
    overall_delta = overall_best - overall_zs
    print(f"관찰 유형: {', '.join(sorted(zs_acc))}")
    print(f"전체 정확도: 제로샷 {overall_zs:.4f} → 최고 설정 {overall_best:.4f} "
          f"({overall_delta:+.4f})\n")

    rows = []
    for name, weights in SCHEMES.items():
        zs = wca(zs_acc, weights)
        best = wca(best_acc, weights)
        rows.append(
            {
                "scheme": name,
                "zeroshot": round(zs, 4),
                "best": round(best, 4),
                "delta": round(best - zs, 4),
                "ratio_to_overall": round((best - zs) / overall_delta, 3),
            }
        )

    out = Path("results/phase3_autoresearch/wca_weight_sensitivity.json")
    out.write_text(
        json.dumps(
            {
                "condition": CONDITION,
                "overall_delta": round(overall_delta, 4),
                "schemes": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=== WCA 가중치 민감도 (Qwen3-VL-2B, PathVQA 평가 500건) ===")
    header = (
        f"{'가중 체계':<10} {'제로샷':>8} {'최고설정':>9} "
        f"{'증가폭':>9} {'전체증가폭 대비':>14}"
    )
    print(header)
    for r in rows:
        print(
            f"{r['scheme']:<10} {r['zeroshot']:>8.4f} {r['best']:>9.4f} "
            f"{r['delta']:>+9.4f} {r['ratio_to_overall']:>14.3f}"
        )
    deltas = [r["delta"] for r in rows]
    print(f"모든 체계에서 증가: {all(d > 0 for d in deltas)}  "
          f"(최소 {min(deltas):+.4f}, 최대 {max(deltas):+.4f})")
    print(f"\n기록: {out}")


if __name__ == "__main__":
    main()
