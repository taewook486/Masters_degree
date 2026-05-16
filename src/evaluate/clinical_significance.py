"""임상적 의미 분석 (WCA + ECE).

동료 심사 III번 지적("정확도 5%p 향상이 임상에서 무슨 의미?")에 대응한 보조 지표.

- WCA (Weighted Clinical Accuracy): PathVQA 7개 질문 유형 분류 + 임상 중요도 가중치
- ECE (Expected Calibration Error): 모델 자신감과 실제 정확도 일치도

참조:
- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
- FDA AI/ML SaMD guidance (2021)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# PathVQA 7개 질문 유형별 임상 중요도 가중치
CLINICAL_WEIGHTS: dict[str, float] = {
    "diagnosis": 1.0,    # Why, What is wrong with - 진단 결정
    "location": 0.8,     # Where - 병변 위치 식별
    "measurement": 0.7,  # How much, How many - 정량 측정
    "description": 0.6,  # What, How - 소견 기술
    "temporal": 0.5,     # When - 시간 정보
    "yes_no": 0.5,       # Yes/No - 이진 판단
    "unknown": 0.5,      # 분류 불가 - 기본값
}


# PathVQA 질문 유형 키워드 매핑 (소문자 기반)
QUESTION_TYPE_KEYWORDS: dict[str, list[str]] = {
    "diagnosis": [
        "why", "what is wrong", "what disease", "what condition",
        "what diagnosis", "what abnormality", "what is the diagnosis",
        "diagnose", "diagnosis of",
    ],
    "location": [
        "where", "which side", "location of", "located",
        "what region", "what area", "what part",
    ],
    "measurement": [
        "how much", "how many", "how large", "how big",
        "how small", "how long", "size of", "diameter",
    ],
    "temporal": [
        "when", "what stage", "what phase", "before or after",
        "duration", "time of",
    ],
}


@dataclass
class ClinicalMetrics:
    """임상적 의미 분석 결과."""
    wca: float                      # Weighted Clinical Accuracy
    ece: float                      # Expected Calibration Error
    per_type_accuracy: dict[str, float]  # 질문 유형별 정확도
    per_type_count: dict[str, int]       # 질문 유형별 샘플 수


def classify_clinical_type(question: str, answer: str) -> str:
    """PathVQA 질문을 임상 중요도 카테고리로 분류한다.

    Args:
        question: 질문 텍스트.
        answer: 정답 텍스트 (Yes/No 판별용).

    Returns:
        CLINICAL_WEIGHTS 키 중 하나.
    """
    answer_lower = answer.strip().lower()
    if answer_lower in {"yes", "no"}:
        return "yes_no"

    question_lower = question.strip().lower()
    for clinical_type, keywords in QUESTION_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in question_lower:
                return clinical_type

    # "what" 으로 시작하면 description으로 분류 (기본)
    if question_lower.startswith("what"):
        return "description"

    return "unknown"


def compute_wca(
    predictions: list[str],
    gold_answers: list[str],
    questions: list[str],
    is_correct_fn,
) -> tuple[float, dict[str, float], dict[str, int]]:
    """Weighted Clinical Accuracy를 계산한다.

    Args:
        predictions: 모델 예측 답변.
        gold_answers: 정답.
        questions: 질문 텍스트.
        is_correct_fn: (pred, gold, type) → bool 형태의 채점 함수.

    Returns:
        (wca_score, per_type_accuracy, per_type_count) 튜플.
    """
    if not predictions:
        return 0.0, {}, {}

    type_correct: dict[str, int] = {key: 0 for key in CLINICAL_WEIGHTS}
    type_total: dict[str, int] = {key: 0 for key in CLINICAL_WEIGHTS}

    for pred, gold, question in zip(predictions, gold_answers, questions):
        clinical_type = classify_clinical_type(question, gold)
        # is_correct_fn 시그니처가 question_type을 요구하므로 yes_no/open 변환
        question_type = "closed" if clinical_type == "yes_no" else "open"
        is_correct = is_correct_fn(pred, gold, question_type)

        type_total[clinical_type] += 1
        if is_correct:
            type_correct[clinical_type] += 1

    # 유형별 정확도
    per_type_accuracy: dict[str, float] = {}
    per_type_count: dict[str, int] = {}
    for ctype in CLINICAL_WEIGHTS:
        count = type_total[ctype]
        per_type_count[ctype] = count
        per_type_accuracy[ctype] = (
            type_correct[ctype] / count if count > 0 else 0.0
        )

    # WCA = Σ(유형별 정확도 × 가중치) / Σ가중치 (관찰된 유형만)
    weighted_sum = 0.0
    weight_sum = 0.0
    for ctype, weight in CLINICAL_WEIGHTS.items():
        if per_type_count[ctype] > 0:
            weighted_sum += per_type_accuracy[ctype] * weight
            weight_sum += weight

    wca = weighted_sum / weight_sum if weight_sum > 0 else 0.0
    return wca, per_type_accuracy, per_type_count


def compute_ece(
    correctness: list[bool],
    confidences: list[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error를 계산한다.

    수식: ECE = Σ_m (|B_m|/n) × |acc(B_m) - conf(B_m)|

    Args:
        correctness: 각 샘플의 정답 여부 (True/False).
        confidences: 각 샘플의 모델 confidence (0.0-1.0).
        n_bins: 신뢰도 구간 개수 (기본 10).

    Returns:
        ECE 값 (낮을수록 calibration 우수).
    """
    if not correctness or len(correctness) != len(confidences):
        return 0.0

    n = len(correctness)
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        # 마지막 빈은 양 끝 포함, 그 외는 (lo, hi]
        if i == n_bins - 1:
            in_bin = [
                idx for idx, c in enumerate(confidences) if lo <= c <= hi
            ]
        else:
            in_bin = [
                idx for idx, c in enumerate(confidences) if lo <= c < hi
            ]
        if not in_bin:
            continue

        bin_correctness = [correctness[idx] for idx in in_bin]
        bin_confidences = [confidences[idx] for idx in in_bin]
        bin_size = len(in_bin)

        acc_bin = sum(bin_correctness) / bin_size
        conf_bin = sum(bin_confidences) / bin_size

        ece += (bin_size / n) * abs(acc_bin - conf_bin)

    return round(ece, 4)


def compute_clinical_metrics(
    predictions: list[str],
    gold_answers: list[str],
    questions: list[str],
    is_correct_fn,
    confidences: list[float] | None = None,
) -> ClinicalMetrics:
    """WCA + ECE 통합 계산.

    Args:
        predictions: 모델 예측.
        gold_answers: 정답.
        questions: 질문 텍스트.
        is_correct_fn: 채점 함수 (pred, gold, question_type) → bool.
        confidences: 모델 confidence (선택, 없으면 ECE는 0 반환).

    Returns:
        ClinicalMetrics dataclass.
    """
    wca, per_type_acc, per_type_count = compute_wca(
        predictions, gold_answers, questions, is_correct_fn
    )

    ece = 0.0
    if confidences is not None and len(confidences) == len(predictions):
        correctness = [
            is_correct_fn(
                pred,
                gold,
                "closed" if classify_clinical_type(q, gold) == "yes_no" else "open",
            )
            for pred, gold, q in zip(predictions, gold_answers, questions)
        ]
        ece = compute_ece(correctness, confidences)

    return ClinicalMetrics(
        wca=round(wca, 4),
        ece=ece,
        per_type_accuracy={k: round(v, 4) for k, v in per_type_acc.items()},
        per_type_count=per_type_count,
    )
