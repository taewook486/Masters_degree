"""VQA evaluation metrics for medical VQA.

Metrics:
  - Closed-ended accuracy (yes/no questions)
  - Open-ended accuracy: Exact Match + BERTScore F1
  - Overall weighted accuracy
"""

from __future__ import annotations

import logging
import re
import string

from scipy import stats

logger = logging.getLogger(__name__)

# REQ-EM-001, REQ-EM-005: BioBERT는 dual BERTScore의 유일한 지원 secondary
# (의료 특화) 모델이다. bert-score 0.3.13의 model2layers 레지스트리에 미등록
# 가능성이 높아 명시적 num_layers가 필요하다 (BioBERT-v1.1 = BERT-base 12층,
# BERTScore 원논문의 bert-base 경험적 최적 레이어 = 9. plan.md 참조).
_BIOBERT_MODEL_ID = "dmis-lab/biobert-v1.1"
_BIOBERT_NUM_LAYERS = 9


def preprocess_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Steps:
        1. Strip whitespace
        2. Lowercase
        3. Remove punctuation
        4. Collapse multiple spaces
    """
    answer = answer.strip().lower()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def compute_closed_accuracy(
    predictions: list[str],
    gold_answers: list[str],
) -> float:
    """Compute accuracy for closed-ended (yes/no) questions.

    Both predictions and golds are preprocessed before comparison.
    Returns accuracy as a float between 0.0 and 1.0.
    Returns 0.0 if the list is empty.
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        pred_clean = preprocess_answer(pred)
        gold_clean = preprocess_answer(gold)

        # Extract yes/no from potentially verbose model outputs
        pred_yn = _extract_yes_no(pred_clean)
        gold_yn = _extract_yes_no(gold_clean)

        if pred_yn == gold_yn:
            correct += 1

    return correct / len(predictions)


# v0.6: 확답 회피/불확실성 문구 — 이런 답은 yes/no로 매핑하지 않고 비확답 처리한다.
# (장황한 출력에서 yes/no를 추출할 때 오탐을 막기 위한 가드)
_REFUSAL_MARKERS = (
    "not possible", "cannot", "can not", "unable", "not able",
    "insufficient", "impossible", "not enough", "cannot be determined",
    "difficult to determine", "cannot determine",
)


def _extract_yes_no(text: str) -> str:
    """Extract yes/no from a (preprocessed, lowercased, no-punct) answer string.

    Handles short forms ('yeah', 'nope', ...), leading yes/no, and — v0.6 —
    yes/no embedded in verbose outputs (e.g. 'the answer is yes'). Refusal /
    uncertainty phrasing ('not possible to determine', 'cannot ...') is treated
    as a non-answer so it does not spuriously map to yes/no.
    """
    text = text.strip()

    if text in {"yes", "yeah", "yep", "correct", "true"}:
        return "yes"
    if text in {"no", "nope", "nah", "incorrect", "false"}:
        return "no"

    # Leading yes/no (기존 동작 유지)
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"

    # v0.6: 회피/불확실 문구는 비확답 → 원문 반환 (yes/no와 불일치 처리)
    if any(marker in text for marker in _REFUSAL_MARKERS):
        return text

    # v0.6: 문장 속 단어 경계 yes/no 추출 (한쪽만 있을 때만 채택)
    has_yes = re.search(r"\byes\b", text) is not None
    has_no = re.search(r"\bno\b", text) is not None
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"

    # 둘 다/둘 다 없음 → 'answer is yes/no' 명시 패턴만 마지막으로 확인
    m = re.search(r"answer(?:\s+is|:)?\s+(yes|no)\b", text)
    if m:
        return m.group(1)

    return text


def compute_open_accuracy(
    predictions: list[str],
    gold_answers: list[str],
) -> float:
    """Compute accuracy for open-ended questions.

    Uses two matching strategies:
        1. Exact match after preprocessing
        2. Recall match: gold answer is contained in prediction

    Returns accuracy as a float between 0.0 and 1.0.
    Returns 0.0 if the list is empty.
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        pred_clean = preprocess_answer(pred)
        gold_clean = preprocess_answer(gold)

        # Exact match or recall match (gold contained in prediction)
        if pred_clean == gold_clean or gold_clean in pred_clean:
            correct += 1

    return correct / len(predictions)


def compute_open_bertscore(
    predictions: list[str],
    gold_answers: list[str],
    threshold: float = 0.7,
    model_type: str = "roberta-large",
    num_layers: int | None = None,
) -> dict[str, float]:
    """Compute BERTScore F1 for open-ended questions.

    Uses roberta-large as the base model by default (v0.2 spec).
    A prediction is considered correct if BERTScore F1 >= threshold.

    Args:
        predictions: Model-generated answers.
        gold_answers: Ground truth answers.
        threshold: BERTScore F1 threshold for correctness (default: 0.7).
        model_type: BERTScore model (default: roberta-large).
        num_layers: Explicit embedding layer count passed to the BERTScore
            backend (REQ-EM-005). Required for models not registered in
            bert-score's model2layers registry (e.g. dmis-lab/biobert-v1.1).
            Left unspecified (None, default) for registry-covered models such
            as roberta-large, whose layer count is not redefined.

    Returns:
        Dict with mean F1, accuracy at threshold, and per-sample F1 scores.
    """
    if not predictions:
        return {"bertscore_f1_mean": 0.0, "bertscore_accuracy": 0.0, "bertscore_f1_scores": []}

    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        logger.warning("bert-score not installed. Run: pip install bert-score")
        return {"bertscore_f1_mean": 0.0, "bertscore_accuracy": 0.0, "bertscore_f1_scores": []}

    score_kwargs: dict[str, str | int | bool] = {
        "model_type": model_type,
        "verbose": False,
    }
    if num_layers is not None:
        score_kwargs["num_layers"] = num_layers

    _, _, f1 = bert_score_fn(predictions, gold_answers, **score_kwargs)

    f1_list = f1.tolist()
    f1_mean = sum(f1_list) / len(f1_list)
    correct = sum(1 for s in f1_list if s >= threshold)
    accuracy = correct / len(f1_list)

    return {
        "bertscore_f1_mean": round(f1_mean, 4),
        "bertscore_accuracy": round(accuracy, 4),
        "bertscore_f1_scores": [round(s, 4) for s in f1_list],
    }


def compute_overall_accuracy(
    predictions: list[str],
    gold_answers: list[str],
    question_types: list[str],
    compute_bertscore: bool = True,
    bertscore_models: list[str] | None = None,
) -> dict[str, float | int]:
    """Compute closed, open, and overall accuracy.

    Args:
        predictions: Model-generated answers.
        gold_answers: Ground truth answers.
        question_types: List of "open" or "closed" for each sample.
        compute_bertscore: If True, also compute BERTScore F1 for open-ended.
        bertscore_models: Optional list of BERTScore model identifiers to
            score in addition to the primary roberta-large model (REQ-EM-001,
            Phase 2 opt-in only). When None or omitted (default), behavior is
            unchanged from before this parameter existed: only roberta-large
            is scored and only the primary result keys (`open_bertscore_f1`,
            `open_bertscore_accuracy`) are produced (REQ-EM-002 — Phase 1
            backward compatibility). roberta-large is ALWAYS the primary/
            decision metric (REQ-EM-004) regardless of this list's contents;
            any other entry (e.g. "dmis-lab/biobert-v1.1") is treated as a
            secondary (informational-only) model and reported under the
            `_biobert` result keys plus Spearman/Pearson correlation against
            the primary per-sample F1 vector (REQ-EM-003).

    Returns:
        Dictionary with accuracy metrics and counts.
    """
    closed_preds, closed_golds = [], []
    open_preds, open_golds = [], []

    for pred, gold, qtype in zip(predictions, gold_answers, question_types):
        if qtype == "closed":
            closed_preds.append(pred)
            closed_golds.append(gold)
        else:
            open_preds.append(pred)
            open_golds.append(gold)

    closed_acc = compute_closed_accuracy(closed_preds, closed_golds)
    open_acc = compute_open_accuracy(open_preds, open_golds)

    total = len(predictions)
    total_correct = 0
    if closed_preds:
        total_correct += round(closed_acc * len(closed_preds))
    if open_preds:
        total_correct += round(open_acc * len(open_preds))

    overall_acc = total_correct / total if total > 0 else 0.0

    result: dict[str, float | int | list] = {
        "closed_accuracy": round(closed_acc, 4),
        "open_accuracy": round(open_acc, 4),
        "overall_accuracy": round(overall_acc, 4),
        "closed_count": len(closed_preds),
        "open_count": len(open_preds),
        "total_count": total,
    }

    if compute_bertscore and open_preds:
        # REQ-EM-004: roberta-large는 항상 primary/결정 지표 — bertscore_models
        # 인자의 내용과 무관하게 재정의되지 않는다 (비-게이팅).
        bs = compute_open_bertscore(open_preds, open_golds, model_type="roberta-large")
        result["open_bertscore_f1"] = bs["bertscore_f1_mean"]
        result["open_bertscore_accuracy"] = bs["bertscore_accuracy"]

        # REQ-EM-001: roberta-large 외 요청된 모델은 secondary(정보 제공용)로
        # 계산한다. 현재 지원 대상은 BioBERT 하나뿐이며(결과 키 스키마가
        # `_biobert`로 고정), 그 외 항목이 있으면 첫 번째만 secondary로 취급한다.
        secondary_models = [m for m in (bertscore_models or []) if m != "roberta-large"]
        if secondary_models:
            secondary_model = secondary_models[0]
            secondary_num_layers = (
                _BIOBERT_NUM_LAYERS if secondary_model == _BIOBERT_MODEL_ID else None
            )
            bs_secondary = compute_open_bertscore(
                open_preds,
                open_golds,
                model_type=secondary_model,
                num_layers=secondary_num_layers,
            )
            result["open_bertscore_f1_biobert"] = bs_secondary["bertscore_f1_mean"]
            result["open_bertscore_accuracy_biobert"] = bs_secondary[
                "bertscore_accuracy"
            ]

            # REQ-EM-003: Spearman(주 보고값) + Pearson(병기). 표본 < 2이면
            # 상관이 정의되지 않으므로 계산을 생략한다(키 자체를 만들지 않음).
            primary_scores = bs["bertscore_f1_scores"]
            secondary_scores = bs_secondary["bertscore_f1_scores"]
            same_len = len(primary_scores) == len(secondary_scores)
            if len(primary_scores) >= 2 and same_len:
                spearman = stats.spearmanr(primary_scores, secondary_scores).correlation
                pearson = stats.pearsonr(primary_scores, secondary_scores)[0]
                result["open_bertscore_spearman"] = round(float(spearman), 4)
                result["open_bertscore_pearson"] = round(float(pearson), 4)

    return result
