"""VQA evaluation metrics for medical VQA."""

from __future__ import annotations

import re
import string


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


def _extract_yes_no(text: str) -> str:
    """Extract yes/no from a text string.

    Handles variations like 'yeah', 'yep', 'nope', 'nah', and
    model outputs like 'yes the image shows...' or 'no there is no...'.
    """
    text = text.strip()

    if text in {"yes", "yeah", "yep", "correct", "true"}:
        return "yes"
    if text in {"no", "nope", "nah", "incorrect", "false"}:
        return "no"

    # Check if the answer starts with yes/no
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"

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


def compute_overall_accuracy(
    predictions: list[str],
    gold_answers: list[str],
    question_types: list[str],
) -> dict[str, float | int]:
    """Compute closed, open, and overall accuracy.

    Args:
        predictions: Model-generated answers.
        gold_answers: Ground truth answers.
        question_types: List of "open" or "closed" for each sample.

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

    return {
        "closed_accuracy": round(closed_acc, 4),
        "open_accuracy": round(open_acc, 4),
        "overall_accuracy": round(overall_acc, 4),
        "closed_count": len(closed_preds),
        "open_count": len(open_preds),
        "total_count": total,
    }
