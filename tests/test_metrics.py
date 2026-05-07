"""metrics.py 테스트 (REQ-RI-001).

compute_overall_accuracy의 compute_bertscore 기본값이 True인지 확인.
"""

from __future__ import annotations

import inspect


def test_compute_overall_accuracy_bertscore_default_is_true():
    """compute_overall_accuracy의 compute_bertscore 기본값이 True여야 한다."""
    from src.evaluate.metrics import compute_overall_accuracy

    sig = inspect.signature(compute_overall_accuracy)
    default = sig.parameters["compute_bertscore"].default
    assert default is True, f"compute_bertscore 기본값이 {default}이지만 True여야 함"


def test_compute_overall_accuracy_includes_bertscore_by_default():
    """기본 호출 시 BERTScore 키가 결과에 포함되어야 한다."""
    from src.evaluate.metrics import compute_overall_accuracy

    # open-ended 질문이 있으면 BERTScore가 기본 포함되어야 한다
    predictions = ["the heart", "yes"]
    gold_answers = ["the heart", "yes"]
    question_types = ["open", "closed"]

    result = compute_overall_accuracy(predictions, gold_answers, question_types)
    assert "open_bertscore_f1" in result, "BERTScore F1이 기본으로 포함되어야 함"
    assert "open_bertscore_accuracy" in result


def test_compute_overall_accuracy_closed_only_no_bertscore():
    """closed-only 질문에서는 BERTScore 키가 포함되지 않아야 한다."""
    from src.evaluate.metrics import compute_overall_accuracy

    predictions = ["yes", "no"]
    gold_answers = ["yes", "no"]
    question_types = ["closed", "closed"]

    result = compute_overall_accuracy(predictions, gold_answers, question_types)
    # open 질문이 없으면 BERTScore가 계산되지 않음
    assert "open_bertscore_f1" not in result
