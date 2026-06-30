"""metrics.py 테스트 (REQ-RI-001).

compute_overall_accuracy의 compute_bertscore 기본값이 True인지 확인.
BERTScore, closed/open accuracy, preprocess_answer 전반 테스트.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch


def test_compute_overall_accuracy_bertscore_default_is_true():
    """compute_overall_accuracy의 compute_bertscore 기본값이 True여야 한다."""
    from src.evaluate.metrics import compute_overall_accuracy

    sig = inspect.signature(compute_overall_accuracy)
    default = sig.parameters["compute_bertscore"].default
    assert default is True, f"compute_bertscore 기본값이 {default}이지만 True여야 함"


def test_compute_overall_accuracy_includes_bertscore_by_default():
    """기본 호출 시 BERTScore 키가 결과에 포함되어야 한다."""
    # bert_score.score를 mock하여 실제 모델 로드 없이 테스트
    mock_f1 = MagicMock()
    mock_f1.tolist.return_value = [0.85, 0.90]

    with patch("src.evaluate.metrics.bert_score_fn", create=True):
        with patch.dict("sys.modules", {"bert_score": MagicMock()}):
            with patch("src.evaluate.metrics.compute_open_bertscore") as mock_bs:
                mock_bs.return_value = {
                    "bertscore_f1_mean": 0.875,
                    "bertscore_accuracy": 1.0,
                    "bertscore_f1_scores": [0.85, 0.90],
                }
                from src.evaluate.metrics import compute_overall_accuracy

                predictions = ["the heart", "yes"]
                gold_answers = ["the heart", "yes"]
                question_types = ["open", "closed"]

                result = compute_overall_accuracy(predictions, gold_answers, question_types)
                assert "open_bertscore_f1" in result
                assert "open_bertscore_accuracy" in result


def test_compute_overall_accuracy_closed_only_no_bertscore():
    """closed-only 질문에서는 BERTScore 키가 포함되지 않아야 한다."""
    from src.evaluate.metrics import compute_overall_accuracy

    predictions = ["yes", "no"]
    gold_answers = ["yes", "no"]
    question_types = ["closed", "closed"]

    result = compute_overall_accuracy(predictions, gold_answers, question_types)
    assert "open_bertscore_f1" not in result


def test_preprocess_answer_normalizes():
    """preprocess_answer가 정규화를 올바르게 수행한다."""
    from src.evaluate.metrics import preprocess_answer

    assert preprocess_answer("  The Heart!  ") == "the heart"
    assert preprocess_answer("HELLO,  WORLD...") == "hello world"
    assert preprocess_answer("   ") == ""


def test_compute_closed_accuracy_perfect():
    """모든 예측이 정답일 때 정확도 1.0."""
    from src.evaluate.metrics import compute_closed_accuracy

    preds = ["yes", "no", "yes"]
    golds = ["yes", "no", "yes"]
    assert compute_closed_accuracy(preds, golds) == 1.0


def test_compute_closed_accuracy_empty():
    """빈 리스트일 때 0.0 반환."""
    from src.evaluate.metrics import compute_closed_accuracy

    assert compute_closed_accuracy([], []) == 0.0


def test_compute_closed_accuracy_partial():
    """일부만 맞을 때 올바른 비율 반환."""
    from src.evaluate.metrics import compute_closed_accuracy

    preds = ["yes", "yes", "no"]
    golds = ["yes", "no", "no"]
    assert compute_closed_accuracy(preds, golds) == 2 / 3


def test_extract_yes_no_variants():
    """_extract_yes_no가 다양한 변형을 올바르게 처리한다."""
    from src.evaluate.metrics import _extract_yes_no

    # yes 변형 (line 69-70)
    assert _extract_yes_no("yes") == "yes"
    assert _extract_yes_no("yeah") == "yes"
    assert _extract_yes_no("yep") == "yes"
    assert _extract_yes_no("correct") == "yes"
    assert _extract_yes_no("true") == "yes"

    # no 변형 (line 72)
    assert _extract_yes_no("no") == "no"
    assert _extract_yes_no("nope") == "no"
    assert _extract_yes_no("nah") == "no"
    assert _extract_yes_no("incorrect") == "no"
    assert _extract_yes_no("false") == "no"

    # starts with yes/no (lines 75-80)
    assert _extract_yes_no("yes the image shows") == "yes"
    assert _extract_yes_no("no there is no") == "no"

    # 그 외 반환 (line 80)
    assert _extract_yes_no("maybe") == "maybe"


def test_compute_open_accuracy_exact_match():
    """정확히 일치할 때 정확도 1.0."""
    from src.evaluate.metrics import compute_open_accuracy

    preds = ["the heart", "lung"]
    golds = ["the heart", "lung"]
    assert compute_open_accuracy(preds, golds) == 1.0


def test_compute_open_accuracy_recall_match():
    """gold가 prediction에 포함될 때 정답 처리."""
    from src.evaluate.metrics import compute_open_accuracy

    preds = ["the answer is heart", "it is the lung tissue"]
    golds = ["heart", "lung"]
    assert compute_open_accuracy(preds, golds) == 1.0


def test_compute_open_accuracy_empty():
    """빈 리스트일 때 0.0 반환."""
    from src.evaluate.metrics import compute_open_accuracy

    assert compute_open_accuracy([], []) == 0.0


def test_compute_open_bertscore_with_mock():
    """BERTScore 계산이 올바르게 동작한다 (bert_score mock)."""
    import torch

    mock_score = MagicMock()
    # bert_score.score는 (P, R, F1) 튜플을 반환
    mock_score.return_value = (
        torch.tensor([0.80, 0.90]),  # P
        torch.tensor([0.85, 0.88]),  # R
        torch.tensor([0.82, 0.89]),  # F1
    )

    with patch.dict("sys.modules", {"bert_score": MagicMock(score=mock_score)}):
        import importlib
        import src.evaluate.metrics as metrics_mod

        importlib.reload(metrics_mod)
        result = metrics_mod.compute_open_bertscore(
            ["pred1", "pred2"],
            ["gold1", "gold2"],
            threshold=0.7,
        )

    assert "bertscore_f1_mean" in result
    assert result["bertscore_f1_mean"] > 0
    assert "bertscore_accuracy" in result
    assert "bertscore_f1_scores" in result
    assert len(result["bertscore_f1_scores"]) == 2


def test_compute_open_bertscore_empty_predictions():
    """빈 prediction 리스트일 때 기본값 반환."""
    from src.evaluate.metrics import compute_open_bertscore

    result = compute_open_bertscore([], [])
    assert result["bertscore_f1_mean"] == 0.0
    assert result["bertscore_accuracy"] == 0.0
    assert result["bertscore_f1_scores"] == []


def test_compute_open_bertscore_import_error():
    """bert-score가 설치되지 않았을 때 기본값 반환."""
    with patch.dict("sys.modules", {"bert_score": None}):
        import importlib
        import src.evaluate.metrics as metrics_mod

        importlib.reload(metrics_mod)
        result = metrics_mod.compute_open_bertscore(["pred"], ["gold"])

    assert result["bertscore_f1_mean"] == 0.0


def test_compute_overall_accuracy_all_metrics():
    """전체 정확도 계산이 올바른 구조를 반환한다."""
    from src.evaluate.metrics import compute_overall_accuracy

    preds = ["yes", "heart", "no"]
    golds = ["yes", "heart", "yes"]
    qtypes = ["closed", "open", "closed"]

    result = compute_overall_accuracy(preds, golds, qtypes, compute_bertscore=False)
    assert "closed_accuracy" in result
    assert "open_accuracy" in result
    assert "overall_accuracy" in result
    assert "closed_count" in result
    assert "open_count" in result
    assert result["closed_count"] == 2
    assert result["open_count"] == 1


def test_compute_overall_accuracy_empty_input():
    """빈 입력일 때 0.0 반환."""
    from src.evaluate.metrics import compute_overall_accuracy

    result = compute_overall_accuracy([], [], [])
    assert result["overall_accuracy"] == 0.0
    assert result["total_count"] == 0
