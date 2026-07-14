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


# ---------------------------------------------------------------------------
# SPEC-EVAL-METRICS-001: BioBERT 이중 BERTScore (num_layers, dual-model, 상관)
# ---------------------------------------------------------------------------


def test_compute_open_bertscore_biobert_num_layers_passed_to_backend():
    """AC-1-3a, REQ-EM-005: 의료 모델 num_layers=9 백엔드 전달 (엣지케이스).

    BioBERT(dmis-lab/biobert-v1.1)가 채점 대상으로 선택되면, 채점 백엔드를
    num_layers=9 인자와 함께 호출해야 한다 (백엔드 mock으로 호출 인자 캡처).
    """
    import torch

    mock_score = MagicMock()
    mock_score.return_value = (
        torch.tensor([0.80, 0.90]),
        torch.tensor([0.85, 0.88]),
        torch.tensor([0.82, 0.89]),
    )

    with patch.dict("sys.modules", {"bert_score": MagicMock(score=mock_score)}):
        import importlib
        import src.evaluate.metrics as metrics_mod

        importlib.reload(metrics_mod)
        metrics_mod.compute_open_bertscore(
            ["pred1", "pred2"],
            ["gold1", "gold2"],
            model_type="dmis-lab/biobert-v1.1",
            num_layers=9,
        )

    _, call_kwargs = mock_score.call_args
    assert call_kwargs.get("num_layers") == 9


def test_compute_open_bertscore_roberta_large_no_num_layers_override():
    """AC-1-3b, REQ-EM-005: 범용 모델 레이어 수 비재정의.

    roberta-large 채점 시 임베딩 레이어 수를 재정의하지 않아야 한다
    (레이어 레지스트리 기본값 사용, num_layers 미지정 — 호출 인자에
    num_layers 키 자체가 없어야 함).
    """
    import torch

    mock_score = MagicMock()
    mock_score.return_value = (
        torch.tensor([0.80, 0.90]),
        torch.tensor([0.85, 0.88]),
        torch.tensor([0.82, 0.89]),
    )

    with patch.dict("sys.modules", {"bert_score": MagicMock(score=mock_score)}):
        import importlib
        import src.evaluate.metrics as metrics_mod

        importlib.reload(metrics_mod)
        metrics_mod.compute_open_bertscore(
            ["pred1", "pred2"],
            ["gold1", "gold2"],
            model_type="roberta-large",
        )

    _, call_kwargs = mock_score.call_args
    assert "num_layers" not in call_kwargs


def test_compute_overall_accuracy_default_omits_biobert_and_correlation_keys():
    """AC-1-1, REQ-EM-002: 하위 호환 (Phase 1 보존).

    이중 BERTScore 평가를 지정하지 않고(기본값) overall-accuracy 계산이
    호출되면, 결과에 범용 primary 키(open_bertscore_f1, open_bertscore_accuracy)만
    포함하고 의료/상관 키(open_bertscore_f1_biobert, open_bertscore_spearman,
    open_bertscore_pearson)는 포함하지 않으며, 기존 roberta-large 단일 동작을
    유지해야 한다(BERTScore 백엔드는 정확히 1회만 호출됨).
    """
    from src.evaluate.metrics import compute_overall_accuracy

    with patch("src.evaluate.metrics.compute_open_bertscore") as mock_bs:
        mock_bs.return_value = {
            "bertscore_f1_mean": 0.85,
            "bertscore_accuracy": 1.0,
            "bertscore_f1_scores": [0.85, 0.90],
        }
        predictions = ["the heart", "the lung"]
        gold_answers = ["the heart", "the lung"]
        question_types = ["open", "open"]

        result = compute_overall_accuracy(predictions, gold_answers, question_types)

    assert "open_bertscore_f1" in result
    assert "open_bertscore_accuracy" in result
    assert "open_bertscore_f1_biobert" not in result
    assert "open_bertscore_spearman" not in result
    assert "open_bertscore_pearson" not in result
    assert mock_bs.call_count == 1


def test_compute_overall_accuracy_dual_model_returns_biobert_and_correlation():
    """AC-1-2, REQ-EM-001, REQ-EM-003: 이중 모델 계산 + 상관.

    호출자가 범용·의료 두 BERTScore 모델을 명시적으로 요청한 상태에서
    overall-accuracy 계산이 실행되면(백엔드 mock이 모델별로 상이한 표본별
    F1을 반환), 결과에 open_bertscore_f1(범용), open_bertscore_f1_biobert(의료),
    그리고 두 모델의 표본별 F1 벡터로부터 산출한 open_bertscore_spearman·
    open_bertscore_pearson을 모두 포함해야 한다.
    """
    from scipy import stats as scipy_stats

    from src.evaluate.metrics import compute_overall_accuracy

    roberta_scores = [0.90, 0.85, 0.80, 0.75, 0.70]
    biobert_scores = [0.88, 0.80, 0.82, 0.70, 0.65]

    def _fake_compute_open_bertscore(
        preds, golds, threshold=0.7, model_type="roberta-large", num_layers=None
    ):
        is_biobert = model_type == "dmis-lab/biobert-v1.1"
        scores = biobert_scores if is_biobert else roberta_scores
        return {
            "bertscore_f1_mean": round(sum(scores) / len(scores), 4),
            "bertscore_accuracy": round(
                sum(1 for s in scores if s >= threshold) / len(scores), 4
            ),
            "bertscore_f1_scores": scores,
        }

    with patch(
        "src.evaluate.metrics.compute_open_bertscore",
        side_effect=_fake_compute_open_bertscore,
    ):
        predictions = ["a", "b", "c", "d", "e"]
        gold_answers = ["a", "b", "c", "d", "e"]
        question_types = ["open"] * 5

        result = compute_overall_accuracy(
            predictions,
            gold_answers,
            question_types,
            bertscore_models=["roberta-large", "dmis-lab/biobert-v1.1"],
        )

    expected_spearman = round(
        float(scipy_stats.spearmanr(roberta_scores, biobert_scores).correlation), 4
    )
    expected_pearson = round(
        float(scipy_stats.pearsonr(roberta_scores, biobert_scores)[0]), 4
    )
    expected_roberta_f1 = round(sum(roberta_scores) / len(roberta_scores), 4)
    expected_biobert_f1 = round(sum(biobert_scores) / len(biobert_scores), 4)

    assert result["open_bertscore_f1"] == expected_roberta_f1
    assert result["open_bertscore_f1_biobert"] == expected_biobert_f1
    assert result["open_bertscore_spearman"] == expected_spearman
    assert result["open_bertscore_pearson"] == expected_pearson


def test_compute_overall_accuracy_primary_metrics_unaffected_by_biobert():
    """AC-1-4, REQ-EM-004: primary 지표 불변 / 비-게이팅.

    correctness, overall_accuracy, open_bertscore_accuracy는 범용(roberta-large)
    경로에만 근거하여 산출되어야 하며, 의료 모델 F1 값이 크게 달라지더라도
    이들 primary 수치가 변경되지 않아야 한다(이중 게이팅 없음).
    """
    from src.evaluate.metrics import compute_overall_accuracy

    roberta_scores = [0.90, 0.85, 0.80]

    def _make_fake(biobert_scores):
        def _fake(
            preds, golds, threshold=0.7, model_type="roberta-large", num_layers=None
        ):
            is_biobert = model_type == "dmis-lab/biobert-v1.1"
            scores = biobert_scores if is_biobert else roberta_scores
            return {
                "bertscore_f1_mean": round(sum(scores) / len(scores), 4),
                "bertscore_accuracy": round(
                    sum(1 for s in scores if s >= threshold) / len(scores), 4
                ),
                "bertscore_f1_scores": scores,
            }

        return _fake

    predictions = ["a", "b", "c"]
    gold_answers = ["a", "b", "c"]
    question_types = ["open"] * 3

    with patch(
        "src.evaluate.metrics.compute_open_bertscore",
        side_effect=_make_fake([0.10, 0.05, 0.02]),  # BioBERT 극단적으로 낮은 F1
    ):
        result_low_biobert = compute_overall_accuracy(
            predictions,
            gold_answers,
            question_types,
            bertscore_models=["roberta-large", "dmis-lab/biobert-v1.1"],
        )

    with patch(
        "src.evaluate.metrics.compute_open_bertscore",
        side_effect=_make_fake([0.99, 0.98, 0.97]),  # BioBERT 극단적으로 높은 F1
    ):
        result_high_biobert = compute_overall_accuracy(
            predictions,
            gold_answers,
            question_types,
            bertscore_models=["roberta-large", "dmis-lab/biobert-v1.1"],
        )

    assert (
        result_low_biobert["open_bertscore_accuracy"]
        == result_high_biobert["open_bertscore_accuracy"]
    )
    assert (
        result_low_biobert["open_bertscore_f1"]
        == result_high_biobert["open_bertscore_f1"]
    )
    assert (
        result_low_biobert["overall_accuracy"]
        == result_high_biobert["overall_accuracy"]
    )
    assert (
        result_low_biobert["open_bertscore_f1_biobert"]
        != result_high_biobert["open_bertscore_f1_biobert"]
    )


def test_compute_overall_accuracy_correlation_omitted_for_single_sample():
    """상관 미정의 엣지케이스 (plan.md 설계 노트): 표본 < 2이면 상관 생략.

    open 샘플이 1개뿐이면 open_bertscore_spearman/pearson 키가 생성되지
    않아야 한다.
    """
    from src.evaluate.metrics import compute_overall_accuracy

    def _fake(preds, golds, threshold=0.7, model_type="roberta-large", num_layers=None):
        scores = [0.8] if model_type == "dmis-lab/biobert-v1.1" else [0.9]
        return {
            "bertscore_f1_mean": scores[0],
            "bertscore_accuracy": 1.0,
            "bertscore_f1_scores": scores,
        }

    with patch("src.evaluate.metrics.compute_open_bertscore", side_effect=_fake):
        result = compute_overall_accuracy(
            ["a"],
            ["a"],
            ["open"],
            bertscore_models=["roberta-large", "dmis-lab/biobert-v1.1"],
        )

    assert "open_bertscore_f1_biobert" in result
    assert "open_bertscore_spearman" not in result
    assert "open_bertscore_pearson" not in result
