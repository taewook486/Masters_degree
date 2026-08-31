"""llavamed_recall.py 테스트 — LLaVA-Med 원본 채점 재현 검증.

이 모듈의 존재 이유는 "우리 recall이 합리적인가"가 아니라 "LLaVA-Med v1.0.0의
recall과 같은 값을 내는가"이다. 따라서 원본 `calculate_f1score`를 참조 구현으로
그대로 옮겨 두고 차분 대조한다. 참조 구현을 고치면 비교의 전제가 깨지므로
원본과 다르게 만들지 않는다.

참조: microsoft/LLaVA-Med v1.0.0, llava/eval/eval_metrics/evaluate_metrics.py
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from src.evaluate.llavamed_recall import llavamed_recall
from src.evaluate.vendor.llavamed_glossary import normalize_word


def _reference_split_sentence(sentence: str, n: int) -> dict[str, int]:
    """원본 utils.split_sentence."""
    words: dict[str, int] = defaultdict(int)
    tmp_sentence = sentence.lower().strip().split()
    for i in range(len(tmp_sentence) - n + 1):
        tmp_words = " ".join(tmp_sentence[i : i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words


def _reference_f1score(candidate: str, reference: str):
    """원본 calculate_f1score. (f1, precision, recall)을 반환한다."""
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = _reference_split_sentence(candidate, 1)
    reference_words = _reference_split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)

    tp = fp = fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]

    if len(candidate_words) == 0:
        return 0, 0, 0
    elif len(reference_words) == 0:
        return 0, 0, 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0, 0, 0
        return 2 * precision * recall / (precision + recall), precision, recall


PAIRS = [
    # (생성 답변, 정답)
    ("CT", "CT"),
    ("Lung", "Lung, Spinal Cord"),
    ("the histone proteins", "the histone subunits"),
    ("right", "right"),
    ("", "lung"),
    ("lung", ""),
    ("", ""),
    ("lung lung", "lung"),
    ("the the a lung", "lung lung"),
    ("Right.", "right"),
    ("2 masses", "two masses"),
    ("no overlap here", "totally different"),
    ("Yes, there is a mass in the left upper lobe.", "mass"),
    ("chest x-ray", "Chest X-Ray"),
]


@pytest.mark.parametrize(("candidate", "reference"), PAIRS)
def test_matches_llavamed_reference(candidate: str, reference: str) -> None:
    """원본 recall과 값이 정확히 일치해야 한다."""
    _, _, expected = _reference_f1score(candidate, reference)
    assert llavamed_recall(candidate, reference) == pytest.approx(float(expected))


def test_full_overlap_is_one() -> None:
    assert llavamed_recall("lung", "lung") == pytest.approx(1.0)


def test_no_overlap_is_zero() -> None:
    assert llavamed_recall("liver", "lung") == 0.0


def test_partial_overlap_is_fraction() -> None:
    """정답 2토큰 중 1토큰만 맞히면 1/2."""
    assert llavamed_recall("lung", "lung heart") == pytest.approx(0.5)


def test_empty_inputs_do_not_raise() -> None:
    assert llavamed_recall("", "") == 0.0
    assert llavamed_recall(None, None) == 0.0
