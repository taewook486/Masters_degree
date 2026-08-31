"""LLaVA-Med와 동일한 open-ended 채점 기준(recall) 재현.

논문 4.4.6은 주관식 성능을 BERTScore F1 >= 0.7 통과 여부로 채점하는 반면,
LLaVA-Med는 정답 토큰이 생성 응답에 나타난 비율(recall)을 사용한다. 척도가 달라
두 수치를 직접 비교할 수 없다는 것이 5.3 (2)가 지적한 한계다.

이 모듈은 LLaVA-Med v1.0.0의 `calculate_f1score`가 반환하는 recall을 그대로
재현해, 저장된 예측 원본에 동일 기준을 적용할 수 있게 한다. 구현상의 특이점
(tp는 후보 쪽 빈도로, fn은 정답 쪽 빈도로 세는 비대칭)까지 원본을 따른다 —
비교 가능성이 목적이므로 "더 올바른" 변형을 만들면 안 된다.

출처: microsoft/LLaVA-Med, tag v1.0.0
      llava/eval/eval_metrics/evaluate_metrics.py `calculate_f1score`
      llava/eval/run_eval.py (open 질문에 대해 recall을 평균)
"""

from __future__ import annotations

from collections import defaultdict

from .vendor.llavamed_glossary import normalize_word


def split_sentence(sentence: str, n: int = 1) -> dict[str, int]:
    """LLaVA-Med `utils.split_sentence`와 동일한 n-gram 빈도 집계."""
    words: dict[str, int] = defaultdict(int)
    tmp = sentence.lower().strip().split()
    for i in range(len(tmp) - n + 1):
        gram = " ".join(tmp[i : i + n])
        if gram:
            words[gram] += 1
    return words


def llavamed_recall(candidate: str, reference: str) -> float:
    """LLaVA-Med `calculate_f1score`의 recall 성분.

    candidate: 모델 생성 답변, reference: 정답.
    빈 입력이나 겹치는 토큰이 없으면 원본과 동일하게 0을 반환한다.
    """
    candidate = normalize_word(candidate or "")
    reference = normalize_word(reference or "")

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)

    if len(candidate_words) == 0 or len(reference_words) == 0:
        return 0.0

    word_set = set(candidate_words) | set(reference_words)
    tp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]

    if tp == 0:
        return 0.0
    return tp / (tp + fn)
