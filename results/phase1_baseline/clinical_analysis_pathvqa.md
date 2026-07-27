# 임상적 의미 분석 (WCA) — pathvqa (seed 42)

- 지표: Weighted Clinical Accuracy (WCA) + 질문 유형별 정확도
- 가중치: diagnosis=1.0, location=0.8, measurement=0.7, description=0.6, temporal=0.5, yes_no=0.5, unknown=0.5
- ECE: N/A — per-sample confidence 미저장 (산출 불가)

> **주의(§5.3)**: WCA 가중치는 외부 검증 없는 임시 척도다. 절대적 임상 중요도로 해석 불가하며, primary 지표(정확도, BERTScore)를 보완하는 참고용이다.

## 모델별 WCA 요약

| 모델 | Overall Acc | WCA | n |
|------|:-----------:|:---:|:-:|
| gemma4-e2b | 0.1055 | 0.0743 | 6719 |
| qwen25-vl-3b | 0.3245 | 0.1094 | 6719 |
| qwen3-vl-2b | 0.3472 | 0.1335 | 6719 |
| smolvlm2-2b | 0.3085 | 0.0954 | 6719 |

## gemma4-e2b — 질문 유형별 정확도

| 유형 | 가중치 | 정확도 | 샘플 수 |
|------|:------:|:------:|:-------:|
| diagnosis | 1.0 | 0.0000 | 23 |
| location | 0.8 | 0.0647 | 433 |
| measurement | 0.7 | 0.2121 | 33 |
| description | 0.6 | 0.0425 | 2729 |
| temporal | 0.5 | 0.0000 | 9 |
| yes_no | 0.5 | 0.1633 | 3362 |
| unknown | 0.5 | 0.0692 | 130 |

## qwen25-vl-3b — 질문 유형별 정확도

| 유형 | 가중치 | 정확도 | 샘플 수 |
|------|:------:|:------:|:-------:|
| diagnosis | 1.0 | 0.0000 | 23 |
| location | 0.8 | 0.1062 | 433 |
| measurement | 0.7 | 0.0909 | 33 |
| description | 0.6 | 0.0224 | 2729 |
| temporal | 0.5 | 0.0000 | 9 |
| yes_no | 0.5 | 0.6130 | 3362 |
| unknown | 0.5 | 0.0692 | 130 |

## qwen3-vl-2b — 질문 유형별 정확도

| 유형 | 가중치 | 정확도 | 샘플 수 |
|------|:------:|:------:|:-------:|
| diagnosis | 1.0 | 0.0000 | 23 |
| location | 0.8 | 0.1478 | 433 |
| measurement | 0.7 | 0.1515 | 33 |
| description | 0.6 | 0.0447 | 2729 |
| temporal | 0.5 | 0.0000 | 9 |
| yes_no | 0.5 | 0.6336 | 3362 |
| unknown | 0.5 | 0.0923 | 130 |

## smolvlm2-2b — 질문 유형별 정확도

| 유형 | 가중치 | 정확도 | 샘플 수 |
|------|:------:|:------:|:-------:|
| diagnosis | 1.0 | 0.0000 | 23 |
| location | 0.8 | 0.0762 | 433 |
| measurement | 0.7 | 0.0909 | 33 |
| description | 0.6 | 0.0198 | 2729 |
| temporal | 0.5 | 0.0000 | 9 |
| yes_no | 0.5 | 0.5892 | 3362 |
| unknown | 0.5 | 0.0154 | 130 |
