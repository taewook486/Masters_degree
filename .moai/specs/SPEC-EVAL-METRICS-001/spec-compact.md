# SPEC-EVAL-METRICS-001 (Compact)

id: SPEC-EVAL-METRICS-001 | status: draft | priority: high | issue: 0 | version: 0.1.1

## REQ (EARS)

- REQ-EM-001 (Optional): **Where** 이중 BERTScore 평가를 명시 요청하면, 요청된 각 모델(범용 roberta-large, 의료 BioBERT)의 open-ended F1을 각각 계산한다(shall).
- REQ-EM-002 (Ubiquitous): **항상** 이중 평가 미요청 시 범용 모델(roberta-large) 단일 채점만 수행하여 Phase 1 스키마·동작을 보존한다(shall).
- REQ-EM-003 (Event): **When** 이중 모델 계산 시, 표본별 F1 간 Spearman(주)+Pearson(병기) 상관을 결과에 포함한다(shall).
- REQ-EM-004 (Ubiquitous): **항상** 범용 모델@0.7이 유일 결정 지표, 의료 모델은 secondary, 이중 게이팅 금지(shall not).
- REQ-EM-005 (Unwanted): **If** 의료 모델이 표준 레이어 자동결정에 실패할 수 있으면, **then** 문서화된 고정 임베딩 레이어 수를 명시 전달해 채점 실패를 방지한다(shall).
- REQ-EM-006 (Ubiquitous): **항상** 2-독립표본 양측 순위합 검정(Mann-Whitney U, 비모수) 기능을 제공한다(shall).
- REQ-EM-007 (Ubiquitous): **항상** 순위합 검정에 효과 크기(rank-biserial)를 병기하고, 부호는 첫 표본이 둘째 표본보다 큰 경향일 때 양수로 정의·문서화한다(shall).
- REQ-EM-008 (Ubiquitous): **항상** 유의성 판정을 형제 통계 함수와 동일 규약(유의수준 0.05 기준 bool)으로 보고한다(shall).
- REQ-EM-009 (Unwanted): **If** 동순위 존재 시, **then** 표준 정규근사 동점 보정으로 예외 없이 유효 p-value∈[0,1]를 반환한다(shall).

## Acceptance

- AC-1-1 [REQ-EM-002]: 이중 평가 미지정 → 범용 primary 키만 포함, 의료/상관 키 부재, roberta 단일 동작 유지.
- AC-1-2 [REQ-EM-001,003]: 이중 모델 요청 → f1/f1_biobert/spearman/pearson 모두 존재, spearman이 상관 주값.
- AC-1-3a [REQ-EM-005]: 의료 모델 채점 시 백엔드가 num_layers=9로 호출, KeyError 없이 완료.
- AC-1-3b [REQ-EM-005]: 범용 모델 채점 시 레이어 수 미재정의(레지스트리 기본).
- AC-1-4 [REQ-EM-004]: overall/open_bertscore_accuracy는 범용 경로에만 근거, 의료 F1 변화 무영향.
- AC-2-1 [REQ-EM-006,008]: 순위합 반환 구조(u_stat, p_value∈[0,1], n1, n2, rank_biserial_r, significant).
- AC-2-2 [REQ-EM-009]: 동순위 표본 → 예외 없이 유효 p_value.
- AC-2-3a [REQ-EM-007]: 완전 분리(x>y) → rank_biserial_r 양의 극단(+1 근처), significant=True.
- AC-2-3b [REQ-EM-007]: 인자 반전(y,x) → rank_biserial_r 부호 음수 반전.
- AC-2-4 [REQ-EM-008]: significant가 형제 함수와 동일 규약(p<0.05 bool).

## Files to Modify

행위 요약. 함수/파라미터명·반환 dict 키·호출 시그니처·고정 레이어 수 값 등 구현 명세는 `plan.md` 참조.

- `src/evaluate/metrics.py` (수정: 요청 시 모델별 BERTScore 이중 계산 + 상관, 의료 모델 레이어 수 안전장치)
- `src/baseline/evaluate_zero_shot.py` (수정: 이중 평가 요청 선택적 전달, 미지정 시 기존 동작 유지)
- `src/finetune/train_qlora.py` (수정: Phase 2 평가에서 이중 모델 평가 활성화)
- `src/evaluate/statistics.py` (신규 함수: 2-독립표본 양측 순위합 검정)
- `tests/test_metrics.py` (테스트 추가: 이중 모델/상관/레이어 수 전달, 백엔드 mock)
- `tests/test_statistics.py` (신규 파일: 순위합 검정 단위 테스트)

## Exclusions

1. 문서 개정(item 3, §5.3 4건 + §4.4 한 줄, v0.7) — 범위 외, 비-SPEC manager-docs 태스크.
2. 순위합 검정의 Phase 3 분석 스크립트 배선 — 범위 외(Phase 3 미착수).
3. 기존 5개 무테스트 함수(bootstrap_accuracy_ci, run_cochran_q, run_mcnemar, run_wilcoxon, run_kruskal_wallis) 백필 — 범위 외.
4. 의료 모델 threshold 재보정 — 범위 외(0.7 informational-only).
5. Phase 1 결과 재계산 — 범위 외(REQ-EM-002가 보존 보장).
