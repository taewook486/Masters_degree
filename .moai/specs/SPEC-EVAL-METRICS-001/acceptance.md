# 인수 기준 (Acceptance Criteria): SPEC-EVAL-METRICS-001

각 인수 기준은 정확히 하나의 EARS 트리거(When / Where / If-then / While / Ubiquitous "항상")가 문장 전체를 지배하며, 여러 결과가 필요한 경우 동일 문장 안에서 "~하며"/"~고"로 결합한다(트리거 범위 밖의 별도 서술 문장을 덧붙이지 않는다). 관찰 가능 조건(결과 키 존재/부재, 값 범위, 인자 전달 등)은 그대로 검증 대상이다. 모든 BERTScore 시나리오는 BERTScore 백엔드를 mock하여 실모델 다운로드 없이 검증한다.

---

## 영역 1: BioBERT 이중 BERTScore

### AC-1-1 — 하위 호환 (Phase 1 보존) [REQ-EM-002]

**When** open-ended 항목이 포함된 상태에서 이중 BERTScore 평가를 지정하지 않고(기본값) overall-accuracy 계산이 호출되면, 시스템은 결과에 범용 primary 키(`open_bertscore_f1`, `open_bertscore_accuracy`)만 포함하고 의료/상관 키(`open_bertscore_f1_biobert`, `open_bertscore_spearman`, `open_bertscore_pearson`)는 포함하지 않으며, 기존 roberta-large 단일 동작을 유지해야 한다(shall).

### AC-1-2 — 이중 모델 계산 + 상관 [REQ-EM-001, REQ-EM-003]

**When** 호출자가 범용·의료 두 BERTScore 모델을 명시적으로 요청한 상태에서 overall-accuracy 계산이 실행되면(백엔드 mock이 모델별로 상이한 표본별 F1을 반환), 시스템은 결과에 `open_bertscore_f1`(범용), `open_bertscore_f1_biobert`(의료), 그리고 두 모델의 표본별 F1 벡터로부터 산출한 `open_bertscore_spearman`·`open_bertscore_pearson`을 모두 포함하고, `open_bertscore_spearman`을 상관의 주 보고값으로 제공해야 한다(shall).

### AC-1-3a — 의료 모델 레이어 수(=9) 전달 (엣지케이스) [REQ-EM-005]

**If** 의료 특화 모델(`dmis-lab/biobert-v1.1`)이 채점 대상으로 선택되면, **then** 시스템은 채점 백엔드를 `num_layers=9` 인자와 함께 호출하여 레이어 레지스트리 미등록으로 인한 `KeyError` 없이 채점을 완료해야 한다(shall). (백엔드 mock으로 호출 인자를 캡처하여 검증)

### AC-1-3b — 범용 모델 레이어 수 비재정의 [REQ-EM-005]

시스템은 **항상** 범용 모델(roberta-large) 채점 시 임베딩 레이어 수를 재정의하지 않아야 한다(shall). (레이어 레지스트리 기본값 사용, `num_layers` 미지정)

### AC-1-4 — primary 지표 불변 / 비-게이팅 [REQ-EM-004]

시스템은 **항상** correctness, `overall_accuracy`, `open_bertscore_accuracy`를 범용(roberta-large) 경로에만 근거하여 산출해야 하며(shall), 의료 모델 F1 값이 크게 달라지더라도 이들 primary 수치가 변경되지 않아야 한다(이중 게이팅 없음).

---

## 영역 2: Mann-Whitney U 검정

### AC-2-1 — 기본 양측 검정 반환 구조 [REQ-EM-006, REQ-EM-008]

**When** 두 run-level 정확도 표본 x(n=10), y(n=10)에 대해 2-독립표본 순위합 검정이 호출되면, 시스템은 `u_stat`, [0,1] 범위의 `p_value`, 두 그룹 크기 `n1=10`/`n2=10`, 효과 크기 `rank_biserial_r`, 그리고 `significant`(= `p_value < 0.05`의 bool)를 담은 결과를 반환해야 한다(shall).

### AC-2-2 — 동순위(tied ranks) 처리 (엣지케이스) [REQ-EM-009]

**If** 두 표본에 동일 값이 다수 포함되면(예: x=[0.5,0.5,0.5,...], y=[0.5,0.5,0.6,...]), **then** 시스템은 예외 없이 [0,1] 범위의 유효한 `p_value`를 반환해야 한다(shall). (정규근사 동점 보정)

### AC-2-3a — 효과 크기 극단값 (순방향) [REQ-EM-007]

**When** 첫 표본이 둘째 표본보다 확률적으로 큰 완전 분리 상태이면(x의 모든 값 > y의 모든 값), 시스템은 양의 극단(+1 근처)의 `rank_biserial_r`와 `significant=True`를 반환해야 한다(shall).

### AC-2-3b — 효과 크기 부호 반전 (인자 반전) [REQ-EM-007]

**When** 동일 표본을 인자 순서를 반전하여(y, x) 호출하면, 시스템은 부호가 반전된(음수) `rank_biserial_r`를 반환해야 한다(shall).

### AC-2-4 — 스키마 일관성 [REQ-EM-008]

시스템은 **항상** `significant` 플래그를 형제 순위검정 함수(`run_wilcoxon`, `run_kruskal_wallis`)와 동일한 규약, 즉 `p_value < 0.05`의 bool로 보고해야 한다(shall).

---

## 품질 게이트 / Definition of Done

- [ ] REQ-EM-001~009 각각에 대응하는 테스트 통과 (AC-1-1, AC-1-2, AC-1-3a, AC-1-3b, AC-1-4, AC-2-1, AC-2-2, AC-2-3a, AC-2-3b, AC-2-4)
- [ ] `tests/test_metrics.py` 이중 모델 테스트가 실모델 다운로드 없이 mock으로 통과
- [ ] `tests/test_statistics.py` 신규 생성, 순위합 검정 테스트 통과 (scipy 실호출)
- [ ] 기존 `tests/test_metrics.py` 14개 테스트 회귀 없음(하위 호환 보장)
- [ ] `ruff check` (E, F, I, W) 통과, line-length 88
- [ ] `mypy` 타입 체크 통과(신규 시그니처 타입 힌트 포함)
- [ ] Phase 1 결과 스키마 불변 확인(이중 평가 미요청 경로 회귀 테스트)
