---
id: SPEC-EVAL-METRICS-001
version: 0.1.1
status: in-progress
created_at: 2026-07-14
updated: 2026-07-14
author: manager-spec
priority: high
issue_number: 0
labels: [evaluation, bertscore, statistics, phase2, phase3]
---

# SPEC-EVAL-METRICS-001: BioBERT 이중 BERTScore + Mann-Whitney U 검정 (평가 코드)

## HISTORY

- 2026-07-14: 초기 SPEC 생성. 동료 심사(`docs/비판적_동료심사_v0.5.md`) 및 설계서(`docs/THESIS_PROPOSAL_FINAL_v0.6.md`) 대조 결과 발견된 설계-구현 gap 중 **코드 산출물 2건**(Phase 2 BioBERT 이중 BERTScore, Phase 3 Mann-Whitney U 검정)을 대상으로 함. 문서 개정(item 3: §5.3 한계점 재작성)은 본 SPEC 범위 외(비-SPEC manager-docs 태스크로 별도 추적). 검토자 확정 결정 반영: 상관계수 = Spearman 주 + Pearson 병기, BioBERT 레이어 수 고정, rank-biserial 부호 규약 명시.
- 2026-07-14: plan-audit iteration 1 지적사항 반영(v0.1.0 → v0.1.1). MP-2(acceptance.md EARS 표현화), MP-3(frontmatter `created_at`/`labels` 추가, priority 소문자화), RQ-4(REQ 본문에서 코드 식별자·라이브러리 버전·함수 시그니처·dict 키를 제거하고 `plan.md`로 이관). 요구사항의 관찰 가능 조건(테스트 내용)은 불변.
- 2026-07-14: iteration 3 감사에서 지적된 REQ-EM-004의 미트리거 후행 문장을 단일 EARS 트리거 문장으로 병합 (사용자 승인 하에 4차 감사 생략).

---

## 개요

### 배경

설계서 `THESIS_PROPOSAL_FINAL_v0.6.md`는 두 가지를 요구하나 코드베이스에 미구현 상태이다.

1. **Phase 2 BioBERT 이중 BERTScore** (§4.4, L171-174): open-ended 정확도를 범용 모델(roberta-large, threshold >= 0.7) F1과 의료 특화 모델(BioBERT, `dmis-lab/biobert-v1.1`) F1로 **이중 보고**하고 두 지표의 상관관계를 분석. 설계서는 이 요구를 **Phase 2로만 한정**(Phase 1 제외).
   - Phase 1/2가 동일한 평가 진입점을 공유하므로, 의료 모델 로딩은 반드시 **opt-in**이어야 한다(무조건 로딩 시 Phase 1 회귀 및 VRAM/시간 비용 위험 — `THESIS_CHANGELOG.md` v0.2 "BERTScore 모델 VRAM 충돌 가능성" 리스크).

2. **Phase 3 Mann-Whitney U 검정** (§4.5): run-level 비교에서 "Autoresearch vs Optuna 쌍별 비교 (n=10 vs n=10)"를 Kruskal-Wallis와 병행. `THESIS_REVISION_REPORT.md`에도 구현된 기능으로 기재되어 있으나 **repo 전체에 구현 0건**(doc/shell 언급만 존재).

### 목표

논문 설계서가 명시한 두 평가 산출물을 코드로 구현하여 설계-구현 정합성을 확보하고, Phase 2/3 결과 표·통계 검정의 증거 품질을 보장한다. 기존 Phase 1 동작 및 고정된 baseline 수치(`results/phase1_baseline/phase1_summary.csv`)는 회귀 없이 보존한다.

### 관련 SPEC / 태스크

- `SPEC-RESEARCH-IMPROVE-001`(완료): Phase 1 파이프라인의 BERTScore 기본 활성화 등을 포함. 본 SPEC은 그 위에 **Phase 2 한정 이중 모델**을 opt-in으로 추가.
- **item 3 (문서 개정)**: 본 SPEC 범위 외. `docs/THESIS_PROPOSAL_FINAL_v0.7.md` 신규 생성 + §5.3 한계점 4건 + §4.4 한 줄(본 SPEC의 primary 규칙 명시)을 비-SPEC manager-docs 태스크로 별도 처리. 문서 태스크는 본 SPEC의 primary 규칙(REQ-EM-004) 확정 이후 실행되는 의존 관계.

---

## 영향 범위

파일 단위 변경 범위(WHAT/WHERE). 함수 시그니처·파라미터명·반환 스키마·라이브러리 호출·고정 레이어 수 값 등 **구현 수준 명세는 본 SPEC이 아니라 `plan.md`에 정의**한다.

| 모듈 | 파일 | 변경 유형 | 행위 요약 |
|------|------|-----------|-----------|
| 평가 지표 | `src/evaluate/metrics.py` | 수정 | 요청 시 모델별 BERTScore F1 이중 계산 + 상관 산출, 의료 모델의 레이어 수 안전장치 |
| Phase 1 경로 | `src/baseline/evaluate_zero_shot.py` | 수정 | 이중 평가 요청을 선택적으로 전달(미지정 시 기존 동작 유지) |
| Phase 2 경로 | `src/finetune/train_qlora.py` | 수정 | Phase 2 평가에서 이중 모델 평가를 명시적으로 활성화(opt-in) |
| 통계 검정 | `src/evaluate/statistics.py` | 신규 함수 | 2-독립표본 양측 순위합 검정 기능 추가 |
| 테스트 | `tests/test_metrics.py` | 테스트 추가 | 이중 모델 / 상관 / 레이어 수 전달 검증(BERTScore 백엔드 mock) |
| 테스트 | `tests/test_statistics.py` | 신규 파일 | 순위합 검정 단위 테스트 |

> 상세 구현 명세(함수/파라미터명, 반환 dict 키, 라이브러리·호출 시그니처, 고정 임베딩 레이어 수 값, 결과 키 스키마)는 `plan.md`의 "수정/신규 대상 파일" 및 설계 섹션을 참조한다.

---

## 요구사항 (EARS Format)

### 영역 1: BioBERT 이중 BERTScore (Phase 2 opt-in)

#### REQ-EM-001: 이중 모델 opt-in 계산

**유형**: Optional
**EARS**: **Where** 호출자가 이중 BERTScore 평가를 명시적으로 요청하는 경우(Phase 2 평가 경로), 시스템은 요청된 각 모델(범용 모델 roberta-large 및 의료 특화 모델 BioBERT)에 대해 open-ended F1을 각각 계산해야 한다(shall).

**근거**: 설계서 §4.4가 이중 보고를 Phase 2로 한정. 범용/의료 모델 각각의 F1을 산출하는 경로 신설.

---

#### REQ-EM-002: 기본값 하위 호환 (Phase 1 보존)

**유형**: Ubiquitous
**EARS**: 시스템은 **항상** 이중 BERTScore 평가가 명시적으로 요청되지 않으면 범용 모델(roberta-large) 단일 채점만 수행하여 기존 Phase 1 결과 스키마와 동작을 보존해야 한다(shall).

**근거**: 평가 진입점이 Phase 1/2 공통. Phase 1 baseline 수치는 `results/phase1_baseline/phase1_summary.csv`에 고정. 무조건적 이중 로딩은 회귀·VRAM·시간 위험.

---

#### REQ-EM-003: 이중 지표 상관관계 (Spearman 주 + Pearson 병기)

**유형**: Event-Driven
**EARS**: **When** 이중 BERTScore 모델이 계산되면, 시스템은 두 모델의 표본별 F1 점수 벡터 간 **Spearman 순위 상관계수(주 보고값)** 와 **Pearson 상관계수(병기)** 를 산출하여 결과에 포함해야 한다(shall).

**근거**: BERTScore F1 분포는 비정규·경계값 편중 가능. "두 지표가 답변을 유사하게 순위화하는가"는 단조 일치 문제이므로 Spearman이 견고. 검토자 확정: Spearman headline, Pearson 병기.

---

#### REQ-EM-004: primary 지표 불변 및 비-게이팅 (동료심사 #7 해소)

**유형**: Ubiquitous
**EARS**: 시스템은 **항상** roberta-large @ threshold 0.7을 정확도·통계 검정의 primary(결정) 지표로 사용하고, BioBERT F1은 보조(secondary) 지표로만 보고하며, 정오 판정에 이중 게이팅(dual-gating)을 수행하지 않아야 한다(shall).

**근거**: 하류 정확도 수치·통계 검정의 유일 기준을 범용 모델로 유지하여 선행 연구 비교 가능성 보존. 의료 모델은 의료 용어 민감도 관점의 supplementary 지표.

---

#### REQ-EM-005: 의료 모델 레이어 수 안전장치

**유형**: Unwanted Behavior
**EARS**: **If** 의료 특화 BERTScore 모델이 표준 레이어 자동 결정에 실패할 수 있는 경우, **then** 시스템은 문서화된 고정 임베딩 레이어 수를 채점 루틴에 명시적으로 제공하여 채점 실패를 방지해야 한다(shall).

**근거**: 의료 모델은 BERTScore 표준 레이어 자동결정 대상에 포함되지 않을 가능성이 높아, 명시적 레이어 수 미지정 시 채점이 실패할 수 있음. 검토자 확정: 문서화된 고정 레이어 수 사용(구체 값·산정 근거는 `plan.md` 참조).

---

### 영역 2: Mann-Whitney U 검정 (Phase 3 scope, 함수+테스트만)

#### REQ-EM-006: 2-독립표본 순위합 검정 제공

**유형**: Ubiquitous
**EARS**: 시스템은 **항상** 두 독립 표본(Autoresearch run-level vs Optuna run-level, n=10 vs n=10 사용 예상)에 대한 양측 2-독립표본 순위합 검정(Mann-Whitney U, 비모수) 기능을 제공해야 한다(shall).

**근거**: 설계서 §4.5 요구. repo 전체에 구현 부재. 기존 통계 검정 함수군과 동일한 형식으로 제공.

---

#### REQ-EM-007: 효과 크기(rank-biserial) + 부호 규약

**유형**: Ubiquitous
**EARS**: 시스템은 **항상** 순위합 검정 결과와 함께 효과 크기(rank-biserial 상관)를 보고해야 하며, 효과 크기의 부호는 **첫 표본이 둘째 표본보다 큰 경향일 때 양수**가 되도록 정의하고 문서화해야 한다(shall).

**근거**: 기존 통계 모듈의 확립된 관행 — p-value에 항상 효과 크기 병기. 검토자 확정: 부호 방향성 명시. 효과 크기 산식은 `plan.md` 참조.

---

#### REQ-EM-008: 유의성 판정 규약 일관성

**유형**: Ubiquitous
**EARS**: 시스템은 **항상** 순위합 검정 결과의 유의성 판정을 기존 형제 통계 함수들과 동일한 규약(유의수준 0.05 기준 boolean)으로 보고해야 한다(shall).

**근거**: 함수 간 일관성으로 하류 분석 스크립트의 파싱 로직 통일.

---

#### REQ-EM-009: 동순위(tied ranks) 처리

**유형**: Unwanted Behavior
**EARS**: **If** 두 표본에 동순위(tied ranks)가 존재하면, **then** 순위합 검정은 표준 정규근사 기반 동점 보정을 사용하여 예외 없이 유효한 p-value(∈ [0,1])를 반환해야 한다(shall).

**근거**: run-level accuracy 값은 반올림으로 동점 발생 가능. 표준 정규근사(동점 보정)를 사용.

---

## 확정된 설계 결정 (검토자 승인)

정규 요구사항이 아니라 검토 과정에서 확정된 결정 기록이다. 구체 구현 값은 `plan.md`에 있다.

| 지점 | 확정 결정 |
|------|-----------|
| C 상관계수 | Spearman을 주 상관 보고값으로 사용, Pearson 병기 (REQ-EM-003) |
| D 의료 모델 레이어 | 의료 모델 채점 시 문서화된 고정 임베딩 레이어 수 사용 (구체 값·산정 근거는 `plan.md`) (REQ-EM-005) |
| F 효과 크기 부호 | rank-biserial 상관: 양수 = 첫 표본이 둘째 표본보다 큰 경향 (문서화) (REQ-EM-007) |
| primary 규칙 | 범용 모델 @ 0.7이 유일 결정 지표, 의료 모델은 secondary, 비-게이팅 (REQ-EM-004) |

---

## Exclusions (What NOT to Build)

1. **문서 개정 (item 3)** — `docs/THESIS_PROPOSAL_FINAL_v0.7.md` §5.3 한계점 재작성(WCA 임시 가중치, 다중비교 보정 부재, cross-dataset CF 개념 오용, Gemma4-E2B MoE 공정성 4건) 및 §4.4 한 줄 명시는 **본 SPEC 범위 외**. 비-SPEC manager-docs 태스크로 별도 추적하며, 본 SPEC의 REQ-EM-004 확정 이후 실행된다.
2. **Mann-Whitney의 Phase 3 분석 스크립트 통합** — 순위합 검정 함수를 미래 `analyze_phase3.py` 등 실행기에 배선하는 작업은 **범위 외**. Phase 3는 아직 미착수 상태이며(`scripts/analyze_phase1.py`가 Phase 1 완료 후에야 추가된 패턴과 동일), 본 SPEC은 함수 + 단위 테스트만 다룬다.
3. **기존 무테스트 통계 함수 백필** — `src/evaluate/statistics.py`의 기존 5개 함수(`bootstrap_accuracy_ci`, `run_cochran_q`, `run_mcnemar`, `run_wilcoxon`, `run_kruskal_wallis`)에 대한 테스트 추가는 **범위 외**(scope-creep 방지). 본 SPEC은 신규 순위합 검정 함수만 테스트한다.
4. **BioBERT threshold 재보정** — 의료 모델 F1의 정확도 임계값을 범용 모델의 0.7과 다르게 재calibration하는 연구는 범위 외. 의료 모델 accuracy는 동일 0.7 임계값에서 informational-only로 보고하며 재보정하지 않는다.
5. **Phase 1 결과 재계산** — 고정된 `results/phase1_baseline/` 수치는 재계산·재채점하지 않는다(REQ-EM-002가 이를 보장).
