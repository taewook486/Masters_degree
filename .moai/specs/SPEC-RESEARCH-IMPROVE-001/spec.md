---
id: SPEC-RESEARCH-IMPROVE-001
version: 1.0.0
status: completed
created: 2026-05-07
updated: 2026-06-30
author: manager-spec
priority: High
issue_number: null
---

# SPEC-RESEARCH-IMPROVE-001: 석사 연구 코드베이스 3대 영역 개선

## HISTORY

- 2026-05-07: 초기 SPEC 생성. SPEC-IMPROVE-001(전체 개선 계획)의 후속으로, Phase 1 파이프라인/autoresearch/데이터 파이프라인 3개 영역에 집중.

---

## 개요

### 배경

석사 연구 프로젝트(Medical VQA VLM)는 v0.2 설계서 요구사항 구현(BERTScore, CF 측정, max_steps, GPU time 분리)과 autoresearch 에이전트 개선(온도 스케줄링, 재시도 로직, 구조화 프롬프트)을 최근 커밋으로 반영하였다. 그러나 코드 분석 결과 다음 3개 영역에서 추가 개선이 필요하다:

1. **Phase 1 실험 파이프라인**: BERTScore 기본 비활성화, 결과 집계 불완전, 환경 정보 미기록
2. **Autoresearch 에이전트**: 중복 설정 실패 방지, 에이전트 응답 검증 강화, 탐색/활용 전략 고도화
3. **데이터 처리/저장**: 실험 결과 저장 형식 비일관, 로깅 비구조화, 체크포인트 관리 미흡

### 목표

연구 재현성, 실험 신뢰도, 데이터 추적성을 강화하여 논문 작성 및 심사 대응에 필요한 증거 품질을 확보한다.

### SPEC-IMPROVE-001과의 관계

SPEC-IMPROVE-001은 12개 REQ 전체를 포괄하는 마스터 SPEC이다. 본 SPEC은 그 중 REQ-005(Phase 1 집계), REQ-007(BERTScore), REQ-008(로깅), REQ-012(환경 정보)와 직접 관련되며, autoresearch 에이전트와 데이터 파이프라인에 대한 신규 요구사항을 추가한다.

---

## 영향 범위

### 수정 대상 파일

| 모듈 | 파일 | 변경 유형 |
|------|------|-----------|
| Phase 1 파이프라인 | `src/baseline/evaluate_zero_shot.py` | 수정 (BERTScore 활성화, 환경 정보) |
| Phase 1 파이프라인 | `src/baseline/run_all.py` | 수정 (집계 로직, BERTScore 컬럼) |
| Phase 1 파이프라인 | `src/evaluate/metrics.py` | 수정 (BERTScore 기본값 변경) |
| Autoresearch | `src/autoresearch/agent.py` | 수정 (응답 검증, 중복 방지) |
| Autoresearch | `src/autoresearch/strategies.py` | 수정 (중복 설정 감지) |
| Autoresearch | `src/autoresearch/loop.py` | 수정 (체크포인트, 재개 로직) |
| Autoresearch | `src/autoresearch/tracker.py` | 수정 (결과 저장 형식 통일) |
| 데이터 파이프라인 | `src/utils/logging_config.py` | 신규 (구조화 로깅) |
| 데이터 파이프라인 | `src/utils/environment.py` | 신규 (환경 정보 수집) |
| 데이터 파이프라인 | `src/utils/checkpoint.py` | 신규 (체크포인트 관리) |

---

## 요구사항 (EARS Format)

### 영역 1: Phase 1 실험 파이프라인

#### REQ-RI-001: BERTScore 기본 활성화

**유형**: Ubiquitous
**EARS**: 시스템은 **항상** Phase 1 및 Phase 2의 open-ended 질문 평가 시 BERTScore F1을 계산하여 결과 JSON의 `summary` 필드에 포함해야 한다(shall).

**근거**: `compute_overall_accuracy()`의 기본값이 `compute_bertscore=False`이나, v0.2 설계서에서 BERTScore를 주요 평가 지표로 지정. 현재 Phase 1 결과에 BERTScore가 포함되지 않아 논문 Table 작성 불가.

**영향 파일**:
- `src/baseline/evaluate_zero_shot.py` (93번째 줄: `compute_overall_accuracy` 호출에 `compute_bertscore=True` 추가)
- `src/evaluate/metrics.py` (163번째 줄: `compute_bertscore` 기본값을 `True`로 변경)
- `src/baseline/run_all.py` (`_aggregate_seed_results`에 BERTScore 관련 키 추가)

---

#### REQ-RI-002: Phase 1 결과 집계 완전성

**유형**: **When** Phase 1 베이스라인 평가가 완료되면, 시스템은 모든 시드의 결과를 올바르게 집계하여 mean +/- std를 산출해야 한다(shall).

**근거**: `phase1_summary.csv`에서 `num_seeds=1`로 기록되거나, std 값이 0.0인 경우가 발견됨. `_aggregate_seed_results()`는 올바르게 구현되어 있으나, `skip_existing` 로직에서 기존 결과를 로드할 때 BERTScore 키가 누락되면 집계 오류 발생 가능.

**영향 파일**:
- `src/baseline/run_all.py` (`_load_existing_result`에서 BERTScore 키 포함 검증, `_aggregate_seed_results`에 BERTScore 집계 추가)

---

#### REQ-RI-003: 실험 환경 정보 자동 기록

**유형**: **When** 실험이 시작되면, 시스템은 실험 환경 정보(Python 버전, torch 버전, CUDA 버전, GPU 이름, GPU 메모리, OS, transformers 버전, peft 버전)를 결과 JSON의 `metadata.environment` 필드에 자동 기록해야 한다(shall).

**근거**: 현재 결과 JSON의 `metadata`에 환경 정보가 없음. 논문 재현성 요건 및 리뷰어 요청 대응에 필수.

**영향 파일**:
- `src/utils/environment.py` (신규: `get_environment_info() -> dict`)
- `src/baseline/evaluate_zero_shot.py` (결과 JSON의 `metadata`에 환경 정보 추가)

---

### 영역 2: Autoresearch 에이전트

#### REQ-RI-004: 중복 설정 감지 및 방지

**유형**: **While** HPO 루프가 실행 중이면, 시스템은 이전에 완료된 trial과 동일한 하이퍼파라미터 설정이 제안되었을 때 이를 감지하고 재제안을 요청해야 한다(shall).

**근거**: `AutoresearchStrategy.suggest()`가 에이전트의 응답을 그대로 사용하므로, 이미 시도한 동일 설정이 반복될 수 있음. 40회 trial 예산에서 중복은 탐색 효율을 저하시킴.

**영향 파일**:
- `src/autoresearch/strategies.py` (`AutoresearchStrategy.suggest`에 중복 검사 로직 추가)
- `src/autoresearch/agent.py` (중복 시 재시도 메시지 구성)

---

#### REQ-RI-005: 에이전트 응답 검증 강화

**유형**: **If** 에이전트의 JSON 응답에 필수 키가 누락되거나 유효 범위를 벗어나는 값이 포함되면, **then** 시스템은 누락 키에 대해 기본값을 적용하고 경고 로그를 기록해야 한다(shall).

**근거**: `_validate_config()`가 범위 클램핑을 수행하지만, 필수 키 누락 시 `KeyError` 발생 가능성 존재. `epochs` 키가 여전히 `_VALID_EPOCHS`에 정의되어 있으나 v0.2에서는 `max_steps`로 전환되었으므로 하위 호환성 처리 필요.

**영향 파일**:
- `src/autoresearch/agent.py` (`_validate_config`에 필수 키 존재 여부 확인, `epochs`->`max_steps` 마이그레이션 로직)

---

#### REQ-RI-006: 탐색/활용 전환 전략 로깅

**유형**: **When** autoresearch 전략이 다음 trial을 제안하면, 시스템은 현재 탐색/활용(exploration/exploitation) 단계, 온도 값, 에이전트 추론 요약을 구조화된 로그로 기록해야 한다(shall).

**근거**: 온도 스케줄링(`temperature = 1.0 - 0.7 * progress`)이 구현되어 있으나, 탐색-활용 전환 과정을 추적할 수 있는 구조화된 로그가 없음. 논문에서 autoresearch 전략의 행동 분석에 필요.

**영향 파일**:
- `src/autoresearch/agent.py` (구조화 로그 추가)
- `src/autoresearch/tracker.py` (trial 메타데이터에 phase/temperature 필드 추가)

---

### 영역 3: 데이터 처리/저장

#### REQ-RI-007: 구조화된 로깅 시스템

**유형**: Ubiquitous
**EARS**: 시스템은 **항상** 일관된 구조화 로깅 포맷을 사용하고, 로그를 파일과 콘솔에 동시 출력해야 한다(shall).

**근거**: 현재 각 모듈의 `main()`에서 독립적으로 `logging.basicConfig()` 호출. `run_all.py:254`, `evaluate_zero_shot.py:304`에서 중복 설정. 장시간 GPU 실험(Phase 1: ~6시간, Phase 3: ~10시간)에서 로그 파일 보존이 필수.

**영향 파일**:
- `src/utils/logging_config.py` (신규: `setup_logging(log_dir, experiment_name)`)
- `src/baseline/run_all.py`, `src/baseline/evaluate_zero_shot.py`, `src/autoresearch/loop.py` (기존 `logging.basicConfig()` 호출을 `setup_logging()` 호출로 교체)

---

#### REQ-RI-008: HPO 루프 체크포인트 및 재개

**유형**: **If** HPO 루프가 예기치 않게 중단되면(OOM, 전원 차단, 프로세스 종료), **then** 시스템은 마지막으로 완료된 trial 이후부터 루프를 재개할 수 있어야 한다(shall).

**근거**: `ExperimentTracker`가 TSV에 trial 결과를 append 하므로 데이터 손실은 없으나, `run_hpo_loop()`가 항상 trial 0부터 시작하여 이미 완료된 trial을 다시 실행함. 40trial x 15분 = 10시간 실험에서 중단 후 재개 기능 필수.

**영향 파일**:
- `src/autoresearch/loop.py` (`run_hpo_loop`에 기존 trial 수 확인 후 이어서 실행하는 로직)
- `src/utils/checkpoint.py` (신규: 루프 상태 저장/복원)

---

#### REQ-RI-009: 결과 저장 형식 통일

**유형**: Ubiquitous
**EARS**: 시스템은 **항상** 모든 Phase의 실험 결과를 동일한 스키마(metadata, summary, per_sample)로 저장해야 한다(shall).

**근거**: Phase 1 결과는 `{model}_{dataset}_seed{N}.json` 형식으로 `metadata/summary/per_sample` 구조를 따르나, Phase 3 결과(`tracker.py`)는 TSV 플랫 형식. CF 측정 결과(`catastrophic_forgetting.py`)는 또 다른 구조. 결과 간 비교/병합 시 파싱 로직 중복 발생.

**영향 파일**:
- `src/autoresearch/tracker.py` (JSON 내보내기 메서드 추가)
- `src/evaluate/catastrophic_forgetting.py` (출력 스키마를 metadata/summary 구조로 통일)

---

## Exclusions (What NOT to Build)

1. **새로운 모델 추가 또는 모델 아키텍처 변경** - 기존 3개 모델(Qwen2.5-VL-3B, Qwen3-VL-2B, SmolVLM-2.2B) 유지
2. **데이터셋 전처리 파이프라인 재설계** - 기존 데이터 로더(`load_medical_vqa_dataset`) 인터페이스 유지
3. **CI/CD 파이프라인 또는 자동 배포** - 로컬 실행 환경 전제
4. **웹 UI 또는 실험 대시보드** - CLI/스크립트 기반 인터페이스 유지
5. **통계 분석 모듈 구현** - SPEC-IMPROVE-001 REQ-004에서 별도 처리
6. **시각화 모듈 구현** - SPEC-IMPROVE-001 REQ-009에서 별도 처리
7. **단위 테스트 추가** - SPEC-IMPROVE-001 REQ-001/010/011에서 별도 처리
8. **분산 학습 지원** - 단일 GPU(RTX 4090) 환경 전제
