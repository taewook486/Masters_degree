# SPEC-IMPROVE-001: 석사 연구 프로젝트 전체 개선 계획

## 메타데이터

| 항목 | 값 |
|------|-----|
| SPEC ID | SPEC-IMPROVE-001 |
| 제목 | Medical VQA VLM 프로젝트 코드 품질 및 실험 파이프라인 개선 |
| 상태 | completed |
| 우선순위 | High |
| 생성일 | 2026-04-28 |
| 완료일 | 2026-07-01 |
| 구현 범위 | 10/12 REQ 완료, 1/12 대체 경로로 해결 (REQ-005 — 원 인수 조건은 미충족이나 Internal v0.6 방법론 수정으로 근본 문제 해결), 1/12 스킵 (REQ-006 — 선택사항, 미구현) |

---

## 개요

석사 연구 프로젝트(Medical VQA VLM)의 코드베이스 심층 분석 결과, 코드 품질, 실험 파이프라인 자동화, 연구 방법론 강화, 프로젝트 관리 개선이 필요한 12개 구체적 영역을 식별하였다. 본 SPEC은 Phase 1 베이스라인 완료 후 ~ Phase 2 진입 전 시점에 적용할 개선 사항을 정의한다.

---

## 현재 상태 분석

### 강점
- 모듈형 3단계 파이프라인 구조 (baseline/finetune/autoresearch)가 잘 분리됨
- YAML 기반 설정 관리로 실험 재현성 확보
- 이중 백엔드 (Unsloth/HF PEFT) 자동 선택 구현
- OOM 폴백 처리, 배치 추론 최적화 구현
- ExperimentTracker TSV 기반 결과 추적 구현

### 발견된 문제

1. **테스트 부재**: `tests/florence2_step_test.py` 1개만 존재하며, pytest 형식이 아닌 수동 스크립트. 핵심 모듈(metrics, data loader, seed)에 단위 테스트 없음
2. **phase1_summary.csv 데이터 불일치**: `num_seeds=1`로 기록되어 있으나 실제 3개 시드 결과 파일 존재 (집계 로직 버그 또는 재실행 필요)
3. **MEDICAL_PROMPT 중복 정의**: `model_loader.py`와 `prepare_data.py`에 동일한 프롬프트 문자열 중복
4. **통계 검증 코드 미구현**: 설계서에 명시된 ANOVA, Tukey HSD, Paired t-test 등의 통계 분석 코드 미작성
5. **run_phase1.bat 경직성**: Windows .bat 파일에 절대 경로 하드코딩, Python 스크립트 대비 유연성 부족
6. **로깅 비일관성**: 각 모듈이 독립적으로 `logging.basicConfig()` 호출, 구조화된 로깅 미적용
7. **catastrophic_forgetting.py 인터페이스 불일치**: `evaluate_on_vqav2()`가 `generate_answer()`에 `prompt` 키워드 인자 전달하나, 실제 함수 시그니처에는 `question` 파라미터 사용
8. **BERTScore 기본 미활성화**: `compute_overall_accuracy()`에서 `compute_bertscore=False`가 기본값이나, 설계서 v0.2에서는 BERTScore를 주요 지표로 요구
9. **SmolVLM 시드 결과 불완전**: SmolVLM-2.2B의 slake/vqa_rad 데이터셋에 seed 123, 456 결과 파일 누락

---

## 요구사항 (EARS Format)

### REQ-001: [코드 품질] 핵심 모듈 단위 테스트 추가

**유형**: Ubiquitous
**EARS 서술**: 시스템은 **항상** 핵심 평가 함수(`preprocess_answer`, `_extract_yes_no`, `compute_closed_accuracy`, `compute_open_accuracy`, `compute_overall_accuracy`)에 대해 pytest 기반 단위 테스트를 갖추어야 한다.

**근거**: 현재 `tests/` 디렉토리에 pytest 형식 테스트가 0개. 메트릭 계산 로직은 논문 결과의 정확성에 직결되므로 테스트 필수.

**인수 조건**:
- [x] `tests/test_metrics.py` 파일 생성
- [x] `preprocess_answer`에 대해 최소 8개 테스트 케이스 (공백, 대소문자, 구두점, 빈 문자열 등)
- [x] `_extract_yes_no`에 대해 최소 10개 테스트 케이스 (yes/no 변형, 문장 시작, 비해당 텍스트)
- [x] `compute_closed_accuracy`, `compute_open_accuracy`에 대해 각 5개 이상 테스트 케이스
- [x] `compute_overall_accuracy`에 대해 3개 이상 통합 테스트 케이스
- [x] `pytest tests/test_metrics.py` 실행 시 100% 통과
- [x] 커버리지 85% 이상 (`src/evaluate/metrics.py` 기준)

---

### REQ-002: [코드 품질] MEDICAL_PROMPT 상수 중복 제거

**유형**: Ubiquitous
**EARS 서술**: 시스템은 **항상** 의료 VQA 프롬프트 템플릿을 단일 소스(single source of truth)에서 관리해야 한다.

**근거**: `src/baseline/model_loader.py:17`과 `src/finetune/prepare_data.py:21`에 동일한 `MEDICAL_PROMPT` 문자열이 중복 정의됨. 프롬프트 수정 시 양쪽 동기화 실패 위험.

**인수 조건**:
- [x] `src/utils/constants.py` 또는 `src/utils/prompts.py`에 `MEDICAL_PROMPT` 단일 정의
- [x] `model_loader.py`와 `prepare_data.py`에서 해당 모듈 import
- [x] 기존 동작과 결과에 변경 없음 확인

---

### REQ-003: [코드 품질] catastrophic_forgetting.py 인터페이스 버그 수정

**유형**: IF [비정상 상태] THEN [시스템 대응]
**EARS 서술**: **IF** `evaluate_on_vqav2()` 함수가 `generate_answer()`를 호출할 때 잘못된 키워드 인자를 전달하면, **THEN** 런타임 에러가 발생하므로 인터페이스를 수정해야 한다.

**근거**: `catastrophic_forgetting.py:70`에서 `generate_answer(..., prompt=prompt)`로 호출하나, `model_loader.py`의 `generate_answer()` 시그니처는 `(model, processor, image, question, config)`. `prompt` 키워드가 존재하지 않아 Phase 2에서 CF 측정 시 `TypeError` 발생 예상.

**인수 조건**:
- [x] `evaluate_on_vqav2()`의 `generate_answer()` 호출을 올바른 시그니처로 수정
- [x] CF 측정이 정상 동작하는 것을 검증하는 테스트 케이스 추가
- [x] `question` 파라미터로 전달하도록 수정

---

### REQ-004: [실험 파이프라인] 통계 분석 모듈 구현

**유형**: WHEN [이벤트] THEN [동작]
**EARS 서술**: **WHEN** Phase 1 베이스라인 평가가 완료되면, **THEN** 시스템은 설계서에 명시된 통계 검증(ANOVA, Tukey HSD, Cohen's d, Paired t-test, Wilcoxon signed-rank)을 자동 수행해야 한다.

**근거**: 설계서 Section 4.3-4.5에 통계 검증 방법이 명시되어 있으나, 이를 수행하는 코드가 전혀 없음. 수동으로 수행하면 재현성 저하.

**인수 조건**:
- [x] `src/evaluate/statistics.py` 모듈 생성
- [x] Phase 1용: `run_anova_models()` - 3개 모델 간 성능 차이 ANOVA + Tukey HSD
- [x] Phase 2용: `run_paired_ttest()` - Base vs Fine-tuned Paired t-test + Cohen's d
- [x] Phase 2용: `run_wilcoxon()` - 비모수 Wilcoxon signed-rank test
- [x] Phase 3용: `run_kruskal_wallis()` - 4개 HPO 전략 간 비교
- [x] 모든 함수가 p-value, 검정 통계량, effect size를 dict로 반환
- [x] `scipy.stats` 활용, 유의수준 alpha=0.05 기본값

---

### REQ-005: [실험 파이프라인] Phase 1 결과 집계 버그 수정

**유형**: IF [비정상 상태] THEN [시스템 대응]
**EARS 서술**: **IF** `phase1_summary.csv`에 `num_seeds=1`로 기록되어 있으나 실제 3개 시드 결과 파일이 존재하면, **THEN** 시스템은 모든 시드의 결과를 올바르게 집계하여 mean +/- std를 산출해야 한다.

**근거**: `results/phase1_baseline/phase1_summary.csv`에서 모든 조건의 `closed_acc_std`, `open_acc_std` 등이 0.0으로 기록됨. 3개 시드에 대한 표준편차가 0인 것은 집계 미완료를 의미.

**인수 조건**:
- [ ] `run_all.py`의 시드 집계 로직 검증 및 수정
- [ ] 재실행 후 `phase1_summary.csv`에 `num_seeds=3`, 비-제로 std 값 확인
- [ ] SmolVLM-2.2B의 누락된 시드 결과(slake seed123/456, vqa_rad seed123/456, pathvqa seed456) 완성

**2026-07-14 재검증 주석**: 위 인수 조건은 문자 그대로는 충족되지 않았다 (`num_seeds=3` 재실행은 수행되지 않음). 그러나 이 REQ가 다루려던 근본 문제 — Phase 1 결과 신뢰도 확보 — 는 다른 경로로 해결되었다. Internal v0.6 방법론 수정(`docs/THESIS_CHANGELOG.md` 참조, 2026-07-11, 동료심사 `docs/비판적_동료심사_v0.5.md` 항목 #2 "Phase 1 STD=0.0 데이터 무결성"에 대한 대응으로 추정)에서, zero-shot 평가는 greedy decoding으로 결정론적이므로 3-시드 분산 자체가 수학적으로 불가능하다는 점이 확인되었다. 즉 원 인수 조건(`num_seeds=3`, 비-제로 std)은 애초에 성립할 수 없는 전제(시드 간 복원 가능한 분산이 존재한다는 가정)에 기반한 것이었다. 실제 적용된 수정은 단일 시드(42) + bootstrap 95% CI를 통한 불확실성 정량화이며, `results/phase1_baseline/phase1_summary.csv`의 `num_seeds=1` + `closed_acc_ci_low`/`closed_acc_ci_high` 컬럼과 `src/baseline/run_all.py`의 집계 로직에서 직접 확인됨.

---

### REQ-006: [실험 파이프라인] .bat 스크립트를 Python CLI로 전환

**유형**: Ubiquitous
**EARS 서술**: 시스템은 **항상** 실험 실행 진입점을 크로스 플랫폼 Python 스크립트로 제공해야 한다.

**근거**: `run_phase1.bat`에 절대 경로(`D:\project\Masters_degree`) 하드코딩. 환경 이동 시 수정 필요. Python CLI(`python -m src.baseline.run_all`)는 이미 존재하므로, 래퍼 스크립트를 Python으로 통합 가능.

**인수 조건**:
- [ ] `run_experiment.py` 통합 CLI 스크립트 생성 (argparse 기반)
- [ ] `python run_experiment.py phase1 --seeds 42 123 456` 형식 지원
- [ ] `python run_experiment.py phase2 --model qwen25_vl_3b --dataset pathvqa` 형식 지원
- [ ] `python run_experiment.py phase3 --strategy autoresearch --trials 40` 형식 지원
- [ ] 기존 .bat 파일은 호환성을 위해 유지하되, 새 CLI를 호출하도록 변경

---

### REQ-007: [연구 방법론] BERTScore 기본 활성화

**유형**: Ubiquitous
**EARS 서술**: 시스템은 **항상** Phase 1 및 Phase 2의 open-ended 질문 평가 시 BERTScore F1을 계산하여 결과에 포함해야 한다.

**근거**: 설계서 v0.2에서 BERTScore를 주요 평가 지표로 추가하였으나, `compute_overall_accuracy()`의 기본값이 `compute_bertscore=False`. Phase 1 결과에 BERTScore 미포함.

**인수 조건**:
- [x] `evaluate_zero_shot.py`에서 `compute_overall_accuracy()` 호출 시 `compute_bertscore=True` 전달
- [x] 결과 JSON의 `summary` 필드에 `open_bertscore_f1`, `open_bertscore_accuracy` 포함
- [x] BERTScore 계산에 소요되는 추가 시간 로깅
- [x] `phase1_summary.csv`에 BERTScore 관련 컬럼 추가

---

### REQ-008: [코드 품질] 구조화된 로깅 시스템 적용

**유형**: Ubiquitous
**EARS 서술**: 시스템은 **항상** 일관된 구조화 로깅 포맷을 사용하고, 로그를 파일과 콘솔에 동시 출력해야 한다.

**근거**: 현재 각 모듈의 `main()` 함수에서 독립적으로 `logging.basicConfig()` 호출. 중복 핸들러 설정, 로그 레벨 비일관성 문제. 장시간 실험 시 로그 파일이 필수.

**인수 조건**:
- [x] `src/utils/logging_config.py` 모듈 생성
- [x] `setup_logging(log_dir, experiment_name)` 함수 구현
- [x] 콘솔(INFO 레벨) + 파일(DEBUG 레벨) 동시 출력
- [x] 로그 파일명에 타임스탬프 포함 (예: `experiment_2026-04-28_143000.log`)
- [x] 모든 모듈에서 직접 `logging.basicConfig()` 호출 제거, `setup_logging()` 사용

**2026-07-14 재검증 주석**: 위 체크박스는 완료로 표시되어 있었으나, 재검증 시점에 6개 파일(`src/autoresearch/run_phase3.py`, `src/data/download.py`, `src/data/general_vqa.py`, `src/finetune/run_phase2.py`, `src/finetune/train_one.py`, `src/finetune/train_qlora.py`)이 여전히 `logging.basicConfig()`를 직접 호출하고 있는 것이 발견되었다. 이 격차는 커밋 `506d516`으로 이미 해소되었다 — 6개 파일 모두 `src/baseline/run_all.py` / `src/baseline/evaluate_zero_shot.py`의 기존 패턴과 동일하게 `setup_logging()`을 사용하도록 마이그레이션 완료. 현재 시점 REQ-008은 실질적으로 완전히 충족된 상태이다.

---

### REQ-009: [실험 파이프라인] 실험 결과 시각화 모듈 구현

**유형**: WHEN [이벤트] THEN [동작]
**EARS 서술**: **WHEN** 각 Phase의 실험이 완료되면, **THEN** 시스템은 논문에 삽입 가능한 표준 시각화(차트, 테이블)를 자동 생성해야 한다.

**근거**: matplotlib/seaborn이 의존성에 포함되어 있으나, 시각화 코드 미구현. 논문 작성 시 수동 시각화는 비효율적이고 재현성 저하.

**인수 조건**:
- [x] `src/evaluate/visualize.py` 모듈 생성
- [x] Phase 1: 모델별 x 데이터셋별 정확도 히트맵/바 차트
- [x] Phase 1: Closed vs Open 정확도 비교 그래프
- [x] Phase 2: Base vs Fine-tuned 성능 비교 차트
- [x] Phase 3: HPO 전략별 탐색 궤적 (trial vs accuracy) 라인 차트
- [x] 모든 차트를 `results/figures/` 디렉토리에 PNG + PDF 형식 저장
- [x] 논문용 폰트 크기/스타일 적용 (matplotlib rcParams 설정)

---

### REQ-010: [프로젝트 관리] 데이터 로더 테스트 및 검증

**유형**: Ubiquitous
**EARS 서술**: 시스템은 **항상** 데이터셋 로더(`load_medical_vqa_dataset`)의 정합성을 검증하는 테스트를 갖추어야 한다.

**근거**: `src/data/dataset.py`의 데이터 로더가 올바른 필드(image, question, answer, question_type)를 반환하는지 검증하는 테스트 없음. 데이터셋 변경/업데이트 시 사일런트 실패 위험.

**인수 조건**:
- [x] `tests/test_data_loader.py` 생성
- [x] 각 데이터셋(pathvqa, slake, vqa_rad)에 대해 로딩 가능 여부 테스트
- [x] 반환 샘플의 필드 타입 검증 (image: PIL.Image, question: str, answer: str, question_type: str)
- [x] question_type이 "open" 또는 "closed"만 포함하는지 검증
- [x] 빈 문자열 answer가 없는지 검증

---

### REQ-011: [코드 품질] 시드 기반 재현성 검증 테스트

**유형**: Ubiquitous
**EARS 서술**: 시스템은 **항상** 동일 시드로 동일한 결과를 산출함을 검증하는 재현성 테스트를 갖추어야 한다.

**근거**: `set_seed()`가 구현되어 있으나, 실제로 재현성이 보장되는지 검증하는 테스트 없음. `torch.backends.cudnn.deterministic = True` 설정만으로는 완전한 재현성 보장 불가능한 경우 존재.

**인수 조건**:
- [x] `tests/test_reproducibility.py` 생성
- [x] `set_seed()` 후 `torch.randn()` 결과 동일성 검증
- [x] `set_seed()` 후 `random.random()`, `np.random.rand()` 결과 동일성 검증
- [ ] 가능하면 소규모 모델 추론에 대한 재현성 테스트 (max_samples=5)

---

### REQ-012: [프로젝트 관리] 실험 환경 정보 자동 기록

**유형**: WHEN [이벤트] THEN [동작]
**EARS 서술**: **WHEN** 실험이 시작되면, **THEN** 시스템은 실험 환경 정보(Python 버전, torch 버전, CUDA 버전, GPU 정보, OS, 주요 라이브러리 버전)를 결과 JSON에 자동 기록해야 한다.

**근거**: 현재 결과 JSON의 `metadata`에 환경 정보 미포함. 논문 재현성 및 리뷰어 요청 대응에 필수.

**인수 조건**:
- [x] `src/utils/environment.py` 모듈 생성
- [x] `get_environment_info() -> dict` 함수 구현
- [x] 반환 정보: Python version, torch version, CUDA version, GPU name, GPU memory, OS, transformers version, peft version
- [x] 모든 결과 JSON의 `metadata.environment` 필드에 포함
- [x] `check_progress.py`에서도 환경 정보 출력

---

## 우선순위 매트릭스

| 요구사항 | 우선순위 | 노력 | 영향 | 카테고리 |
|----------|----------|------|------|----------|
| REQ-003: CF 인터페이스 버그 수정 | Primary | Low | High | 코드 품질 |
| REQ-005: Phase 1 집계 버그 수정 | Primary | Low | High | 실험 파이프라인 |
| REQ-002: MEDICAL_PROMPT 중복 제거 | Primary | Low | Medium | 코드 품질 |
| REQ-001: 핵심 모듈 단위 테스트 | Primary | Medium | High | 코드 품질 |
| REQ-007: BERTScore 기본 활성화 | Primary | Low | High | 연구 방법론 |
| REQ-004: 통계 분석 모듈 | Secondary | Medium | High | 연구 방법론 |
| REQ-012: 환경 정보 자동 기록 | Secondary | Low | Medium | 프로젝트 관리 |
| REQ-008: 구조화된 로깅 | Secondary | Medium | Medium | 코드 품질 |
| REQ-009: 시각화 모듈 | Final | High | High | 연구 방법론 |
| REQ-010: 데이터 로더 테스트 | Final | Medium | Medium | 프로젝트 관리 |
| REQ-011: 재현성 검증 테스트 | Final | Low | Medium | 코드 품질 |
| REQ-006: .bat -> Python CLI 전환 | Optional | Medium | Low | 실험 파이프라인 |

---

## 구현 단계

### Phase A: 긴급 수정 (Primary)
- REQ-003: CF 인터페이스 버그 수정
- REQ-005: Phase 1 집계 버그 수정 + SmolVLM 누락 시드 완성
- REQ-002: MEDICAL_PROMPT 중복 제거
- REQ-001: metrics.py 단위 테스트 추가
- REQ-007: BERTScore 기본 활성화

### Phase B: 연구 기반 강화 (Secondary)
- REQ-004: 통계 분석 모듈 구현
- REQ-012: 환경 정보 자동 기록
- REQ-008: 구조화된 로깅 시스템

### Phase C: 논문 준비 (Final)
- REQ-009: 시각화 모듈 구현
- REQ-010: 데이터 로더 테스트
- REQ-011: 재현성 검증 테스트

### Phase D: 선택적 개선 (Optional)
- REQ-006: 통합 Python CLI 스크립트

---

## 비목표 (Non-goals)

- 모델 아키텍처 변경 또는 새로운 모델 추가
- 데이터셋 전처리 파이프라인 재설계
- 분산 학습 지원
- CI/CD 파이프라인 구축
- 웹 UI 또는 대시보드 개발

---

## 추적성 (Traceability)

| 요구사항 | 관련 파일 | 설계서 참조 |
|----------|----------|-------------|
| REQ-001 | src/evaluate/metrics.py | - |
| REQ-002 | src/baseline/model_loader.py, src/finetune/prepare_data.py | - |
| REQ-003 | src/evaluate/catastrophic_forgetting.py | Section 4.4 |
| REQ-004 | (신규) src/evaluate/statistics.py | Section 4.3, 4.4, 4.5 |
| REQ-005 | src/baseline/run_all.py, results/phase1_baseline/ | Section 4.3 |
| REQ-006 | run_phase1.bat, run_phase2.bat, run_phase3.bat | - |
| REQ-007 | src/evaluate/metrics.py, src/baseline/evaluate_zero_shot.py | Section 4.3 (v0.2) |
| REQ-008 | src/utils/ (신규 logging_config.py) | - |
| REQ-009 | (신규) src/evaluate/visualize.py | Section 4.3-4.5 |
| REQ-010 | src/data/dataset.py | - |
| REQ-011 | src/utils/seed.py | Section 4.3 |
| REQ-012 | (신규) src/utils/environment.py | Section 5 재현성 |
