# SPEC-RESEARCH-IMPROVE-001: 구현 계획

## 개요

9개 요구사항을 3개 마일스톤으로 분류하여 의존성 순서대로 구현한다.

---

## 마일스톤 1: 기반 인프라 (Priority High)

Phase 1 파이프라인과 데이터 파이프라인 개선의 기반이 되는 유틸리티 모듈을 먼저 구현한다.

### M1-1: 구조화 로깅 시스템 (REQ-RI-007)

**작업 항목**:
1. `src/utils/logging_config.py` 신규 생성
   - `setup_logging(log_dir: str, experiment_name: str, console_level=INFO, file_level=DEBUG)` 함수
   - 콘솔 핸들러 (INFO) + 파일 핸들러 (DEBUG) 동시 출력
   - 로그 파일명: `{experiment_name}_{YYYYMMDD_HHMMSS}.log`
   - 포맷: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`
2. 기존 `logging.basicConfig()` 호출 제거 및 `setup_logging()` 호출로 교체
   - `src/baseline/run_all.py:254`
   - `src/baseline/evaluate_zero_shot.py:304`
   - `src/autoresearch/loop.py` (해당 시 추가)

**리스크**: 기존 로그 출력 형식에 의존하는 외부 파싱 로직이 있을 경우 호환성 문제. 현재 코드 분석상 로그 파싱 의존성 없음.

### M1-2: 환경 정보 수집 모듈 (REQ-RI-003)

**작업 항목**:
1. `src/utils/environment.py` 신규 생성
   - `get_environment_info() -> dict` 함수
   - 수집 항목: Python version, torch version, CUDA version, GPU name, GPU memory, OS, transformers version, peft version
2. `src/baseline/evaluate_zero_shot.py`의 결과 JSON `metadata`에 `environment` 필드 추가

**리스크**: GPU가 없는 환경(CPU-only)에서 CUDA 관련 정보 수집 실패 가능. `torch.cuda.is_available()` 분기 처리 필요.

---

## 마일스톤 2: Phase 1 파이프라인 개선 (Priority High)

M1 완료 후, Phase 1 실험 파이프라인의 핵심 기능을 개선한다.

### M2-1: BERTScore 기본 활성화 (REQ-RI-001)

**작업 항목**:
1. `src/evaluate/metrics.py:163` - `compute_bertscore` 기본값을 `True`로 변경
2. `src/baseline/evaluate_zero_shot.py:93` - `compute_overall_accuracy()` 호출 시 `compute_bertscore=True` 명시적 전달
3. BERTScore 계산 소요 시간을 별도 타이밍 로그로 기록

**기술적 접근**:
- `bert-score` 패키지가 미설치된 환경에서는 경고 로그 후 BERTScore 없이 진행 (기존 `compute_open_bertscore`의 fallback 로직 활용)
- BERTScore 모델(`roberta-large`) 최초 로딩 시 ~2GB VRAM 추가 사용. RTX 4090(24GB) 기준 VLM 모델과 동시 로딩 가능 여부 확인 필요

### M2-2: 결과 집계 완전성 (REQ-RI-002)

**작업 항목**:
1. `src/baseline/run_all.py`의 `_aggregate_seed_results()`에 BERTScore 관련 키 집계 추가
   - `bertscore_f1_mean`, `bertscore_f1_std`, `bertscore_acc_mean`, `bertscore_acc_std`
2. `_load_existing_result()`에서 BERTScore 키 존재 여부 확인. 없으면 `None` 반환하여 재평가 유도
3. `generate_summary_csv()`의 DataFrame에 BERTScore 컬럼 포함

**리스크**: 기존 Phase 1 결과 JSON에 BERTScore 키가 없으므로, 기존 결과를 활용하려면 BERTScore만 추가 계산하는 패치 스크립트가 필요할 수 있음.

---

## 마일스톤 3: Autoresearch 및 데이터 파이프라인 고도화 (Priority Medium)

### M3-1: 중복 설정 감지 (REQ-RI-004)

**작업 항목**:
1. `src/autoresearch/strategies.py`의 `AutoresearchStrategy.suggest()`에 중복 검사 추가
   - 완료된 trial의 설정과 비교 (lora_rank, lora_alpha, learning_rate, batch_size, grad_accum_steps, lora_targets, max_steps)
   - 동일 설정 감지 시 최대 3회 재제안 요청
   - 3회 초과 시 Random fallback
2. 중복 판정 기준: learning_rate는 상대 오차 5% 이내를 동일로 간주

### M3-2: 에이전트 응답 검증 강화 (REQ-RI-005)

**작업 항목**:
1. `_validate_config()`에 필수 키 존재 확인 추가
   - 필수 키: `lora_rank`, `lora_alpha`, `learning_rate`, `batch_size`, `grad_accum_steps`, `lora_targets`
   - 누락 키에 대해 기본값 적용 + 경고 로그
2. `epochs` -> `max_steps` 마이그레이션
   - `epochs` 키가 존재하고 `max_steps`가 없으면, `epochs * 100` 으로 `max_steps` 추정
   - `_VALID_EPOCHS` 상수는 하위 호환성을 위해 유지하되, `max_steps` 우선 적용

### M3-3: 탐색/활용 로깅 (REQ-RI-006)

**작업 항목**:
1. `ask_agent_for_config()`에 구조화 로그 추가
   - 로그 내용: trial 번호, progress 비율, 온도 값, 단계(EXPLORATION/TRANSITION/EXPLOITATION)
2. `TrialResult` dataclass에 `temperature` 및 `phase` 필드 추가 (선택)

### M3-4: HPO 체크포인트 및 재개 (REQ-RI-008)

**작업 항목**:
1. `run_hpo_loop()`에 기존 완료 trial 수 확인 로직 추가
   - `tracker.load_by_strategy(strategy_name, repeat_id)`로 기존 완료 수 확인
   - `range(existing_count, max_trials)`로 반복 범위 조정
2. `src/utils/checkpoint.py` 신규 생성 (선택적)
   - 루프 상태(현재 trial 인덱스, best 결과) JSON 저장/복원

### M3-5: 결과 저장 형식 통일 (REQ-RI-009)

**작업 항목**:
1. `ExperimentTracker`에 `export_json(output_dir: str)` 메서드 추가
   - 각 trial을 `metadata/summary` 구조로 변환하여 JSON 내보내기
2. `catastrophic_forgetting.py`의 출력 스키마에 `metadata` 섹션 추가
   - `model_name`, `timestamp`, `environment` 등 메타데이터 포함

---

## 구현 순서 요약

```
M1 (기반 인프라)
  M1-1: logging_config.py
  M1-2: environment.py
      |
      v
M2 (Phase 1 파이프라인)
  M2-1: BERTScore 활성화
  M2-2: 집계 완전성
      |
      v
M3 (Autoresearch + 데이터)
  M3-1: 중복 감지
  M3-2: 응답 검증
  M3-3: 탐색/활용 로깅
  M3-4: 체크포인트
  M3-5: 형식 통일
```

---

## 기술적 제약사항

- **GPU**: RTX 4090 24GB 단일 GPU. BERTScore 모델(roberta-large, ~2GB) + VLM 모델(~8-14GB) 동시 VRAM 수용 확인 필요
- **Python**: 3.13+, uv 패키지 매니저
- **의존성**: bert-score, scipy, optuna, anthropic (기존 pyproject.toml에 포함)
- **OS**: Windows 11 (경로 구분자 주의)
