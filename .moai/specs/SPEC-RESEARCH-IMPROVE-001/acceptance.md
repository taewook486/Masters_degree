# SPEC-RESEARCH-IMPROVE-001: 인수 조건

## 영역 1: Phase 1 실험 파이프라인

### AC-001: BERTScore 기본 활성화 (REQ-RI-001)

**Given** Phase 1 zero-shot 평가가 실행될 때
**When** `evaluate_with_loaded_model()`이 호출되면
**Then** 결과 JSON의 `summary` 필드에 `open_bertscore_f1`과 `open_bertscore_accuracy` 키가 포함되어야 한다

**Given** `bert-score` 패키지가 설치되지 않은 환경에서
**When** `compute_overall_accuracy(compute_bertscore=True)`가 호출되면
**Then** 경고 로그를 기록하고 BERTScore 필드를 0.0으로 설정하되, 나머지 메트릭은 정상 반환해야 한다

**검증 방법**:
- [ ] `results/phase1_baseline/` 내 결과 JSON 파일에 `open_bertscore_f1` 키 존재 확인
- [ ] `compute_overall_accuracy`의 기본 시그니처에서 `compute_bertscore=True` 확인
- [ ] bert-score 미설치 시 fallback 동작 확인 (수동 또는 모의 테스트)

---

### AC-002: 결과 집계 완전성 (REQ-RI-002)

**Given** 3개 시드(42, 123, 456)에 대한 Phase 1 결과가 모두 존재할 때
**When** `_aggregate_seed_results()`가 호출되면
**Then** `num_seeds=3`이 기록되고, `closed_acc_std`, `open_acc_std`, `overall_acc_std` 값이 0.0이 아닌 실제 표준편차를 반영해야 한다

**Given** 기존 결과 JSON에 BERTScore 키가 없을 때
**When** `_load_existing_result()`가 해당 파일을 로드하면
**Then** `None`을 반환하여 재평가를 유도해야 한다

**검증 방법**:
- [ ] `phase1_summary.csv`에서 `num_seeds` 컬럼 값이 3인 행 존재 확인
- [ ] `bertscore_f1_mean`, `bertscore_f1_std` 컬럼이 CSV에 포함 확인
- [ ] BERTScore 키 없는 레거시 결과에 대해 재평가가 트리거되는지 확인

---

### AC-003: 환경 정보 자동 기록 (REQ-RI-003)

**Given** 실험이 시작될 때
**When** 결과 JSON이 저장되면
**Then** `metadata.environment` 필드에 최소 다음 키가 포함되어야 한다: `python_version`, `torch_version`, `cuda_version`, `gpu_name`, `os`

**Given** CPU-only 환경에서 실험이 실행될 때
**When** `get_environment_info()`가 호출되면
**Then** `cuda_version`은 `"N/A"`, `gpu_name`은 `"CPU-only"`로 기록되어야 한다

**검증 방법**:
- [ ] 결과 JSON 파일에서 `metadata.environment.torch_version` 값이 실제 설치 버전과 일치 확인
- [ ] `get_environment_info()` 함수의 반환 dict에 8개 필수 키 존재 확인

---

## 영역 2: Autoresearch 에이전트

### AC-004: 중복 설정 감지 (REQ-RI-004)

**Given** 이전 trial에서 `lora_rank=16, lora_alpha=32, learning_rate=2e-4, lora_targets=minimal, max_steps=400` 설정이 완료되었을 때
**When** 에이전트가 동일한 설정을 제안하면
**Then** 시스템이 "중복 설정 감지" 경고를 로그에 기록하고, 최대 3회까지 재제안을 요청해야 한다

**Given** 3회 재제안 후에도 중복 설정이 반복될 때
**When** 4번째 시도에서도 중복이 감지되면
**Then** RandomSearchStrategy로 fallback하여 무작위 설정을 반환해야 한다

**검증 방법**:
- [ ] 동일 설정을 반복 반환하는 모의 에이전트로 중복 감지 로직 확인
- [ ] 3회 초과 시 Random fallback 동작 확인
- [ ] learning_rate 상대 오차 5% 이내 판정 확인 (예: 2e-4 vs 2.05e-4는 중복)

---

### AC-005: 에이전트 응답 검증 강화 (REQ-RI-005)

**Given** 에이전트가 `{"lora_rank": 16, "learning_rate": 2e-4}` (불완전한 JSON)을 반환할 때
**When** `_validate_config()`가 호출되면
**Then** 누락된 `lora_alpha`, `batch_size` 등에 기본값을 적용하고, 각 누락 키에 대해 WARNING 로그를 기록해야 한다

**Given** 에이전트가 `epochs` 키만 포함하고 `max_steps`가 없는 응답을 반환할 때
**When** `_validate_config()`가 호출되면
**Then** `epochs * 100`으로 `max_steps`를 계산하고, `epochs` 키를 제거하며 마이그레이션 INFO 로그를 기록해야 한다

**검증 방법**:
- [ ] 필수 키 누락 시 기본값 적용 및 경고 로그 확인
- [ ] `epochs=3` 입력 시 `max_steps=300` 변환 확인
- [ ] 반환된 config에 `epochs` 키가 제거되고 `max_steps` 키만 존재 확인

---

### AC-006: 탐색/활용 로깅 (REQ-RI-006)

**Given** trial 5/40 (progress=0.125)에서 에이전트가 호출될 때
**When** `ask_agent_for_config()`가 실행되면
**Then** 로그에 `phase=EXPLORATION`, `temperature=0.91`, `trial=5/40` 정보가 구조화 형식으로 기록되어야 한다

**Given** trial 35/40 (progress=0.875)에서 에이전트가 호출될 때
**When** `ask_agent_for_config()`가 실행되면
**Then** 로그에 `phase=EXPLOITATION`, `temperature=0.39` 정보가 기록되어야 한다

**검증 방법**:
- [ ] 로그 출력에서 phase, temperature, trial 정보 파싱 가능 확인
- [ ] progress < 0.25 시 EXPLORATION, 0.25-0.75 시 TRANSITION, > 0.75 시 EXPLOITATION 레이블 확인

---

## 영역 3: 데이터 처리/저장

### AC-007: 구조화 로깅 (REQ-RI-007)

**Given** `setup_logging("results/logs", "phase1_baseline")`이 호출될 때
**When** 로깅이 설정되면
**Then** `results/logs/phase1_baseline_YYYYMMDD_HHMMSS.log` 파일이 생성되어야 한다

**Given** 구조화 로깅이 설정된 상태에서
**When** `logger.info("test message")`가 호출되면
**Then** 콘솔(INFO 레벨)과 파일(DEBUG 레벨) 양쪽에 메시지가 출력되어야 한다

**검증 방법**:
- [ ] 로그 파일이 지정된 디렉토리에 타임스탬프 포함 파일명으로 생성되는지 확인
- [ ] 콘솔과 파일에 동시 출력 확인
- [ ] 기존 `logging.basicConfig()` 호출이 모든 main() 함수에서 제거되었는지 확인

---

### AC-008: HPO 체크포인트 및 재개 (REQ-RI-008)

**Given** 20/40 trial까지 완료된 HPO 루프가 중단되었을 때
**When** 동일한 strategy + repeat_id로 `run_hpo_loop()`가 재실행되면
**Then** trial 20부터 이어서 실행하여 총 40 trial을 완료해야 한다 (기존 20개 trial 재실행 없음)

**Given** tracker에 strategy="autoresearch", repeat_id=0으로 15개 완료 trial이 있을 때
**When** `run_hpo_loop(max_trials=40)`가 호출되면
**Then** 25개의 새로운 trial만 실행해야 한다

**검증 방법**:
- [ ] 기존 trial 수가 N인 상태에서 재시작 시, `range(N, max_trials)` 범위로 실행되는지 확인
- [ ] 재시작 시 기존 trial 결과가 strategy에 올바르게 전달되는지 확인

---

### AC-009: 결과 저장 형식 통일 (REQ-RI-009)

**Given** Phase 3 HPO 실험이 완료될 때
**When** `tracker.export_json(output_dir)`가 호출되면
**Then** 각 trial이 `metadata` (trial_id, strategy, timestamp), `summary` (val_accuracy, train_loss 등), `hyperparameters` 구조로 JSON 파일에 저장되어야 한다

**Given** CF 측정 결과가 저장될 때
**When** `run_cf_measurement()`가 완료되면
**Then** 결과 JSON에 `metadata` (model_name, timestamp, environment) 섹션이 포함되어야 한다

**검증 방법**:
- [ ] Phase 3 결과 JSON 파일의 최상위 키가 `metadata`, `summary`를 포함하는지 확인
- [ ] CF 결과 JSON의 `metadata` 섹션에 환경 정보가 포함되는지 확인
- [ ] Phase 1 결과 JSON과 Phase 3 결과 JSON의 `metadata` 키 구조가 일관적인지 확인

---

## Quality Gate 기준

### Definition of Done

- [ ] 모든 요구사항(REQ-RI-001 ~ REQ-RI-009)의 인수 조건 충족
- [ ] 신규 모듈(`logging_config.py`, `environment.py`, `checkpoint.py`)에 docstring 포함
- [ ] 기존 테스트(`tests/florence2_step_test.py`) 통과에 영향 없음
- [ ] `ruff check src/` 실행 시 신규 경고 0건
- [ ] Phase 1 결과 JSON에 BERTScore 및 환경 정보 포함 확인 (실제 실행 또는 dry-run)

### Edge Cases

- bert-score 패키지 미설치 환경에서의 graceful degradation
- GPU 없는 환경(CPU-only)에서 환경 정보 수집
- 빈 experiment tracker (trial 0개)에서 HPO 재개
- 에이전트 API 완전 실패 시 Random fallback 체인
- BERTScore 모델과 VLM 모델 동시 VRAM 부족 시 OOM 처리
