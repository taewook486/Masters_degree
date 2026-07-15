# Changelog

All notable changes to this research project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — Phase 2 실행 중

Phase 2 QLoRA 미세조정 실험 진행 중 (RunPod RTX 4090, 36조건 × max_steps=500)

---

## [0.3.0] — 2026-07-14

### Added — SPEC-EVAL-METRICS-001 (BioBERT dual BERTScore + Mann-Whitney)

- `src/evaluate/statistics.py`: `run_mann_whitney(x, y)` — 2-sample two-sided Mann-Whitney U with rank-biserial effect size and tie correction (REQ-EM-006~009)
- `src/evaluate/metrics.py`: `compute_open_bertscore()` — `num_layers` 파라미터 추가 (dmis-lab/biobert-v1.1용 `num_layers=9` 명시적 지정)
- `src/evaluate/metrics.py`: `compute_overall_accuracy()` — `bertscore_models` opt-in 파라미터 (기본값 `None` → Phase 1 backward compat)
- `src/baseline/evaluate_zero_shot.py`: `evaluate_with_loaded_model()` — `bertscore_models` opt-in 전파
- `src/finetune/train_qlora.py`: Phase 2 post-training 평가에 `bertscore_models=["roberta-large","dmis-lab/biobert-v1.1"]` 명시적 전달
- `tests/test_statistics.py`: 5개 테스트 (AC-2-1..AC-2-4, scipy 실호출)
- `tests/test_metrics.py`: 6개 테스트 (AC-1-1..AC-1-4 + BioBERT opt-in 엣지케이스)

### Added — Phase 2 기능

- `scripts/run_phase2_main.sh`: `--no_cf` 플래그 (CF 베이스라인 생략, 스모크 가속)
- `scripts/run_phase2_main.sh`: `--max_test_samples` 플래그 (post-training 평가 상한, 스모크 가속)
- `src/finetune/prepare_data.py`: `MOAI_CHAT_CACHE_DIR` env var로 chat dataset 디스크 캐시 라우팅 (조건당 재빌드 제거)

### Fixed — Phase 2 인프라

- **디스크 quota**: `HF_HOME=/hf_cache`, `MOAI_CHAT_CACHE_DIR=/hf_cache` → 컨테이너 디스크(60GB)로 라우팅, 볼륨 디스크 50GB quota 소진 방지
- **학습 시간**: `max_steps=500` cap (`configs/finetune/base_qlora.yaml`) — 26h/조건 → ~1.8h/조건 (36조건 합산 ~65h)
- **eval 타이밍**: `eval/save_steps=max_steps` — 학습 종료 시점 1회 평가 (중간 평가 제거)
- **백엔드 격리**: unsloth/standard 백엔드를 별도 서브프로세스(`train_one.py`)로 실행 — SFTTrainer 전역 패치 충돌 해결
- **SmolVLM2 dtype**: `get_image_features()` 출력을 모델 dtype으로 캐스트 (`inputs_merger` 수정)
- **Standard backend**: TRL 0.24 native VLM API로 재작성
- **Gemma4-E2B**: standard backend + 텍스트 전용 LoRA 타깃으로 활성화
- **CF baseline OOM**: main-process GPU에서 모델 완전 해제 후 CF generation
- **dtype 충돌**: standard backend post-training 평가/CF generation에 autocast 래퍼 추가
- **bf16/fp16**: 모델 dtype 기반 자동 설정 + bnb compute dtype 일치
- **CF baseline**: `load_model` 호출 시그니처 수정 (`'str' has no model_id` 오류)

### Fixed — 기타

- `fix`: .bat 파일 git 충돌 마커 제거 + 남은 모듈 6개 구조화 로깅 통일 (SPEC-IMPROVE-001)

### Documentation

- `docs/RUNPOD_GUIDE.md`: Phase 2 실행 준비 상태 및 4.0절 검증 절차 추가
- `docs/RUNPOD_GUIDE.md`: Gemma4 상태 갱신 (수정 완료, 4모델 12조건 스모크)
- `.moai/specs/SPEC-EVAL-METRICS-001/spec.md`: sync-phase close (status → completed)
- `.moai/specs/SPEC-IMPROVE-001`: REQ-005/REQ-008 추적 정확도 수정

---

## [0.2.0] — 2026-07 (Phase 1 베이스라인 확정)

### Added

- Phase 1 제로샷 베이스라인 평가 완료
  - 4개 모델 (Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B)
  - 3개 데이터셋 (PathVQA, SLAKE, VQA-RAD)
  - 3개 시드 (42, 123, 456) — greedy decoding, STD=0.0 기대값
  - 결과 확정 및 frozen (`results/phase1_baseline/`)
- SPEC-IMPROVE-001 완료 (10/12 REQ): 구조화 로깅, 재현성 개선, SLAKE 이중언어 처리

---

## [0.1.0] — 2026-05 (프로젝트 초기화)

### Added

- 프로젝트 구조 초기화: Phase 1 베이스라인 평가 파이프라인
- VLM 대상: Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B
- 데이터셋: PathVQA, SLAKE, VQA-RAD
- QLoRA 미세조정 인프라 (Phase 2 skeleton)
- 논문 설계서 v0.1~v0.5 작성
- Florence-2-large 제외 (transformers 5.x SA causal mask 캐시 버그)
