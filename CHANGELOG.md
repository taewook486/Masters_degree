# Changelog

All notable changes to this research project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — Phase 3 스모크 진행 중

Phase 3 자율 HPO 스모크 테스트 진행 중 (2/8 trial, ~25분/trial, 2026-07-27). 전체 실행 규모·비용 재추정 중.

---

## [0.4.0] — 2026-07-27

Phase 2 QLoRA 미세조정 완료(75조건), Phase 3 자율 HPO 착수, Phase 1 방법론 정정.

### Added — Phase 2 완료 및 분석

- **Phase 2 전체 75조건 완료** — Main 36(4모델×3데이터셋×3시드) + Ablation A(비율 5종×3시드) + B(rank 5종×3시드) + C(target 3종×3시드) = 75조건 (RunPod)
- Ablation 최적 설정 확정: `rank=64, alpha=128, target_modules=full(all-linear)` → `configs/finetune/base_qlora.yaml` 반영 (단, 세 축 동시 적용 조합 자체는 미검증 — 한계점 기록)
- Cross-dataset CF 72/72 조건 측정 완료 (Table 4.2b(B)). `transformers 5.5.0` TokenizersBackend 리팩터링으로 빠진 `build_inputs_with_special_tokens` 버그 수정 (`src/evaluate/metrics.py`)
- RQ2 3중 검증(paired t-test + BCa Bootstrap Cohen's d + Wilcoxon): **모델별 이질적 효과** — Qwen2.5-VL-3B(d=+2.65)·Qwen3-VL-2B(d=+1.62) 향상, SmolVLM2-2.2B(d=-2.28) 저하. Mixed-Effects pooled는 이질성 상쇄로 비유의(해석 주의)
- `scripts/analyze_clinical.py`(WCA), `scripts/analyze_phase2.py`(RQ2), `scripts/analyze_phase3.py`, `scripts/measure_cross_dataset_cf.py` 추가

### Added — Phase 3 (진행 중)

- 자율 HPO 스모크 테스트 착수 — 4전략(Manual/Random/Optuna TPE/Autoresearch) 비교. 2/8 trial 완료(~25분/trial)
- `src/autoresearch/`: LLM 에이전트 기반 HPO — temperature 스케줄링(1.0→0.3, 초반 탐색/후반 활용), `configs/autoresearch/program.md` 시스템 프롬프트

### Changed — Phase 1 방법론 정정 (설계서 v0.6)

- zero-shot은 greedy 결정적이므로 **3시드 → 1시드(42), 12조건**으로 정정. 불확실성은 부트스트랩 95% CI로 보고
- RQ1 모델 비교: ANOVA + Tukey HSD → **Cochran's Q + McNemar(Bonferroni)** (공유 테스트셋 짝지은 검정)
- `results/phase1_baseline/`를 1시드 12조건 정본으로 정정(구 3시드 산출물은 `_3seed_debug`로 분리)

### Fixed — Phase 3 인프라

- `anthropic` 의존성이 `pyproject.toml`/`uv.lock`에 누락돼 Autoresearch 전략이 새 pod에서 즉시 실패하던 문제 수정

### Fixed — Phase 2 캐시/디스크 인프라 (2026-07-18~19)

- **SIGKILL(cgroup 메모리 초과)**: `src/data/dataset.py`에 `iter_medical_vqa_dataset()`(제너레이터)/`dataset_length()` 추가, `prepare_data.py`가 `Dataset.from_generator()`로 스트리밍 빌드하도록 변경 — PathVQA train(19,654장) 전체를 리스트로 한번에 들고 있다가 죽던 문제 해결
- **컨테이너 디스크 풀(30/36조건 연쇄 실패)**: 모델 캐시(hub)·데이터셋 캐시(chat_cache)를 `/workspace` 볼륨으로 이관
- **불완전 캐시 영구 재실패**: `prepare_data.py`에 `_atomic_save_to_disk()` 추가 — 임시 경로에 저장 후 성공 시에만 원자적 rename, 디스크 풀 등으로 빌드가 중간에 끊겨도 불완전한 캐시 폴더가 남지 않도록 수정
- **볼륨 쿼터 초과(Disk quota exceeded)**: 모델 캐시(hub, 상대적으로 고정 용량)는 컨테이너 디스크로, 데이터셋 캐시(chat_cache, 누적 용량이 큼)는 볼륨으로 분리 배치
- **VQA-RAD eval split 회귀**: `train_qlora.py`의 "validation split 없으면 train 마지막 10%로 대체" 폴백이, 스트리밍 빌드 전환 후 `ValueError`가 `datasets`에 의해 `DatasetGenerationError`로 감싸지면서 더는 잡히지 않던 문제 수정 — `__cause__`를 확인해 진짜 "split 없음"일 때만 대체하고 그 외(디스크 풀 등 실제 에러)는 그대로 재발생

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
