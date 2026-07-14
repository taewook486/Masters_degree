# 프로젝트 구조: Medical VQA VLM

## 전체 아키텍처 개요

단일 리포지토리 구조의 Python ML 연구 프로젝트. 3단계 실험 파이프라인(베이스라인 평가 - QLoRA 파인튜닝 - 자율 HPO)을 순차적으로 실행하는 모듈형 아키텍처.

```
[configs/] --> [src/baseline/] --> [results/phase1_baseline/]
                    |
             [src/finetune/] --> [results/phase2_finetune/]
                    |
           [src/autoresearch/] --> [results/phase3_autoresearch/]
                    |
             [src/evaluate/] <-- 공통 평가 모듈
             [src/data/]     <-- 공통 데이터 로더
             [src/utils/]    <-- 유틸리티 (시드, VRAM 모니터)
```

---

## 디렉토리 구조

```
Masters_degree/
|-- configs/                     # 실험 설정 파일
|   |-- models/                  # 모델별 YAML 설정
|   |   |-- qwen25_vl_3b.yaml
|   |   |-- qwen3_vl_2b.yaml
|   |   |-- smolvlm2_2b.yaml
|   |   |-- florence2_large.yaml
|   |-- finetune/                # QLoRA 파인튜닝 설정
|   |   |-- base_qlora.yaml      # 기본 QLoRA 하이퍼파라미터
|   |   |-- ablation/            # Ablation study 설정
|   |       |-- target_minimal.yaml
|   |       |-- target_medium.yaml
|   |       |-- target_full.yaml
|   |-- autoresearch/            # 자율 HPO 에이전트 설정
|       |-- program.md           # LLM 에이전트 시스템 프롬프트
|
|-- src/                         # 소스 코드
|   |-- baseline/                # Phase 1: 제로샷 베이스라인 평가
|   |   |-- evaluate_zero_shot.py  # 단일 조건 평가
|   |   |-- model_loader.py        # 모델 로드/언로드
|   |   |-- run_all.py             # 전체 베이스라인 실행 오케스트레이터
|   |
|   |-- finetune/                # Phase 2: QLoRA 파인튜닝
|   |   |-- train_qlora.py         # QLoRA 학습 (Unsloth/HF PEFT 이중 백엔드)
|   |   |-- prepare_data.py        # 학습 데이터 전처리
|   |   |-- run_phase2.py          # Phase 2 실행 오케스트레이터
|   |
|   |-- autoresearch/            # Phase 3: 자율 HPO
|   |   |-- agent.py               # Claude API 기반 HPO 에이전트
|   |   |-- loop.py                # 자율 실험 루프
|   |   |-- strategies.py          # HPO 전략 (Random, Optuna, Autoresearch)
|   |   |-- tracker.py             # 실험 결과 추적
|   |   |-- run_phase3.py          # Phase 3 실행 오케스트레이터
|   |
|   |-- data/                    # 데이터 로딩 및 전처리
|   |   |-- dataset.py             # 의료 VQA 데이터셋 로더 (추정)
|   |   |-- general_vqa.py         # 범용 VQA 데이터 처리
|   |
|   |-- evaluate/                # 평가 메트릭 (공통)
|   |   |-- metrics.py             # VQA 정확도 (Closed/Open/BERTScore)
|   |   |-- catastrophic_forgetting.py  # Catastrophic Forgetting 측정
|   |
|   |-- utils/                   # 유틸리티
|       |-- seed.py                # 랜덤 시드 고정
|       |-- vram_monitor.py        # GPU VRAM 사용량 모니터링
|
|-- data/                        # 데이터셋 저장소 (gitignore)
|   |-- pathvqa/
|   |-- slake/
|   |-- vqa_rad/
|
|-- results/                     # 실험 결과
|   |-- phase1_baseline/         # Phase 1 결과 JSON + 요약 CSV
|   |-- phase2_finetune/         # Phase 2 결과 (예정)
|   |-- phase3_autoresearch/     # Phase 3 결과 (예정)
|
|-- docs/                        # 논문 관련 문서
|   |-- THESIS_PROPOSAL_FINAL.md   # 논문 설계서 (최종)
|   |-- RISK_ANALYSIS.md           # 리스크 분석
|
|-- tests/                       # 테스트 코드
|   |-- florence2_step_test.py
|
|-- run_phase1.bat               # Phase 1 실행 스크립트 (Windows)
|-- run_phase2.bat               # Phase 2 실행 스크립트
|-- run_phase2_ablation.bat      # Phase 2 Ablation 실행
|-- run_phase3.bat               # Phase 3 실행 스크립트
|-- check_progress.py            # 실험 진행 상황 확인
|-- pyproject.toml               # 프로젝트 메타데이터 및 의존성
|-- uv.lock                      # uv 패키지 매니저 락파일
```

---

## 모듈 관계 및 의존성

### 핵심 모듈 의존성

- `src/baseline/` --> `src/data/`, `src/evaluate/`, `src/utils/`
- `src/finetune/` --> `src/data/`, `src/evaluate/`, `src/baseline/model_loader`, `src/utils/`
- `src/autoresearch/` --> `src/finetune/`, `src/evaluate/`, `src/utils/`
- `src/evaluate/` --> 독립 모듈 (bert-score, scikit-learn 외부 의존)
- `src/data/` --> 독립 모듈 (datasets, Pillow 외부 의존)

### 데이터 흐름

1. **데이터 로딩**: `src/data/` -- HuggingFace datasets 또는 로컬 파일에서 의료 VQA 데이터셋 로드
2. **모델 로딩**: `src/baseline/model_loader.py` -- YAML 설정 기반 VLM 모델 로드 (양자화 옵션 포함)
3. **추론/학습**: `src/baseline/` (제로샷) 또는 `src/finetune/` (QLoRA) 실행
4. **평가**: `src/evaluate/metrics.py` -- Closed/Open accuracy, BERTScore F1 계산
5. **결과 저장**: `results/` 디렉토리에 JSON 형식으로 per-sample 및 summary 결과 저장

---

## 외부 시스템 통합

| 시스템 | 용도 | 프로토콜 |
|--------|------|----------|
| HuggingFace Hub | 모델 다운로드 (Qwen-VL, SmolVLM) | HTTPS/API |
| HuggingFace Datasets | 데이터셋 다운로드 (PathVQA, SLAKE, VQA-RAD) | HTTPS/API |
| Anthropic Claude API | Phase 3 자율 HPO 에이전트 | REST API (anthropic SDK) |
| Weights & Biases | 실험 추적 (선택적) | REST API (wandb SDK) |
| PyTorch CUDA | GPU 연산 (RTX 5060 Ti) | CUDA 12.8 |

---

## 아키텍처 결정 배경

- **단일 리포지토리**: 석사 연구 프로젝트로 단일 연구자가 관리하므로 모노리포 구조 채택
- **YAML 기반 설정**: 실험 재현성을 위해 모든 하이퍼파라미터를 YAML 설정 파일로 관리
- **이중 백엔드 (Unsloth/HF PEFT)**: Qwen 모델은 Unsloth로 가속, 미지원 모델은 표준 HF PEFT로 폴백
- **배치 스크립트**: Windows 환경(RTX 5060 Ti 데스크톱)에서 장시간 실험 실행을 위해 .bat 파일 사용
- **JSON 결과 형식**: per-sample 결과까지 저장하여 사후 분석 및 오류 유형 분석 가능
