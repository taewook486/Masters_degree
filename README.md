# 경량 비전-언어 모델의 의료 VQA 도메인 적응

**Domain Adaptation of Lightweight Vision-Language Models for Medical Visual Question Answering: QLoRA Fine-Tuning with Autonomous Hyperparameter Optimization**

---

건국대학교 정보통신대학원 융합정보기술학과 AI 전공 석사학위 논문 연구 프로젝트

- **제출 목표**: 2026년 9월

---

## 프로젝트 개요

본 연구는 RTX 5060 Ti (16GB VRAM) 수준의 소비자용 GPU 환경에서 경량 비전-언어 모델(VLM)을 의료 시각 질의응답(VQA) 태스크에 도메인 적응시키는 방법론을 탐구합니다.

4-bit NF4 양자화 기반 QLoRA 미세조정과 Optuna 기반 자율 하이퍼파라미터 최적화를 결합하여, 제한된 연산 자원에서도 의료 VQA 성능을 향상시킬 수 있음을 실증합니다.

---

## 연구 목표

1. 경량 VLM의 의료 VQA 제로샷(zero-shot) 성능 기준치 확립
2. QLoRA 미세조정을 통한 도메인 특화 성능 향상 분석
3. Bayesian 최적화 기반 하이퍼파라미터 자율 탐색 파이프라인 구현
4. 16GB VRAM 환경에서 재현 가능하고 실용적인 의료 AI 적응 방법론 제시

---

## 대상 모델

| 모델 | 파라미터 수 | HuggingFace ID | VRAM (FP16) | 상태 |
|------|------------|----------------|-------------|------|
| Qwen3-VL-2B | 2B | `Qwen/Qwen3-VL-2B-Instruct` | ~3.96 GB | 사용 중 |
| Qwen2.5-VL-3B | 3B | `Qwen/Qwen2.5-VL-3B-Instruct` | ~6.99 GB | 사용 중 |
| SmolVLM2-2.2B | 2.2B | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | ~4.72 GB | 사용 중 |
| Gemma4-E2B | 2.3B (active) / 5.1B (total) | `google/gemma-4-E2B-it` | ~10.3 GB | 사용 중 |
| ~~Florence-2-large~~ | ~~0.77B~~ | ~~`microsoft/Florence-2-large`~~ | ~~-~~ | 제외 (transformers 5.x 비호환) |

> Gemma 4 E2B는 Per-Layer Embeddings(PLE) 기술로 2.3B active 파라미터만으로 5.1B급 표현력을 제공하며, Apache 2.0 라이선스로 공개된 최신 멀티모달 모델입니다.

> Florence-2-large는 transformers 5.x 환경에서 SA causal mask 캐시 버그로 인해 정상 추론 불가 판정 후 제외되었습니다.

---

## 데이터셋

| 데이터셋 | 의료 분야 | 이미지 수 | QA 쌍 수 | 언어 |
|---------|---------|---------|---------|------|
| PathVQA | 병리학 | 5,000 | 33,000 | 영어 |
| SLAKE | 방사선 (다분야) | 642 | 14,000 | 영어 + 중국어 |
| VQA-RAD | 방사선학 | 315 | 2,200 | 영어 |

---

## 실험 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA RTX 5060 Ti 16GB GDDR7 |
| CPU | AMD Ryzen 5 5600X |
| RAM | 32GB |
| OS | Windows 11 |
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 (CUDA 12.8) |
| Transformers | 5.5.0 |
| 패키지 관리자 | uv |

---

## 기술 스택

- **딥러닝 프레임워크**: PyTorch 2.10.0+cu128, HuggingFace Transformers 5.5.0
- **미세조정**: PEFT, TRL, BitsAndBytes (4-bit QLoRA)
- **하이퍼파라미터 최적화**: Optuna (Bayesian HPO)
- **실험 추적**: WandB
- **설정 관리**: OmegaConf (YAML)
- **통계 분석**: SciPy, scikit-learn (ANOVA, t-test, Bootstrap CI)

---

## 프로젝트 구조

```
Masters_degree/
├── configs/
│   ├── models/          # 모델별 YAML 설정 (5개 모델)
│   ├── finetune/        # QLoRA 학습 설정 및 ablation 실험
│   └── autoresearch/    # HPO 탐색 공간 설정
├── src/
│   ├── baseline/        # 제로샷 평가 파이프라인
│   │   ├── model_loader.py
│   │   ├── evaluate_zero_shot.py
│   │   └── run_all.py
│   ├── data/            # 데이터셋 다운로드 및 로딩
│   │   ├── download.py
│   │   └── dataset.py
│   ├── finetune/        # QLoRA 미세조정 파이프라인
│   ├── autoresearch/    # 자율 HPO 루프
│   ├── evaluate/        # VQA 평가 지표 (closed/open/overall)
│   └── utils/           # 공용 유틸리티
│       ├── seed.py      # 재현성 시드 고정
│       └── vram_monitor.py  # VRAM 사용량 모니터링
├── results/
│   ├── phase1_baseline/   # Phase 1 실험 결과
│   ├── phase2_finetune/   # Phase 2 실험 결과
│   └── phase3_autoresearch/  # Phase 3 실험 결과
├── docs/                  # 작업 로그 및 문서
├── tests/                 # 단위 테스트
├── pyproject.toml         # 프로젝트 설정 및 의존성
└── uv.lock                # 의존성 잠금 파일
```

---

## 설치 방법

### 사전 요구 사항

- Python 3.11 이상
- CUDA 12.8 호환 NVIDIA GPU (최소 8GB VRAM 권장, 16GB 이상 최적)
- [uv](https://docs.astral.sh/uv/) 패키지 관리자

### uv 설치

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 프로젝트 설치

```bash
# 저장소 클론
git clone <repository-url>
cd Masters_degree

# 가상 환경 생성 및 의존성 설치 (PyTorch CUDA 12.8 빌드 포함)
uv sync

# 선택 사항: Unsloth 가속 학습 (cu128 환경)
uv sync --extra unsloth

# 개발 도구 포함 설치
uv sync --extra dev
```

> `pyproject.toml`에 PyTorch CUDA 12.8 인덱스가 사전 설정되어 있어 별도 설정 없이 GPU 버전이 자동으로 설치됩니다.

---

## 실행 방법

### Phase 1: 제로샷 베이스라인 평가

```bash
# 전체 평가 실행 (4개 모델 x 3개 데이터셋 x 3개 시드 = 36개 조건)
.venv/Scripts/python.exe src/baseline/run_all.py

# 특정 모델만 평가
.venv/Scripts/python.exe src/baseline/run_all.py --model qwen3_vl_2b

# 기존 결과 무시하고 재실행
.venv/Scripts/python.exe src/baseline/run_all.py --no_skip_existing

# 결과 확인
cat results/phase1_baseline/phase1_summary.csv
```

### WandB 실험 추적 설정

```bash
# WandB 로그인 (최초 1회)
.venv/Scripts/python.exe -m wandb login
```

---

## 연구 단계 현황

### Phase 1: 제로샷 베이스라인 평가

**목표**: 미세조정 전 모델의 의료 VQA 기준 성능 측정

| 항목 | 내용 |
|------|------|
| 대상 모델 | Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM2-2.2B, Gemma4-E2B |
| 대상 데이터셋 | PathVQA, SLAKE, VQA-RAD |
| 실험 조건 | 4개 모델 x 3개 시드 (42, 123, 456) = 36개 조건 |
| 평가 지표 | Closed/Open/Overall Accuracy |
| 진행 상태 | 진행 중 (기존 3개 모델 27개 조건 완료, Gemma4-E2B 9개 조건 대기) |

### Phase 2: QLoRA 미세조정 분석

**목표**: 하이퍼파라미터 체계적 분석을 통한 최적 미세조정 설정 탐색

- QLoRA rank (r), alpha, dropout 효과 분석
- 학습률, 배치 크기, 에폭 수 영향 분석
- 데이터셋별 도메인 적응 특성 비교

**진행 상태**: 대기 중 (Phase 1 완료 후 착수)

### Phase 3: 자율 하이퍼파라미터 최적화

**목표**: Optuna 기반 Bayesian 최적화로 자율 하이퍼파라미터 탐색 파이프라인 구현

- 탐색 공간 정의 및 Bayesian HPO 구현
- Phase 2 수동 분석 결과와 자율 탐색 결과 비교
- 최적 하이퍼파라미터 조합 도출

**진행 상태**: 대기 중 (Phase 2 완료 후 착수)

---

## 재현성

모든 실험은 3개의 고정 시드(42, 123, 456)로 수행됩니다. 통계 검증에는 ANOVA + Tukey HSD (다군 비교), Paired t-test + Cohen's d (전후 비교), Bootstrap 95% CI (강건성 검증)를 적용합니다.

```python
# 시드 고정 예시
from src.utils.seed import set_seed
set_seed(42)
```

---

## 개발 환경 설정

```bash
# 린터 및 포매터 실행
uv run ruff check src/
uv run ruff format src/

# 타입 검사
uv run mypy src/

# 테스트 실행
uv run pytest tests/ -v --cov=src
```

---

## 라이선스

MIT License

---

## 작업 로그

상세 작업 내역 및 실험 기록은 [`docs/phase1_work_log.md`](docs/phase1_work_log.md)를 참조하세요.
