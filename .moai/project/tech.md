# 기술 스택: Medical VQA VLM

## 기술 스택 명세

### 언어 및 런타임

| 항목 | 버전 | 비고 |
|------|------|------|
| Python | >= 3.11 | pyproject.toml에 명시 |
| CUDA | 12.8 | PyTorch cu128 인덱스 사용 |

### 핵심 ML 프레임워크

| 라이브러리 | 버전 | 용도 |
|------------|------|------|
| PyTorch | >= 2.1.0 | 딥러닝 프레임워크 (CUDA 12.8) |
| torchvision | >= 0.16.0 | 이미지 전처리 |
| transformers | >= 4.45.0 | VLM 모델 로딩 및 추론 |
| accelerate | >= 0.34.0 | 분산/혼합 정밀도 학습 지원 |

### QLoRA / PEFT

| 라이브러리 | 버전 | 용도 |
|------------|------|------|
| peft | >= 0.13.0 | LoRA/QLoRA 어댑터 |
| bitsandbytes | >= 0.44.0 | NF4 양자화 |
| trl | >= 0.12.0 | SFTTrainer (Supervised Fine-Tuning) |
| unsloth | >= 2025.3.0 | Qwen 모델 가속 학습 (선택적, cu128) |

### 데이터 및 평가

| 라이브러리 | 버전 | 용도 |
|------------|------|------|
| datasets | >= 3.0.0 | HuggingFace 데이터셋 로딩 |
| Pillow | >= 10.0.0 | 이미지 처리 |
| bert-score | >= 0.3.13 | BERTScore F1 (roberta-large 기반) |
| scikit-learn | >= 1.5.0 | 통계 분석 |
| scipy | >= 1.14.0 | 통계 검증 (ANOVA, t-test, Wilcoxon) |
| nltk | >= 3.9.0 | 자연어 처리 유틸리티 |
| num2words | >= 0.5.13 | 숫자-텍스트 변환 |

### HPO (하이퍼파라미터 최적화)

| 라이브러리 | 버전 | 용도 |
|------------|------|------|
| optuna | >= 4.0.0 | Bayesian HPO (TPE) |
| anthropic | (런타임) | Claude API 기반 자율 HPO 에이전트 |

### 유틸리티

| 라이브러리 | 버전 | 용도 |
|------------|------|------|
| pyyaml | >= 6.0 | YAML 설정 파싱 |
| omegaconf | >= 2.3.0 | 구조화된 설정 관리 |
| tqdm | >= 4.66.0 | 진행률 표시 |
| pandas | >= 2.2.0 | 결과 데이터 분석 |
| matplotlib | >= 3.9.0 | 시각화 |
| seaborn | >= 0.13.0 | 통계 시각화 |
| wandb | >= 0.18.0 | 실험 추적 |
| einops | >= 0.8.2 | 텐서 연산 |
| qwen-vl-utils | >= 0.0.8 | Qwen VL 모델 유틸리티 |
| timm | >= 1.0.0 | Florence-2 비전 타워 |

---

## 개발 환경

### 빌드 및 패키지 관리

- **빌드 시스템**: setuptools >= 68.0 + wheel
- **패키지 매니저**: uv (uv.lock 사용)
- **PyTorch 설치**: 별도 인덱스 (https://download.pytorch.org/whl/cu128)

### 개발 도구

| 도구 | 버전 | 용도 |
|------|------|------|
| ruff | >= 0.6.0 | 린터 + 포매터 (line-length: 88, target: py311) |
| mypy | >= 1.11.0 | 정적 타입 체크 |
| pytest | >= 8.0.0 | 테스트 프레임워크 |
| pytest-cov | >= 5.0.0 | 커버리지 측정 |

### ruff 설정

- line-length: 88
- target-version: py311
- lint rules: E, F, I, W

### pytest 설정

- testpaths: tests/
- addopts: -v --tb=short

---

## 실험 환경 (하드웨어)

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA RTX 5060 Ti (16GB VRAM) |
| CPU | AMD Ryzen 5 5600X |
| RAM | 32GB |
| OS | Windows 11 Pro |
| VRAM 예산 | QLoRA 학습 시 ~8-10GB per 모델 |

---

## 테스트 전략

### 현재 상태

- `tests/florence2_step_test.py`: Florence-2 모델 스텝 테스트
- 커버리지 목표: 추후 확장 예정

### 대상 테스트 유형

- **단위 테스트**: 평가 메트릭 함수 (metrics.py), 데이터 전처리
- **통합 테스트**: 모델 로딩-추론-평가 파이프라인
- **실험 재현성 테스트**: 동일 시드로 동일 결과 산출 검증

---

## 재현성 보장

| 항목 | 방법 |
|------|------|
| 코드 관리 | GitHub 리포지토리 |
| 환경 재현 | pyproject.toml + uv.lock + CUDA 버전 명시 |
| 실험 추적 | Weights & Biases + results/ JSON |
| 랜덤 시드 | 모든 실험에 seed 고정 (42, 123, 456) |
| 반복 실험 | 각 조건 최소 3회 반복 |
| 데이터 버전 | 데이터셋 버전 및 다운로드 URL 명시 |
| Git 커밋 | 각 실험 설정을 git commit으로 추적 |

---

## 기술 제약 및 고려사항

- **VRAM 제한**: 16GB로 인해 4-bit 양자화(NF4) 필수, 풀 파인튜닝 불가
- **Windows 환경**: .bat 스크립트 사용, 일부 Linux 전용 도구 제한
- **단일 GPU**: 분산 학습 불가, gradient accumulation으로 effective batch size 확보
- **시간 예산**: Phase 3 HPO는 실험당 15분 고정 (TimeBudgetCallback 구현)
- **API 의존성**: Phase 3 autoresearch는 Anthropic Claude API 필요 (ANTHROPIC_API_KEY)
- **Unsloth 호환성**: Qwen 모델만 지원, SmolVLM 등은 표준 HF PEFT 폴백
