# 석사학위 논문 설계서

> **Version**: v0.3 (2026-04-05)

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v0.1 | 2026-03-22 | 초안 작성 (4개 모델, 기본 실험 설계) |
| v0.2 | 2026-03-24 | Florence-2 탈락 (3개 모델), Phase 2: BERTScore 추가 및 CF 대조군 구체화, Phase 3: epochs->max_steps 전환 및 시간 예산 정의 명확화 |
| v0.3 | 2026-04-05 | Gemma 4 E2B 모델 추가 (4개 모델), transformers 5.5.0 업그레이드, 실험 조건 수 업데이트 |

## 논문 정보

- **대학교**: 건국대학교 정보통신대학원 융합정보기술학과 인공지능전공
- **제출 목표**: 2026년 9월
- **실험 환경**: RTX 5060 Ti (16GB VRAM), Ryzen 5 5600X, RAM 32GB

---

## 1. 제목

**한국어**: 경량 멀티모달 모델의 의료 영상 VQA 도메인 적응: QLoRA 파인튜닝과 자율 하이퍼파라미터 최적화

**영문**: Domain Adaptation of Lightweight Vision-Language Models for Medical Visual Question Answering: QLoRA Fine-Tuning with Autonomous Hyperparameter Optimization

---

## 2. 연구 배경 및 목적

### 2.1 연구 배경

- Vision-Language Model(VLM)의 급속한 발전 (GPT-4V, Gemini 등)
- 의료 영상 분석에서 VLM 활용 가능성 증대
- 범용 VLM의 도메인 특화 성능 한계 (의학 용어, 전문 지식 부족)
- 대규모 GPU 없이도 QLoRA를 통한 효율적 파인튜닝 가능
- 하이퍼파라미터 선택의 자동화는 VLM PEFT의 미해결 과제 (NeurIPS 2024 서베이)

### 2.2 연구 목적

1. 소비자 GPU(16GB) 환경에서 경량 VLM의 의료 VQA 도메인 적응 가능성 실증
2. QLoRA 파인튜닝의 하이퍼파라미터가 성능에 미치는 영향 체계적 분석
3. autoresearch 스타일 자율 실험 루프를 통한 하이퍼파라미터 최적화 방법론 제안

---

## 3. 연구 질문

| # | 연구 질문 | 귀무가설 |
|---|----------|----------|
| RQ1 | 경량 VLM(2-3B)의 의료 VQA 제로샷 성능은 모델별로 유의미한 차이가 있는가? | H0: 모델 간 VQA 정확도 차이 없음 |
| RQ2 | QLoRA 파인튜닝이 의료 VQA 성능을 유의미하게 향상시키는가? | H0: Base = Fine-tuned 성능 |
| RQ3 | 자율 하이퍼파라미터 탐색이 기존 HPO 방법보다 효율적인가? | H0: 자율 탐색 = Random Search |

---

## 4. 실험 설계

### 4.1 대상 모델 (4개)

| 모델 | 파라미터 | 아키텍처 특징 | 예상 QLoRA VRAM |
|------|---------|-------------|:---:|
| Qwen3-VL-2B | 2B | Thinking mode, DeepStack | ~8-10 GB |
| Qwen2.5-VL-3B | 3B | Dynamic Resolution, 19개 언어 OCR | ~8-10 GB |
| SmolVLM-2.2B | 2.2B | HuggingFace 경량 VLM | ~8-10 GB |
| Gemma4-E2B | 2.3B (active) / 5.1B (total) | PLE(Per-Layer Embeddings), Apache 2.0 | ~12-14 GB |

> **v0.2 변경**: Florence-2-large 탈락. transformers 5.x와의 호환성 문제로 실험 환경에서 정상 동작하지 않음.
> **v0.3 변경**: Gemma 4 E2B 추가. PLE 기술로 2.3B active 파라미터만으로 5.1B급 표현력 제공. 제로샷 VRAM ~10.3GB 확인 (호환성 테스트 4/4 PASS).

**선정 기준**:
- 16GB VRAM에서 QLoRA 파인튜닝 가능
- Apache 2.0 또는 MIT 라이선스 (연구 활용 자유)
- 충분한 커뮤니티/프레임워크 지원

### 4.2 데이터셋 (3개)

| 데이터셋 | 이미지 수 | QA 쌍 | 언어 | 도메인 | 질문 유형 |
|----------|:---:|:---:|:---:|:---:|:---:|
| PathVQA | 4,998 | 32,799 | 영어 | 병리학 | Open+Closed (7종) |
| SLAKE | 642 | 14,028 | 영어+중국어 | 방사선/CT | Open+Closed |
| VQA-RAD | 315 | 2,248 | 영어 | 방사선 | Open+Closed |

**데이터 분할**: 각 데이터셋의 공식 train/val/test split 사용

### 4.3 Phase 1: 베이스라인 평가 (RQ1)

**목적**: 파인튜닝 전 각 모델의 의료 VQA 제로샷 성능 측정

**실험 조건**:
- 4개 모델 x 3개 데이터셋 = 12개 조건
- 각 조건 3회 반복 (프롬프트 순서 shuffle, seed: 42, 123, 456)
- 동일 프롬프트 템플릿 사용

**프롬프트 템플릿**:
```
You are a medical AI assistant. Look at this medical image and answer the following question.
Question: {question}
Answer concisely.
```

**측정 지표**:
- Closed-ended accuracy (Yes/No, 선택형)
- Open-ended accuracy (정답 토큰 매칭)
- 응답 시간 (ms/question)
- VRAM 사용량 (peak MB)

**통계 검증**:
- ANOVA (4개 모델 간 성능 차이)
- 사후 검증: Tukey HSD
- 유의수준: alpha = 0.05

### 4.4 Phase 2: QLoRA 파인튜닝 (RQ2)

**목적**: 도메인 특화 파인튜닝의 효과 측정

**기본 QLoRA 설정** (Phase 2 기본값):
```yaml
quantization: nf4 (4-bit NormalFloat)
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_targets: [q_proj, v_proj]
learning_rate: 2e-4
batch_size: 1
gradient_accumulation_steps: 8
epochs: 3
optimizer: paged_adamw_8bit
max_seq_length: 2048
warmup_ratio: 0.03
weight_decay: 0.01
```

**실험 조건**:
- 4개 모델 x 3개 데이터셋 = 12개 조건
- 각 조건 3회 반복 (seed 변경: 42, 123, 456)

**Ablation Study A - 데이터 크기 영향**:
- PathVQA 기준, 최적 모델 1개 선택
- 훈련 데이터 비율: 5%, 10%, 25%, 50%, 100%
- 학습 곡선(learning curve) 분석

**Ablation Study B - LoRA Rank 영향**:
- PathVQA 기준, 최적 모델 1개 선택
- LoRA rank: 4, 8, 16, 32, 64
- 성능 vs 파라미터 효율성 분석

**Ablation Study C - Target Module 영향**:
- [q_proj, v_proj] vs [q_proj, k_proj, v_proj, o_proj] vs [all_linear]
- 성능 vs 학습 시간 트레이드오프

**측정 지표** (Phase 1과 동일 + 추가):
- Open-ended accuracy: Exact Match + **BERTScore F1** (threshold >= 0.7, roberta-large 기반)
- 훈련 시간 (총 분)
- 훈련 가능 파라미터 비율 (%)
- Peak VRAM 사용량 (MB, `src/utils/vram_monitor.py`로 실시간 추적)
- Catastrophic Forgetting 측정 (아래 상세)

> **v0.2 추가**: BERTScore는 'cancer' vs 'malignancy' 등 의미적으로 동일하나 표현이 다른 응답을 올바르게 평가하기 위함.

**Catastrophic Forgetting 측정 방법**:
- 대조군: VQAv2 validation subset (2,000 샘플, 균형 샘플링)
- 측정 시점: 파인튜닝 전(Base) 1회 + 파인튜닝 후(QLoRA) 1회
- 지표: 범용 VQA 정확도 감소율 = (Base - Fine-tuned) / Base x 100%
- 적용 범위: 12개 조건(4모델 x 3데이터셋) 각각에 대해 측정
- Phase 3에서는 최종 best config에서만 측정 (trial 중 매회 측정 시 시간 비용 과다)

**통계 검증**:
- Paired t-test (Base vs Fine-tuned)
- Effect size: Cohen's d
- Wilcoxon signed-rank test (비모수 검증)

### 4.5 Phase 3: 자율 하이퍼파라미터 최적화 (RQ3)

**목적**: autoresearch 패턴 기반 자율 HPO의 효과 검증

**탐색 공간**:
| 파라미터 | 탐색 범위 | 타입 |
|----------|----------|------|
| lora_rank | {4, 8, 16, 32, 64} | 이산 |
| lora_alpha | rank x {1, 2, 4} | 이산 |
| learning_rate | [1e-5, 5e-4] | 연속 (로그스케일) |
| batch_size | {1, 2, 4} | 이산 |
| grad_accum_steps | {4, 8, 16} | 이산 |
| warmup_ratio | [0.0, 0.1] | 연속 |
| weight_decay | [0.0, 0.1] | 연속 |
| lora_targets | {minimal, medium, full} | 범주형 |
| max_steps | {100, 200, 400, 800} | 이산 |

> **v0.2 변경**: `epochs {1, 2, 3, 5}` -> `max_steps {100, 200, 400, 800}`으로 전환.
> 고정 시간 예산(15분) 내에서 epochs 수의 차이가 유의미하지 않을 수 있으므로, step 단위로 학습량을 직접 제어하여 전략 간 공정한 비교를 보장함. Phase 2는 기존 epochs 기반 유지.

**비교 대상 (4가지 HPO 전략)**:

| 전략 | 설명 | 하룻밤 실험 수 |
|------|------|:---:|
| Manual | 연구자가 직접 설정한 기본값 | 1 |
| Random Search | 탐색 공간에서 무작위 샘플링 | ~40 |
| Optuna (TPE) | 베이지안 최적화 (Tree-structured Parzen Estimator) | ~40 |
| **Autoresearch** | LLM 에이전트 기반 자율 탐색 | ~40 |

**자율 탐색 루프 (autoresearch 스타일)**:
```
1. 에이전트가 이전 실험 결과(results.tsv) 읽기
2. 다음 실험 설정 제안 (config.yaml 수정)
3. Git commit
4. 15분 고정 시간 학습 실행 (순수 GPU 학습 시간)
5. 검증 세트 평가 -> VQA accuracy 기록
6. 성능 향상 시 keep, 그렇지 않으면 discard
7. 반복 (40회 또는 하룻밤)
```

**고정 조건**:
- 동일 모델 (Phase 2에서 최적 모델 1개)
- 동일 데이터셋 (PathVQA)
- 고정 시간 예산: 15분/실험 (**순수 GPU 학습 시간**, 모델 로드/평가 시간 제외)
- 동일 검증 세트

> **v0.2 명확화**: 15분 예산은 `trainer.train()` 실행 시간만을 의미함.
> 에이전트의 설정 제안 시간(~30초), 모델 로딩(~1분), 평가(~5분)는 별도 기록하되 예산에 불포함.
> 실제 벽시간(wall-clock)은 trial당 약 20-25분 소요 예상.

**측정 지표**:
- 최종 최적 VQA accuracy
- 최적 도달까지 실험 횟수
- 총 소요 시간 (wall-clock)
- 탐색 효율성 (최적 성능 / 총 실험 수)
- 탐색 궤적 시각화 (실험 번호 vs accuracy 그래프)

**통계 검증**:
- Kruskal-Wallis test (4개 전략 간 최종 성능)
- 각 전략 5회 독립 반복 -> 분포 비교
- Bootstrap confidence interval

---

## 5. 논문 구조 (장별 구성)

### 제1장. 서론 (약 5p)
- 1.1 연구 배경
- 1.2 연구 목적
- 1.3 연구 범위 및 제한
- 1.4 논문 구성

### 제2장. 이론적 배경 (약 15p)
- 2.1 Vision-Language Model 개요
  - 2.1.1 멀티모달 학습의 발전
  - 2.1.2 경량 VLM 아키텍처 (Qwen-VL, SmolVLM 등)
- 2.2 Parameter-Efficient Fine-Tuning
  - 2.2.1 LoRA (Low-Rank Adaptation)
  - 2.2.2 QLoRA (Quantized LoRA)
  - 2.2.3 기타 PEFT 기법 비교
- 2.3 의료 영상 Visual Question Answering
  - 2.3.1 Medical VQA 과제 정의
  - 2.3.2 주요 벤치마크 데이터셋
  - 2.3.3 기존 연구 성과
- 2.4 자율 하이퍼파라미터 최적화
  - 2.4.1 전통적 HPO (Grid, Random, Bayesian)
  - 2.4.2 LLM 에이전트 기반 최적화 (autoresearch)
- 2.5 선행 연구 요약 및 본 연구의 차별점

### 제3장. 연구 방법 (약 15p)
- 3.1 연구 설계 개요
- 3.2 실험 환경 및 도구
  - 3.2.1 하드웨어 사양
  - 3.2.2 소프트웨어 스택 (HuggingFace, Optuna 등)
- 3.3 대상 모델 및 선정 기준
- 3.4 데이터셋 및 전처리
- 3.5 실험 1: 제로샷 베이스라인 평가
- 3.6 실험 2: QLoRA 파인튜닝
- 3.7 실험 3: 자율 하이퍼파라미터 최적화
- 3.8 평가 지표 및 통계 분석 방법

### 제4장. 실험 결과 및 분석 (약 20p)
- 4.1 Phase 1: 제로샷 베이스라인 결과
  - 4.1.1 모델별 성능 비교
  - 4.1.2 데이터셋별 난이도 분석
  - 4.1.3 오류 유형 분석
- 4.2 Phase 2: QLoRA 파인튜닝 결과
  - 4.2.1 Base vs Fine-tuned 성능 향상
  - 4.2.2 데이터 크기 영향 (Ablation A)
  - 4.2.3 LoRA Rank 영향 (Ablation B)
  - 4.2.4 Target Module 영향 (Ablation C)
  - 4.2.5 Catastrophic Forgetting 분석
- 4.3 Phase 3: 자율 HPO 결과
  - 4.3.1 각 전략의 최적 성능 비교
  - 4.3.2 탐색 효율성 비교
  - 4.3.3 탐색 궤적 분석
  - 4.3.4 발견된 최적 하이퍼파라미터 조합
- 4.4 종합 분석 및 논의

### 제5장. 결론 (약 5p)
- 5.1 연구 요약
- 5.2 연구 기여
- 5.3 한계점
- 5.4 향후 연구 방향

### 참고문헌
### 부록
- A. 상세 실험 결과 표
- B. autoresearch 에이전트 program.md
- C. 실험 재현 가이드

---

## 6. 재현성 보장 계획

| 항목 | 방법 |
|------|------|
| 코드 관리 | GitHub 리포지토리 (실험 코드 + 설정 파일) |
| 환경 재현 | pyproject.toml + CUDA 버전 명시 |
| 실험 추적 | Weights & Biases + results.tsv |
| 랜덤 시드 | 모든 실험에 seed 고정 (42, 123, 456) |
| 반복 실험 | 각 조건 최소 3회 반복 -> 평균 +/- 표준편차 보고 |
| 데이터 버전 | 데이터셋 버전 및 다운로드 URL 명시 |
| Git 커밋 | 각 실험 설정을 git commit으로 추적 |

---

## 7. 예상 결과 테이블 (형식 예시)

### Table 4.1: 제로샷 베이스라인 결과 (VQA Accuracy %)

| 모델 | PathVQA (Open) | PathVQA (Closed) | SLAKE (Open) | SLAKE (Closed) | VQA-RAD (Open) | VQA-RAD (Closed) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | ?.?? +/- ?.?? | ?.?? +/- ?.?? | ... | ... | ... | ... |
| Qwen2.5-VL-3B | ... | ... | ... | ... | ... | ... |
| SmolVLM-2.2B | ... | ... | ... | ... | ... | ... |

### Table 4.2: QLoRA 파인튜닝 전후 비교

| 모델 | 조건 | PathVQA Acc. (EM) | PathVQA Acc. (BERTScore) | SLAKE Acc. | VQA-RAD Acc. | 향상율 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | Zero-shot | ... | ... | ... | ... | - |
| Qwen3-VL-2B | QLoRA | ... | ... | ... | ... | +?.?% |
| ... | ... | ... | ... | ... | ... | ... |

### Table 4.2b: Catastrophic Forgetting 측정 (VQAv2 subset)

| 모델 | 학습 데이터셋 | Base Acc. | Fine-tuned Acc. | 감소율 (%) |
|------|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | PathVQA | ?.?? | ?.?? | ?.?% |
| Qwen3-VL-2B | SLAKE | ?.?? | ?.?? | ?.?% |
| ... | ... | ... | ... | ... |

### Table 4.3: HPO 전략 비교 (PathVQA, 최적 모델)

| HPO 전략 | 최적 Accuracy | 실험 횟수 | 총 시간(h) | 효율성 |
|----------|:---:|:---:|:---:|:---:|
| Manual (기본값) | ?.?? | 1 | 0.25 | - |
| Random Search | ?.?? +/- ?.?? | 40 | ~10 | ?.?? |
| Optuna (TPE) | ?.?? +/- ?.?? | 40 | ~10 | ?.?? |
| **Autoresearch** | ?.?? +/- ?.?? | 40 | ~10 | ?.?? |

---

## 8. 프로젝트 디렉토리 구조

```
Masters_degree/
├── THESIS_PROPOSAL.md          # v0.1 원본
├── THESIS_PROPOSAL_v0.2.md     # 이 문서
├── THESIS_PROPOSAL_FINAL.md    # 교수 제출용
├── RISK_ANALYSIS.md            # 위험 요소 분석
├── configs/                    # 실험 설정 파일
│   ├── models/                 # 모델별 기본 설정
│   │   ├── qwen3_vl_2b.yaml
│   │   ├── qwen25_vl_3b.yaml
│   │   └── smolvlm_2b.yaml
│   ├── finetune/               # 파인튜닝 설정
│   │   ├── base_qlora.yaml     # 기본 QLoRA 설정
│   │   └── ablation/           # Ablation 설정들
│   └── autoresearch/           # 자율 탐색 설정
│       └── program.md          # 에이전트 지시문
│
├── data/                       # 데이터 (gitignore)
│   ├── pathvqa/
│   ├── slake/
│   ├── vqa_rad/
│   └── vqav2_subset/           # v0.2 추가: CF 측정용
│
├── src/                        # 소스 코드
│   ├── evaluate/               # 평가 스크립트
│   │   ├── metrics.py          # VQA accuracy, BERTScore 등
│   │   └── catastrophic_forgetting.py  # CF 측정 (v0.2 추가)
│   ├── finetune/               # 파인튜닝 스크립트
│   │   ├── train_qlora.py      # QLoRA 학습 메인
│   │   └── prepare_data.py     # 데이터 전처리
│   ├── baseline/               # 베이스라인 평가
│   │   └── evaluate_zero_shot.py
│   ├── autoresearch/           # 자율 탐색 파이프라인
│   │   ├── loop.py             # 메인 실험 루프
│   │   ├── agent.py            # LLM 에이전트 연동
│   │   ├── strategies.py       # HPO 전략 (Random, Optuna, Auto)
│   │   └── tracker.py          # 결과 추적 (results.tsv)
│   ├── data/                   # 데이터 로딩
│   │   ├── download.py         # 데이터셋 다운로드
│   │   └── general_vqa.py      # VQAv2 subset 로더 (v0.2 추가)
│   └── utils/                  # 유틸리티
│       ├── seed.py             # 시드 고정
│       └── vram_monitor.py     # VRAM 모니터링
│
├── results/                    # 실험 결과
│   ├── phase1_baseline/
│   ├── phase2_finetune/
│   ├── phase3_autoresearch/
│   └── figures/                # 그래프, 시각화
│
├── thesis/                     # 논문 원고
│   ├── main.tex                # 또는 main.docx
│   └── figures/
│
├── pyproject.toml
└── README.md
```
