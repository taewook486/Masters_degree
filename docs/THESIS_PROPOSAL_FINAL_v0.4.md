# 석사학위 논문 설계서

> **Version**: v0.4 (2026-05-15)

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v0.1 | 2026-03-22 | 교수 제출용 초안 (3개 모델 기준) |
| v0.2 | 2026-03-24 | 리뷰 피드백 반영: Phase 2 평가 지표 정교화(BERTScore, CF 대조군), Phase 3 탐색 공간 개선(max_steps), 시간 예산 정의 명확화 |
| v0.3 | 2026-04-05 | Gemma 4 E2B 모델 추가 (4개 모델), transformers 5.5.0, 실험 조건 수 업데이트 |
| v0.4 | 2026-05-15 | 동료 심사 피드백 반영: RQ3 귀무가설 격상(Optuna), Phase 3 반복 10회, 하드웨어 이원화, BioBERT 병기, cross-dataset CF 추가, max_steps 고정, LLM 재현성 조치, 데이터 오염 한계점 추가 |

## 논문 정보

- **대학교**: 건국대학교 정보통신대학원 융합정보기술학과 인공지능전공
- **제출 목표**: 2026년 9월
- **실험 환경**:
  - **로컬**: RTX 5060 Ti (16GB VRAM), Ryzen 5 5600X, RAM 32GB — 16GB 재현성 검증용
  - **클라우드**: RunPod RTX 4090 (24GB VRAM) — 주 실험 환경

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

1. 소비자 GPU(16-24GB) 환경에서 경량 VLM의 의료 VQA 도메인 적응 가능성 실증
2. QLoRA 파인튜닝의 하이퍼파라미터가 성능에 미치는 영향 체계적 분석
3. autoresearch 스타일 자율 실험 루프를 통한 하이퍼파라미터 최적화 방법론 제안

---

## 3. 연구 질문

| # | 연구 질문 | 귀무가설 |
|---|----------|----------|
| RQ1 | 경량 VLM(2-3B)의 의료 VQA 제로샷 성능은 모델별로 유의미한 차이가 있는가? | H0: 모델 간 VQA 정확도 차이 없음 |
| RQ2 | QLoRA 파인튜닝이 의료 VQA 성능을 유의미하게 향상시키는가? | H0: Base = Fine-tuned 성능 |
| RQ3 | LLM 에이전트 기반 자율 하이퍼파라미터 탐색이 베이지안 최적화(Optuna TPE)와 경쟁적 성능을 달성하면서 해석 가능한 탐색 근거를 제공하는가? | H0: Autoresearch = Optuna (TPE) |

> **v0.4 변경 (RQ3)**: 귀무가설을 `= Random Search`에서 `= Optuna(TPE)`로 격상. LLM 기반 HPO의 차별점은 성능 경쟁력뿐 아니라, (1) 자연어 기반 탐색 근거 설명 가능성, (2) 사전 지식(cross-domain transfer)을 활용한 탐색 공간 구조 이해, (3) 설정 변경 이유의 추적 가능성에 있다. Random Search는 하한선(lower bound)으로만 유지한다.

---

## 4. 실험 설계

### 4.1 대상 모델 (4개)

| 모델 | 파라미터 | 아키텍처 특징 | 예상 QLoRA VRAM |
|------|---------|-------------|:---:|
| Qwen3-VL-2B | 2B | Thinking mode, DeepStack | ~8-10 GB |
| Qwen2.5-VL-3B | 3B | Dynamic Resolution, 19개 언어 OCR | ~8-10 GB |
| SmolVLM-2.2B | 2.2B | HuggingFace 경량 VLM | ~8-10 GB |
| Gemma4-E2B | 2.3B (active) / 5.1B (total) | PLE(Per-Layer Embeddings), Apache 2.0 | ~12-14 GB |

> **v0.3 변경**: Gemma 4 E2B 추가. PLE 기술로 2.3B active 파라미터만으로 5.1B급 표현력 제공. 제로샷 VRAM ~10.3GB 확인.

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

**기본 QLoRA 설정**:

| 파라미터 | 값 |
|----------|-----|
| Quantization | NF4 (4-bit NormalFloat) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, v_proj |
| Learning Rate | 2e-4 |
| Batch Size | 1 (gradient accumulation 8) |
| Epochs | 3 |
| Optimizer | paged_adamw_8bit |

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
- Open-ended accuracy: Exact Match + BERTScore F1 이중 보고
  - 범용 기준: roberta-large (threshold >= 0.7) — 선행 연구 비교용
  - 의료 특화 기준: BioBERT (dmis-lab/biobert-v1.1) — 의료 용어 민감도 향상
  - 두 지표 간 상관관계 분석 보고
- 훈련 시간 (총 분)
- 훈련 가능 파라미터 비율 (%)
- Peak VRAM 사용량 (MB)
- Catastrophic Forgetting 측정 (아래 상세)

**Catastrophic Forgetting 측정 방법**:

(A) 범용 VQA 성능 변화:
- 대조군: VQAv2 validation subset (2,000 샘플, 균형 샘플링)
- 측정 시점: 파인튜닝 전(Base) 1회 + 파인튜닝 후(QLoRA) 1회
- 지표: 범용 VQA 정확도 감소율 = (Base - Fine-tuned) / Base x 100%
- 적용 범위: 12개 조건(4모델 x 3데이터셋) 각각에 대해 측정

(B) 의료 도메인 내 cross-dataset 일반화:
- 훈련 데이터셋 ≠ 평가 데이터셋 조합으로 일반화 능력 측정
- 예: PathVQA로 훈련 → SLAKE, VQA-RAD로 평가 (각 조합 측정)
- 지표: cross-dataset 정확도 변화율
- 적용 범위: 12개 조건 x 2개 cross-dataset = 24회 추가 평가

> **v0.4 변경**: CF 측정을 (A) 범용 VQA + (B) 의료 cross-dataset 이중 구조로 확장. 의료 도메인 내 일반화 능력을 직접 측정하여 CF의 실질적 영향을 파악한다.

**통계 검증**:
- Paired t-test (Base vs Fine-tuned)
- Effect size: Cohen's d
- Wilcoxon signed-rank test (비모수 검증)

### 4.5 Phase 3: 자율 하이퍼파라미터 최적화 (RQ3)

**목적**: autoresearch 패턴 기반 자율 HPO의 효과 검증 — Optuna(TPE) 대비 경쟁적 성능 및 해석 가능성 평가

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

**비교 대상 (4가지 HPO 전략)**:

| 전략 | 설명 | trial 수 |
|------|------|:---:|
| Manual | 연구자가 직접 설정한 기본값 | 1 |
| Random Search | 탐색 공간에서 무작위 샘플링 | ~40 |
| Optuna (TPE) | 베이지안 최적화 (Tree-structured Parzen Estimator) | ~40 |
| Autoresearch | LLM 에이전트 기반 자율 탐색 | ~40 |

**자율 탐색 루프 (autoresearch 스타일)**:
1. 에이전트가 이전 실험 결과(results.tsv) 읽기
2. 다음 실험 설정 제안 + 자연어로 변경 근거 기록 (config.yaml + rationale.md)
3. Git commit
4. 고정 max_steps 학습 실행
5. 검증 세트 평가 -> VQA accuracy 기록
6. 성능 향상 시 keep, 그렇지 않으면 discard
7. 반복 (40회 또는 하룻밤)

**고정 조건**:
- 동일 모델 (Phase 2에서 최적 모델 1개)
- 동일 데이터셋 (PathVQA)
- 고정 학습량: max_steps 고정 (200 steps) — trial 간 동일 학습 step 수 보장
- 동일 검증 세트

**예상 실험 규모**:
- 총 trial 수: Manual 10회 + RS 400회(40x10) + Optuna 400회(40x10) + Autoresearch 400회(40x10) = 1,210회
- trial당 소요: ~10분 (max_steps=200 학습 + 검증 평가, RTX 4090 기준)
- 예상 총 GPU 시간: ~200 GPU-hours (RunPod RTX 4090 기준 약 8-9일, 24h 가동)

> **v0.4 변경**: "15분 고정 시간 예산"에서 "max_steps 고정"으로 변경. batch_size × grad_accum 조합에 따라 동일 시간 내 처리 데이터량이 달라지는 confounding을 해소. effective_batch_size(= batch × grad_accum)와 total_samples_seen(= steps × effective_batch), wall_clock_time을 별도 보고하여 데이터 처리량 차이를 투명하게 공개한다.

**측정 지표**:
- 최종 최적 VQA accuracy
- 최적 도달까지 실험 횟수
- 총 소요 시간 (wall-clock)
- 탐색 효율성 (최적 성능 / 총 실험 수)
- 탐색 궤적 시각화 (실험 번호 vs accuracy 그래프)
- effective_batch_size, total_samples_seen 보고
- Autoresearch: 각 trial별 자연어 탐색 근거 로그

**통계 검증**:
- Kruskal-Wallis test (4개 전략 간 최종 성능)
- 각 전략 10회 독립 반복 -> 분포 비교
- Bootstrap confidence interval
- Mann-Whitney U test: Autoresearch vs Optuna 쌍별 비교

> **v0.4 변경**: 반복 횟수 5회 → 10회. 통계적 검정력(statistical power)을 ~0.6-0.7 수준으로 확보. Autoresearch vs Optuna 쌍별 비교를 위한 Mann-Whitney U test 추가.

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
  - 2.3.3 기존 연구 성과 (LLaVA-Med, Med-Flamingo, CheXagent 등 의료 특화 VLM 포함)
- 2.4 자율 하이퍼파라미터 최적화
  - 2.4.1 전통적 HPO (Grid, Random, Bayesian)
  - 2.4.2 LLM 에이전트 기반 최적화 (autoresearch)
    - 베이지안 최적화 대비 LLM HPO의 이론적 차별점: cross-domain transfer, 탐색 공간 구조 이해, 자연어 기반 설명 가능성
    - 조기 학습 신호 기반 HPO의 이론적 정당성: Hyperband(Li et al., 2018)의 successive halving, 초반 학습 곡선과 최종 수렴 성능 간 순위 상관관계 (Phase 2 Ablation A 학습 곡선 결과로 본 연구 내 자체 실증)
- 2.5 선행 연구 요약 및 본 연구의 차별점

> **v0.4 변경**: 2.3.3에 의료 특화 VLM(LLaVA-Med, Med-Flamingo, CheXagent 등) 선행 연구 비교 추가. 2.4.2에 LLM HPO의 이론적 근거 및 Optuna(TPE) 대비 차별점 논의 추가.

### 제3장. 연구 방법 (약 15p)
- 3.1 연구 설계 개요
- 3.2 실험 환경 및 도구
  - 3.2.1 하드웨어 사양 (로컬 RTX 5060 Ti + 클라우드 RTX 4090 이원 구성)
  - 3.2.2 소프트웨어 스택 (HuggingFace, Optuna 등)
- 3.3 대상 모델 및 선정 기준
- 3.4 데이터셋 및 전처리
- 3.5 실험 1: 제로샷 베이스라인 평가
- 3.6 실험 2: QLoRA 파인튜닝
- 3.7 실험 3: 자율 하이퍼파라미터 최적화
- 3.8 평가 지표 및 통계 분석 방법
  - 3.8.1 BERTScore 이중 보고 (roberta-large + BioBERT)
  - 3.8.2 Catastrophic Forgetting 이중 측정 (VQAv2 + cross-dataset)

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
  - 4.2.5 Catastrophic Forgetting 분석 (VQAv2 + cross-dataset)
- 4.3 Phase 3: 자율 HPO 결과
  - 4.3.1 각 전략의 최적 성능 비교 (Autoresearch vs Optuna 중심)
  - 4.3.2 탐색 효율성 비교
  - 4.3.3 탐색 궤적 분석
  - 4.3.4 발견된 최적 하이퍼파라미터 조합
  - 4.3.5 Autoresearch 탐색 근거 해석 가능성 분석
- 4.4 종합 분석 및 논의

### 제5장. 결론 (약 5p)
- 5.1 연구 요약
- 5.2 연구 기여
- 5.3 한계점
  - 데이터 오염(Data Contamination): 2024-2025년 출시 모델의 사전훈련 데이터에 PathVQA, SLAKE, VQA-RAD가 포함되었을 가능성을 배제할 수 없음. 다만 Phase 2의 전후 비교는 동일 모델 내 성능 향상을 측정하므로 오염 영향이 상쇄됨
  - 의료 특화 VLM 미비교: LLaVA-Med, Med-Flamingo 등 의료 특화 모델과의 직접 비교는 본 연구 범위 외 (선행 연구 성능 수치를 통한 간접 비교만 수행)
  - Phase 3 confounding: max_steps 고정에도 effective_batch_size에 따른 total_samples_seen 차이 존재
  - LLM 비결정성: Autoresearch의 API 비결정성으로 인한 완전 재현 불가 (temperature=0, 모델 버전 고정, 응답 로깅으로 완화)
  - Ablation Study 일반화: LoRA Rank/Target Module/데이터 크기 Ablation은 PathVQA + 최적 모델 1개 기준으로 수행. 데이터셋 간 경향성 차이 검증은 향후 연구 과제
- 5.4 향후 연구 방향

> **v0.4 변경**: 5.3 한계점에 데이터 오염, 의료 특화 모델 비교 부재, confounding, LLM 비결정성을 명시적으로 서술.

### 참고문헌
### 부록
- A. 상세 실험 결과 표
- B. autoresearch 에이전트 program.md
- C. 실험 재현 가이드
- D. Autoresearch 탐색 근거 로그 (자연어 rationale 전문)

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
| LLM 비결정성 통제 | temperature=0, top_p=1, 모델 ID 및 스냅샷 날짜 고정 |
| API 응답 로깅 | 전체 API 요청/응답 JSON을 실험별 로그 파일로 저장 |
| 변동성 흡수 | Phase 3 각 전략 10회 반복으로 분포 보고 |

> **v0.4 변경**: LLM 비결정성 통제, API 응답 로깅, 변동성 흡수를 위한 10회 반복 항목 추가.

---

## 7. 예상 결과 테이블 (형식 예시)

### Table 4.1: 제로샷 베이스라인 결과 (VQA Accuracy %)

| 모델 | PathVQA (Open) | PathVQA (Closed) | SLAKE (Open) | SLAKE (Closed) | VQA-RAD (Open) | VQA-RAD (Closed) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | ?.?? +/- ?.?? | ?.?? +/- ?.?? | ... | ... | ... | ... |
| Qwen2.5-VL-3B | ... | ... | ... | ... | ... | ... |
| SmolVLM-2.2B | ... | ... | ... | ... | ... | ... |
| Gemma4-E2B | ... | ... | ... | ... | ... | ... |

### Table 4.2: QLoRA 파인튜닝 전후 비교

| 모델 | 조건 | PathVQA (EM) | PathVQA (BERTScore-roberta) | PathVQA (BERTScore-BioBERT) | SLAKE Acc. | VQA-RAD Acc. | 향상율 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | Zero-shot | ... | ... | ... | ... | ... | - |
| Qwen3-VL-2B | QLoRA | ... | ... | ... | ... | ... | +?.?% |
| ... | ... | ... | ... | ... | ... | ... | ... |

### Table 4.2b: Catastrophic Forgetting 측정

**(A) VQAv2 subset (범용 VQA)**

| 모델 | 학습 데이터셋 | Base Acc. | Fine-tuned Acc. | 감소율 (%) |
|------|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | PathVQA | ?.?? | ?.?? | ?.?% |
| Qwen3-VL-2B | SLAKE | ?.?? | ?.?? | ?.?% |
| ... | ... | ... | ... | ... |

**(B) 의료 도메인 내 cross-dataset 일반화**

| 모델 | 훈련 | 평가 | Base Acc. | Fine-tuned Acc. | 변화율 (%) |
|------|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-2B | PathVQA | SLAKE | ?.?? | ?.?? | ?.?% |
| Qwen3-VL-2B | PathVQA | VQA-RAD | ?.?? | ?.?? | ?.?% |
| ... | ... | ... | ... | ... | ... |

### Table 4.3: HPO 전략 비교 (PathVQA, 최적 모델)

| HPO 전략 | 최적 Accuracy | 실험 횟수 | 총 시간(h) | 효율성 | effective_batch 범위 | total_samples 범위 |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Manual (기본값) | ?.?? | 1 | ?.? | - | 8 | ?.?K |
| Random Search | ?.?? +/- ?.?? | 40 | ~?? | ?.?? | 4-64 | ?.?K-?.?K |
| Optuna (TPE) | ?.?? +/- ?.?? | 40 | ~?? | ?.?? | 4-64 | ?.?K-?.?K |
| Autoresearch | ?.?? +/- ?.?? | 40 | ~?? | ?.?? | 4-64 | ?.?K-?.?K |
