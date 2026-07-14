# 프로젝트 개요: Medical VQA VLM

## 미션

소비자급 GPU(16GB VRAM) 환경에서 경량 Vision-Language Model(VLM)을 의료 영상 Visual Question Answering(VQA) 도메인에 적응시키고, LLM 에이전트 기반 자율 하이퍼파라미터 최적화(HPO) 방법론의 효과를 실증하는 석사학위 연구 프로젝트.

## 비전

- 고가의 GPU 인프라 없이도 의료 AI 연구가 가능함을 입증
- QLoRA 파인튜닝을 통한 도메인 적응의 체계적 분석 프레임워크 제공
- autoresearch 패턴 기반 자율 HPO가 기존 방법(Random Search, Optuna TPE)과 비교하여 경쟁력 있음을 검증

## 논문 정보

- **대학교**: 건국대학교 정보통신대학원 융합정보기술학과 인공지능전공
- **제출 목표**: 2026년 9월
- **연구자**: 황태욱 (Taeuk Hwang)
- **라이선스**: MIT

---

## 핵심 사용자 및 시나리오

### 연구자 (본인)

- 석사학위 논문 실험 수행 및 결과 분석
- 3단계 실험 파이프라인(Phase 1-3) 순차 실행
- 실험 재현성 보장 (고정 시드, 버전 관리)

### 학계 및 후속 연구자

- 소비자 GPU 환경에서의 VLM 의료 적응 사례 참고
- 실험 설정 및 코드 재현
- autoresearch HPO 방법론 활용

---

## 핵심 기능 및 실험 구조

### Phase 1: 제로샷 베이스라인 평가 (RQ1)

- 3개 경량 VLM(Qwen3-VL-2B, Qwen2.5-VL-3B, SmolVLM-2.2B)의 의료 VQA 제로샷 성능 측정
- 3개 데이터셋(PathVQA, SLAKE, VQA-RAD) x 3개 시드(42, 123, 456)
- 측정 지표: Closed/Open accuracy, BERTScore F1, 응답 시간, VRAM 사용량
- 통계 검증: ANOVA + Tukey HSD

### Phase 2: QLoRA 파인튜닝 (RQ2)

- NF4 양자화 + LoRA 기반 파라미터 효율적 파인튜닝
- Ablation Study 3종: 데이터 크기, LoRA Rank, Target Module
- Catastrophic Forgetting 분석
- 통계 검증: Paired t-test, Cohen's d, Wilcoxon signed-rank

### Phase 3: 자율 하이퍼파라미터 최적화 (RQ3)

- 4가지 HPO 전략 비교: Manual, Random Search, Optuna TPE, Autoresearch (LLM Agent)
- Claude API 기반 자율 탐색 에이전트 (temperature scheduling, 구조화된 프롬프트)
- 15분 고정 시간 예산 per 실험, 최대 40회 반복
- 탐색 효율성 및 궤적 분석

---

## 연구 질문 (Research Questions)

| # | 연구 질문 | 귀무가설 |
|---|----------|----------|
| RQ1 | 경량 VLM(2-3B)의 의료 VQA 제로샷 성능은 모델별로 유의미한 차이가 있는가? | H0: 모델 간 VQA 정확도 차이 없음 |
| RQ2 | QLoRA 파인튜닝이 의료 VQA 성능을 유의미하게 향상시키는가? | H0: Base = Fine-tuned 성능 |
| RQ3 | 자율 하이퍼파라미터 탐색이 기존 HPO 방법보다 효율적인가? | H0: 자율 탐색 = Random Search |

---

## 비즈니스 목표 및 성공 지표

### 학술 성공 지표

- 석사학위 논문 심사 통과 (2026년 9월)
- 3개 연구 질문에 대한 통계적으로 유의미한 결과 도출
- 모든 실험의 재현성 보장 (코드, 시드, 환경 공개)

### 기술 성공 지표

- Phase 1 베이스라인: 9개 조건(3 모델 x 3 데이터셋) 완료 -- **진행 중 (Qwen2.5-VL-3B, Qwen3-VL-2B 완료)**
- Phase 2 파인튜닝: Base 대비 유의미한 성능 향상 달성
- Phase 3 HPO: autoresearch가 Random Search 대비 동등 이상 효율성 달성

---

## 현재 진행 상황

- Phase 1 베이스라인 평가: Qwen2.5-VL-3B (9/9 완료), Qwen3-VL-2B (9/9 완료), SmolVLM-2.2B (3/9 진행 중)
- Phase 2, Phase 3: 미착수 (Phase 1 완료 후 순차 진행)
- v0.2 설계서 요구사항 구현 완료 (BERTScore, CF 측정, max_steps, GPU time 분리)
