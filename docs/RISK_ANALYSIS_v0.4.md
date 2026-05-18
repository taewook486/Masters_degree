# 위험 요소 분석 및 대응 방안

> **Version**: v0.4 (2026-05-15)

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v0.1 | 2026-03-22 | 초안 작성 (7개 위험 요소) |
| v0.2 | 2026-03-24 | Florence-2 탈락 반영, BERTScore/CF 관련 신규 위험 추가, Phase 3 시간 예산 위험 구체화 |
| v0.3 | 2026-04-05 | Gemma 4 E2B 모델 추가 반영 (4개 모델), VRAM 위험 업데이트 |
| v0.4 | 2026-05-15 | 동료 심사 피드백 반영: 하드웨어 이원화, BioBERT VRAM 충돌, cross-dataset CF, Phase 3 n=10 규모 증가, LLM 재현성 위험 추가 |

---

## 기술적 위험

### 위험 1: VRAM 부족 (OOM)
- **확률**: 중간 (특히 Qwen2.5-VL-3B, 의료 이미지 해상도 높을 때)
- **영향**: 해당 모델 실험 불가
- **대응**:
  1. 이미지 해상도 제한 (max_pixels 조절)
  2. batch_size=1 + gradient_accumulation 증가
  3. max_seq_length 축소 (2048 -> 1024)
  4. gradient_checkpointing 활성화

> **v0.2 변경**: Florence-2 대체 옵션 삭제 (이미 탈락). Phase 1 제로샷에서 Qwen2.5-VL-3B가 7,581MB 사용 확인됨. QLoRA 학습 시 12-14GB 예상으로, 16GB 내 운용 가능하나 peak 모니터링 필수.
> **v0.3 변경**: Gemma 4 E2B 추가. 제로샷 VRAM ~10,298MB (bfloat16). 4개 모델 중 VRAM 사용량 최대이므로, QLoRA 학습 시 OOM 가능성이 가장 높은 모델. batch_size=1 + gradient_checkpointing 우선 적용 대상.

### 위험 2: 의료 VQA 정확도가 너무 낮음 (제로샷)
- **확률**: 중간-높음 (경량 모델의 의료 지식 부족)
- **영향**: 파인튜닝 효과 해석 어려움
- **대응**:
  1. 오히려 논문의 기여: "제로샷 한계를 보이고 파인튜닝 필요성 입증"
  2. Closed-ended (Yes/No) 질문 위주 분석으로 보완
  3. 프롬프트 최적화로 제로샷 성능 개선 시도

### 위험 3: autoresearch 파이프라인 구축 난이도
- **확률**: 중간
- **영향**: Phase 3 실험 지연
- **대응**:
  1. 기존 autoresearch 코드 참고하여 최소 기능 구현 (modify-train-evaluate 루프)
  2. 복잡한 LLM 에이전트 대신 단순 규칙 기반 탐색도 비교군으로 포함
  3. 최악의 경우 Phase 3를 "Optuna vs Random Search" 비교로 축소 가능

### 위험 4: Catastrophic Forgetting
- **확률**: 낮음 (QLoRA는 파라미터 변경이 적어 영향 제한적)
- **영향**: 파인튜닝 모델의 범용 성능 크게 저하
- **대응**:
  1. VQAv2 validation subset (2,000 샘플)으로 사전/사후 성능 비교
  2. 의료 도메인 내 cross-dataset 일반화 측정 (훈련 데이터셋 ≠ 평가 데이터셋)
  3. 학습률 낮춤, epoch 수 제한
  4. NeurIPS 2025 논문의 simple recipe 적용

> **v0.2 변경**: 대조군을 "MMMU 등"에서 **VQAv2 validation subset (2,000 샘플)**로 구체화. 측정 시점도 "파인튜닝 전후 각 1회"로 명확화.
> **v0.4 변경**: 의료 cross-dataset 일반화 측정 추가 (12조건 x 2개 cross-dataset = 24회 추가 평가). 추가 소요: ~6 GPU-hours.

### 위험 8: BERTScore 의존성 및 VRAM 충돌 (v0.2 신규)
- **확률**: 낮음-중간
- **영향**: BERTScore 모델(roberta-large + BioBERT)이 VLM과 동시에 VRAM 점유 시 OOM
- **대응**:
  1. BERTScore 평가는 VLM 모델 unload 후 별도 실행
  2. CPU fallback 옵션 (`device="cpu"`) 제공
  3. Phase 3 trial 중에는 BERTScore 생략, 최종 best config에서만 측정

> **v0.4 변경**: BioBERT(dmis-lab/biobert-v1.1) 추가로 BERTScore 이중 보고. BioBERT는 roberta-large보다 경량(~110M)이므로 VRAM 부담 증가는 미미하나, 두 모델을 순차 실행하여 VRAM 충돌 방지.

### 위험 9: VQAv2 데이터셋 추가 준비 부담 (v0.2 신규)
- **확률**: 낮음 (공개 데이터셋)
- **영향**: CF 측정 지연
- **대응**:
  1. VQAv2는 HuggingFace datasets로 직접 접근 가능
  2. 2,000 샘플 subset만 사용하므로 저장/처리 부담 최소
  3. subset 선정 시 seed 고정하여 재현성 보장

### 위험 10: Phase 3 max_steps 기반 비교 공정성 (v0.2 신규)
- **확률**: 중간
- **영향**: 전략 간 비교의 유효성 저하
- **대응**:
  1. max_steps=200으로 고정 (모든 전략 동일)
  2. HuggingFace Trainer에서 max_steps > 0이면 epochs 무시 → 일관된 동작 보장
  3. effective_batch_size, total_samples_seen, wall-clock 시간을 별도 보고
  4. confounding variable(batch_size별 데이터 처리량 차이)은 한계점에 명시

> **v0.4 변경**: "15분 시간 예산"에서 "max_steps=200 고정"으로 변경. 시간이 아닌 학습 step 수로 통일하여 confounding 완화.

### 위험 12: Phase 3 실험 규모 증가 (v0.4 신규)
- **확률**: 중간
- **영향**: 실험 기간 장기화, RunPod 비용 증가
- **대응**:
  1. 총 ~1,210 trials, ~200 GPU-hours → RunPod RTX 4090 기준 약 8-9일
  2. RunPod Community Cloud 활용 시 ~$78-107
  3. RTX 4090 2대 병렬 시 ~4일로 단축 (비용 동일)
  4. 긴급 시 반복 횟수 10회 → 7회 축소 가능 (power ~0.5 유지)

### 위험 13: Autoresearch LLM API 비결정성 (v0.4 신규)
- **확률**: 높음 (LLM API 본질적 비결정성)
- **영향**: 실험 재현 불가능성
- **대응**:
  1. temperature=0, top_p=1 고정
  2. 모델 ID 및 스냅샷 날짜 고정
  3. 전체 API 요청/응답 JSON 로깅
  4. 10회 독립 반복으로 변동성을 통계적으로 흡수

---

## 일정 위험

### 위험 5: 실험 시간 초과
- **확률**: 중간
- **영향**: 논문 작성 기간 부족
- **대응**:
  1. 데이터셋 3개 -> 2개 축소 (VQA-RAD 제외, 규모 너무 작음)
  2. Ablation study 범위 축소
  3. Phase 3 반복 횟수 10회 → 7회 축소 가능
  4. Phase 3 trial 수 40 → 20회 축소 가능

> **v0.2 변경**: "모델 수 4개 -> 3개 축소" 옵션 삭제 (이미 3개로 확정). BERTScore + CF 추가로 Phase 2에 약 +2-3시간 추가 예상되나, Phase 3에서는 최종 best에만 적용하여 영향 최소화.
> **v0.3 변경**: Gemma 4 E2B 추가로 4개 모델 확정. Phase 1 추가 소요: ~5-6시간 (RunPod RTX 4090 기준). Phase 2 추가 소요: ~3-4시간 (QLoRA 학습). 시간 초과 시 축소 옵션: 모델 4개 → 3개 (Gemma4-E2B 제외).
> **v0.4 변경**: 전체 실험 규모 추정 — Phase 1(~9h) + Phase 2(~65h) + Phase 3(~200h) = ~274 GPU-hours. RunPod RTX 4090 기준 ~11-12일, 비용 ~$107 (Community Cloud).

### 위험 6: 지도교수 피드백으로 방향 변경
- **확률**: 중-높음
- **영향**: 연구 질문 또는 실험 설계 수정
- **대응**:
  1. 가능한 빨리 (4월 초) 지도교수님께 설계서 공유
  2. Phase 1 결과를 먼저 보여드리며 방향 확인
  3. 유연한 설계: 모델/데이터셋 교체가 코드 수정 최소로 가능하도록 모듈화

### 위험 11: 실험 시점 이후 더 성능 좋은 경량 VLM 출시 (v0.2 신규)
- **확률**: 높음 (2026년 9월까지 신모델 출시 거의 확실)
- **영향**: 논문의 시의성 약화
- **대응**:
  1. 논문 5.3절 한계점에 "실험 시점(2026년 상반기) 기준 모델 선정"으로 명시
  2. 코드 모듈화로 신모델 추가 실험은 향후 연구(5.4절)로 제안
  3. 본 연구의 기여는 "방법론(QLoRA + 자율 HPO)"이므로 모델 자체보다 프레임워크에 초점

---

## 데이터 위험

### 위험 7: 데이터셋 접근 불가
- **확률**: 낮음 (3개 모두 공개 데이터셋)
- **영향**: 해당 데이터셋 사용 불가
- **대응**:
  1. PathVQA: HuggingFace에서 직접 다운로드 가능
  2. SLAKE: med-vqa.com에서 다운로드
  3. VQA-RAD: HuggingFace에서 직접 다운로드
  4. 대체 데이터셋: PMC-VQA (더 큰 규모), AI Hub 한국어 데이터

---

## 최소 보장 범위 (Fallback Plan)

Phase 3가 완전히 실패하더라도, Phase 1 + Phase 2만으로 충분한 논문이 됨:

**축소된 제목**: "경량 멀티모달 모델의 의료 영상 VQA 도메인 적응을 위한 QLoRA 파인튜닝 연구"

**축소된 기여**:
1. 소비자 GPU에서 경량 VLM의 의료 VQA 성능 비교
2. QLoRA 파인튜닝의 효과 실증 및 하이퍼파라미터 민감도 분석
3. 실무 가이드라인 (어떤 모델, 어떤 설정이 최적인지)

이것만으로도 건국대 정보통신대학원 수준에서 충분히 통과 가능한 논문임.
