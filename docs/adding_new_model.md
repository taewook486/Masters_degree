# 새 VLM 모델 추가 가이드

새로운 Vision-Language Model을 평가 파이프라인에 추가하는 절차입니다.

---

## 사전 확인

새 모델 추가 전 아래 조건을 확인하세요:

| 항목 | 요구 조건 |
|------|----------|
| 라이선스 | Apache 2.0, MIT 등 연구 활용 가능 |
| VRAM | 제로샷 추론 시 16GB 이하 (로컬), QLoRA 학습 시 16GB 이하 |
| 프레임워크 | HuggingFace transformers 호환 (`from_pretrained` 지원) |
| 모달리티 | 이미지 + 텍스트 입력 지원 (VLM) |
| chat_template | `processor.apply_chat_template()` 지원 |

---

## 절차

### Step 1: 모델 Config 작성

템플릿을 복사하여 새 config 파일을 생성합니다.

```bash
cp configs/models/_template.yaml configs/models/<model_name>.yaml
```

필수 수정 항목:
- `model_name`: 결과 파일에 사용될 식별자 (예: `gemma4-e2b`)
- `model_id`: HuggingFace 모델 ID (예: `google/gemma-4-E2B-it`)
- `model_class`: transformers 모델 클래스명
- `torch_dtype`: `float16` 또는 `bfloat16`

모델 클래스 확인 방법:
```python
# HuggingFace 모델 페이지에서 확인하거나:
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained("모델ID", device_map="cpu")
print(type(model))  # 정확한 클래스명 확인
```

### Step 2: 호환성 테스트

```bash
# GPU 없이 기본 검증 (config + 클래스 + processor)
python scripts/test_model_compat.py configs/models/<model_name>.yaml --skip-inference

# GPU 포함 전체 검증 (모델 로드 + 추론)
python scripts/test_model_compat.py configs/models/<model_name>.yaml
```

4단계 모두 PASS가 나와야 합니다:
1. Config loading - YAML 필수 필드 확인
2. Class resolution - transformers에 모델 클래스 존재 확인
3. Processor + chat_template - 프롬프트 생성 확인
4. Model load + inference - 실제 추론 동작 확인

### Step 3: Phase 1 평가 실행

```bash
# 로컬 실행
python scripts/run_phase1_single.py --config configs/models/<model_name>.yaml

# RunPod 실행
bash scripts/runpod_phase1.sh --config configs/models/<model_name>.yaml --batch_size 8
```

### Step 4: 문서 업데이트

아래 파일들을 업데이트하세요:

| 파일 | 업데이트 내용 |
|------|-------------|
| `README.md` | 대상 모델 테이블에 새 모델 추가 |
| `docs/phase1_work_log.md` | 테스트 결과 및 진행 상황 기록 |
| `docs/THESIS_PROPOSAL_*.md` | 대상 모델 수, 실험 조건 수 업데이트 |
| `docs/RISK_ANALYSIS_*.md` | VRAM 위험 분석에 새 모델 정보 추가 |

업데이트 시 변경 포인트:
- 모델 수: N개 → N+1개
- 실험 조건 수: (N x 3 데이터셋) → ((N+1) x 3 데이터셋)
- ANOVA 모델 수: N개 모델 → N+1개 모델
- Phase 1/2 CF 측정 적용 범위 업데이트

---

## 트러블슈팅

### transformers 버전 문제

새 모델의 클래스가 현재 transformers에 없는 경우:

```bash
# 필요 버전 확인 (모델의 HuggingFace 페이지 참고)
# 업그레이드
uv lock --upgrade-package transformers
uv sync

# 버전 확인
uv run python -c "import transformers; print(transformers.__version__)"
```

### VRAM 초과 (OOM)

제로샷 추론 시 OOM이 발생하면:

1. `batch_size`를 줄이기 (4 → 2 → 1)
2. `torch_dtype`을 `bfloat16`으로 변경
3. `processor_kwargs`에 `max_pixels` 제한 추가
4. 16GB 초과 모델은 RunPod (24GB+) 사용

### chat_template 미지원

`apply_chat_template` 오류 시:
- 모델의 `tokenizer_config.json`에 `chat_template` 필드 확인
- 커스텀 프롬프트 포맷이 필요한 경우 `src/baseline/model_loader.py`에 분기 추가 필요

---

## 파일 구조

```
새 모델 추가 시 생성/수정되는 파일:

configs/models/
├── _template.yaml          ← 복사 원본
└── <model_name>.yaml       ← 새 모델 config

scripts/
├── test_model_compat.py    ← 호환성 테스트 (범용)
├── run_phase1_single.py    ← Phase 1 평가 (범용)
└── runpod_phase1.sh        ← RunPod 실행 (범용)

results/phase1_baseline/
└── <model_name>_*.json     ← 평가 결과 (자동 생성)
```

---

## 체크리스트

새 모델 추가 시 아래 항목을 순서대로 확인하세요:

- [ ] Config YAML 작성 (`configs/models/<model_name>.yaml`)
- [ ] 호환성 테스트 PASS (`scripts/test_model_compat.py`)
- [ ] Phase 1 평가 실행 완료
- [ ] README.md 모델 테이블 업데이트
- [ ] docs/phase1_work_log.md 기록
- [ ] 논문 설계서 모델 수/조건 수 업데이트
- [ ] 위험 분석 VRAM 정보 업데이트
- [ ] 변경사항 커밋 및 푸시
