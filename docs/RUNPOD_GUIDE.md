# RunPod 실험 실행 가이드

Medical VQA VLM 석사 연구 — Phase 1/2/3 전체 파이프라인 실행 매뉴얼

---

## 목차

1. [RunPod 인스턴스 세팅](#1-runpod-인스턴스-세팅)
2. [환경 세팅 (최초 1회)](#2-환경-세팅-최초-1회)
3. [Phase 1: Zero-shot 베이스라인](#3-phase-1-zero-shot-베이스라인)
4. [Phase 2: QLoRA 파인튜닝](#4-phase-2-qlora-파인튜닝)
5. [Phase 2 Ablation](#5-phase-2-ablation)
6. [Phase 3: Autonomous HPO](#6-phase-3-autonomous-hpo)
7. [결과 다운로드](#7-결과-다운로드)
8. [트러블슈팅](#8-트러블슈팅)

---

## 1. RunPod 인스턴스 세팅

### 권장 사양

| 항목 | 권장 |
|------|------|
| GPU | RTX 4090 24GB |
| Template | RunPod PyTorch 2.4+ (CUDA 12.x) |
| Container Disk | 80GB 이상 |
| Volume Disk | 50GB 이상 (`/workspace` 마운트) |

> Phase 2/3에서 Qwen2.5-VL-7B 사용 시 RTX 4090 필수. 3B/2B 모델만 사용 시 A5000(24GB)도 가능.

---

## 2. 환경 세팅 (최초 1회)

RunPod 터미널에서 실행:

```bash
cd /workspace
git clone https://github.com/taewook486/Masters_degree.git
cd Masters_degree

bash scripts/runpod_setup.sh
```

`runpod_setup.sh`가 자동으로 처리하는 항목:
- 의존성 설치 (`pip install -e ".[unsloth]"`)
- GPU 확인
- 의료 VQA 데이터셋 자동 다운로드 (PathVQA, SLAKE, VQA-RAD)
- VQAv2 subset 다운로드 (CF 측정용)

> **데이터 로컬 업로드 불필요** — 모든 데이터셋이 RunPod에서 자동 다운로드됩니다.

### ANTHROPIC_API_KEY 설정 (Phase 3 필수)

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# 세션 재시작 대비 영구 설정
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
```

---

## 3. Phase 1: Zero-shot 베이스라인

**목표**: 4개 모델 × 3개 데이터셋 × 3개 시드 평가 (BERTScore 포함)

**평가 모델**: Qwen2.5-VL-3B, Qwen3-VL-2B, SmolVLM2-2.2B, Qwen2.5-VL-7B

### 전체 모델 일괄 실행

```bash
# 미구현 — 모델별 개별 실행 권장 (아래 참고)
```

### 모델별 개별 실행 (권장)

```bash
# Qwen2.5-VL-3B
bash scripts/runpod_phase1.sh --config configs/models/qwen25_vl_3b.yaml

# Qwen3-VL-2B
bash scripts/runpod_phase1.sh --config configs/models/qwen3_vl_2b.yaml

# SmolVLM2-2.2B
bash scripts/runpod_phase1.sh --config configs/models/smolvlm2_2b.yaml

# Qwen2.5-VL-7B (24GB VRAM 필수)
bash scripts/runpod_phase1.sh --config configs/models/qwen25_vl_7b.yaml

# Gemma 4 (별도 스크립트)
bash scripts/runpod_phase1_gemma4.sh
```

### 기존 결과 덮어쓰기 (BERTScore 재계산)

```bash
bash scripts/runpod_phase1.sh --config configs/models/qwen25_vl_3b.yaml --no_skip_existing
```

### 완료 후 확인

```bash
# 결과 파일 수 확인
ls results/phase1_baseline/*.json | wc -l

# summary 확인
cat results/phase1_baseline/phase1_summary.csv
```

### Best Model 선택

`phase1_summary.csv`의 `overall_acc_mean` 기준으로 최고 성능 모델 선택 후 메모.

---

## 4. Phase 2: QLoRA 파인튜닝

**목표**: 4개 모델 × 3개 데이터셋 × 3개 시드 = 36개 조건

> Phase 1 완료 및 best model 확인 후 실행

```bash
bash scripts/run_phase2_main.sh
```

로그 확인:

```bash
tail -f results/phase2_finetune/run_phase2.log
```

---

## 5. Phase 2 Ablation

**목표**: Best model에 대해 Ablation A/B/C 실험

> Phase 2 main 완료 후 실행

### 실행 전: best model 수정

```bash
nano scripts/run_phase2_ablation.sh
# BEST_MODEL_CONFIG 변수를 Phase 1/2 best model로 수정
```

```bash
bash scripts/run_phase2_ablation.sh
```

---

## 6. Phase 3: Autonomous HPO

**목표**: 4개 전략 × 5회 반복 × 40 trial = HPO 비교 실험

> Phase 2 완료 및 ANTHROPIC_API_KEY 설정 후 실행

### 실행 전: model_config 수정

```bash
nano scripts/run_phase3.sh
# MODEL_CONFIG를 Phase 2 best model로 수정
```

```bash
bash scripts/run_phase3.sh
```

### 체크포인트 재개 (중단 시)

Phase 3는 체크포인트를 자동 저장합니다. 동일 명령어 재실행 시 마지막 완료 trial부터 자동 재개됩니다.

```bash
# 체크포인트 상태 확인
cat results/phase3_autoresearch/checkpoints/hpo_checkpoint.json
```

---

## 7. 결과 다운로드

### 방법 A: scp

```bash
# 로컬(집 컴퓨터) 터미널에서
scp -P <ssh-port> root@<runpod-ip>:/workspace/Masters_degree/results/phase1_baseline/*.json \
    "D:/project/Masters_degree/results/phase1_baseline/"
```

### 방법 B: RunPod UI

RunPod 콘솔 → Files 탭 → `/workspace/Masters_degree/results/` 에서 직접 다운로드

### 방법 C: git push

```bash
# RunPod에서 — results/ 폴더가 .gitignore에서 제외된 경우
git add results/
git commit -m "data: Phase 1 results with BERTScore"
git push
```

---

## 전체 실행 흐름 요약

```
bash scripts/runpod_setup.sh        # 최초 1회 — 의존성 + 데이터 자동 설치
         │
         ▼
bash scripts/runpod_phase1.sh ...   # 모델별 실행 (BERTScore 포함)
         │
         ▼ phase1_summary.csv에서 best model 선택
bash scripts/run_phase2_main.sh
         │
         ▼ run_phase2_ablation.sh 에서 BEST_MODEL_CONFIG 수정
bash scripts/run_phase2_ablation.sh
         │
         ▼ run_phase3.sh 에서 MODEL_CONFIG 수정
export ANTHROPIC_API_KEY=sk-ant-...
bash scripts/run_phase3.sh
```

**수동 개입 포인트**:
- Phase 1 완료 → `phase1_summary.csv` 분석 → best model 결정
- Phase 2 완료 → `run_phase2_ablation.sh`, `run_phase3.sh`에서 모델 변수 수정

---

## 8. 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `CUDA out of memory` | VRAM 부족 | `--batch_size 4` (기본 8) 또는 더 작은 모델 |
| `ModuleNotFoundError` | 의존성 미설치 | `pip install -e ".[unsloth]"` 재실행 |
| BERTScore hang | roberta-large 최초 다운로드 중 | 잠시 대기 (자동 다운로드) |
| Phase 3 random fallback | API key 없음 | `export ANTHROPIC_API_KEY=sk-ant-...` |
| 데이터셋 로드 실패 | `runpod_setup.sh` 미실행 | `bash scripts/runpod_setup.sh` 재실행 |
| `wandb` 오류 | WANDB 미설정 | `export WANDB_MODE=offline` |

---

*최종 업데이트: 2026-05-07*
