# 다음 세션 시작점 (마지막 갱신: 2026-07-25 자정)

이 파일은 컴퓨터가 바뀌어도(로컬 `~/.claude` 메모리는 컴퓨터별로 따로 저장되어 동기화되지 않음)
`git pull` 한 번이면 항상 최신 상태로 받아지도록, 다음에 할 일을 저장소에 직접 남겨둔 것입니다.

## 현재 상태

- **RunPod pod 중지 예정 (2026-07-25 자정)** — 사용자가 오늘 세션을 마무리하며 pod를 중지(stop)함. 재시작 후 아래 "다음 세션 최우선 작업"부터 진행.
- **[주의] GitHub PAT 토큰 노출됨, 무효화 여부 미확인** — 이 세션 중 `git remote -v` 실행 결과가 그대로 대화에 노출되어 토큰이 평문으로 남았음. 다음 세션 시작 시 GitHub → Settings → Developer settings → Personal access tokens에서 해당 토큰이 아직 살아있는지 확인하고, 살아있다면 즉시 revoke 후 재발급할 것.
- **Phase 2 Main+Ablation 전체(75조건) 완료, git 백업+push 완료** ✅ — Main 36 + Ablation A(비율 5종×3시드=15) + B(rank 5종×3시드=15) + C(target 3종×3시드=9) = 75조건. 결과(JSON/CSV/MD만, 어댑터 가중치 제외) 커밋 `8230e64`로 push 완료.
  - **확정**: ablation_a → **ratio=1.0**이 최적(단조 증가, 아직 한계치 안 보임). ablation_b → **rank=64**가 최적(단조 증가, 시간·VRAM 비용 거의 안 늘어남). 재실행 불필요.
- **[긴급] Ablation-C 6개 조건(minimal 3시드 + medium 3시드) 재실행 필요** — 2026-07-25 밤 재검증으로 발견: 이전에 "target_medium/seed123 조건 하나만" 문제라고 진단했었으나(아래 옛 기록), 각 조건 `train_result.json`의 `metadata.timestamp`를 직접 대조한 결과 **minimal 3시드 전부(평균 390분) + medium 3시드 전부(평균 300분), 총 6개 조건**이 `max_steps=500` 캡(커밋 `64a3fe9`) 적용 전에 이미 실행 완료되어 있었던 것으로 확인됨(캡 적용 후 정상 실행된 full은 평균 36.8분). `skip_existing=True` 기본값 때문에 캡 반영 후 재실행해도 이 6개는 스킵되어 옛 결과가 그대로 남아있음. 상세 경위는 로컬 auto-memory `phase2-ablation-c-confound` 참고. **git pull 후 아래로 재실행**:
  ```bash
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_minimal_seed42
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_minimal_seed123
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_minimal_seed456
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_medium_seed42
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_medium_seed123
  rm -rf results/phase2_finetune/ablation_c_qwen3-vl-2b_pathvqa_medium_seed456
  bash scripts/run_phase2_ablation.sh   # skip_existing=True라 나머지 조건은 자동 스킵, 삭제한 6개만 재학습(~37분/조건, 총 3~4시간 예상)
  ```
  재실행 후 `wc -l results/phase2_finetune/phase2_summary.csv`로 중복 행 없이 76줄(헤더+75) 유지되는지 확인할 것 — 검증 안 된 부분.
  (BEST_MODEL_CONFIG가 `qwen3_vl_2b.yaml`이 아니면 디렉터리명의 `qwen3-vl-2b` 부분을 실제 `model_name`으로 바꿀 것.)
  - *(옛 기록, 참고용 — 위 재검증으로 대체됨)*: 2026-07-25 최초 발견 시엔 `target_medium/seed123` 조건 하나만 문제라고 판단해 그 조건만 삭제 후 재실행을 계획했었으나, 표본 하나만 우연히 걸려 나온 불완전한 진단이었음.
- **Phase 2 Main: 36/36 전 조건 완료** ✅ (결과 git 백업 완료, 커밋 d6c6cd5)
- **Phase 2 Ablation: 원인 수정 후 재시작, 07-22 01:33 UTC부터 재실행 중 (오후에 진행 상황 재확인 예정)**
  - **수정 검증됨**: 재시작 후 이전 실패 지점(`unsloth_compile_transformers`의 `Conv1d` AttributeError)을 정상 통과, 모델 로딩·LoRA 패칭 완료, train split 생성까지 에러 없이 진행 확인(01:35 UTC 기준). 캐시 경로 수정이 root cause를 잡은 것으로 사실상 확정.
  - 재확인 시 우선 체크: `ls results/phase2_finetune/ablation_*/train_result.json | wc -l`, `tail -n 30 results/phase2_finetune/run_phase2.log`, `nvidia-smi`
  - 목표: Ablation A(데이터비율 5종)+B(rank 5종)+C(타겟모듈 3종) × 3 seed = 39개 조건, PathVQA로만, full test set 평가.
  - 2026-07-21 21:39 `ratio=1.0 seed=456` 조건에서 `AttributeError: module 'Conv1d' has no attribute 'forward'`(unsloth 컴파일 캐시 쓰기 실패로 추정) 실패 후, 곧이어 Ablation B 준비 중 `_write_ablation_config`가 작은 yaml 파일 하나 쓰다 `OSError: Disk quota exceeded`로 `run_phase2.py` 전체가 죽으며 파이프라인 중단.
  - **Root cause 확정**: `scripts/run_phase2_ablation.sh`에 `run_phase2_main.sh`와 달리 `HF_HOME`/`MOAI_CHAT_CACHE_DIR` 환경변수가 빠져 있었음. 이 컨테이너는 `$HOME=/workspace`라 HF 캐시가 기본값(`$HOME/.cache/huggingface`)으로 저장되며 `/workspace/.cache`(28G)를 채워 quota 초과. 커밋 `2db2361`에서 수정(Main과 동일 캐시 경로로 통일 — 이미 빌드된 chat_cache도 재사용 가능).
  - 조치 완료: 완료된 조건들의 결과(JSON/CSV/MD, 어댑터 가중치 제외)를 git 백업(`d6c6cd5`), stray 캐시 정리(`/workspace/.cache/huggingface`, `/workspace/.cache/uv` 삭제로 `/workspace` 89G→69G).
  - **재확인 시**: 여전히 정상 진행 중이면 그대로 완료까지 방치. 만약 다시 죽어있으면 `tail -n 50 run_phase2.log`로 원인부터 확인(재발 시 disk quota 재발이 아니라 unsloth_zoo 자체 컴파일 캐시 손상 가능성 있음 — `~/.cache/unsloth_compiled_cache` 등 삭제 후 재시도) 후, `bash scripts/run_phase2_ablation.sh`로 재실행하면 `skip_existing=True`가 기본값이라 완료된 조건은 자동 스킵하고 이어감.
  - 예상 소요 ~55-85시간(RTX 3090 기준), 비용 ~$25-40 추가 예상.
- 오늘(07-18~19) 세션에서 disk quota/캐시 손상 버그 4건 + eval-split 회귀 1건 수정 (커밋 981bd6d~52b318a)
- Table 4.2b(B) cross-dataset CF 신규 구현·푸시 완료 (커밋 335c808) — **아직 pod에서 실행 안 함, 미검증**
- SSH 개인키(`runpod.ppk`) gitignore 보호 완료 (커밋 c570c64)

## 다음 세션 최우선 작업 (pod에서 실행)

```bash
git pull   # 8230e64 이상 확인 (Phase 2 Main+Ablation 75조건 결과 백업 포함)

# [중요] 이 세션 전체에 캐시 경로 지정 — run_phase2_ablation.sh가 겪은 것과 같은
# Disk quota exceeded를 막기 위해, .sh 래퍼 없이 bare python으로 실행하는
# 아래 1)/2) 단계 전에 반드시 먼저 실행할 것 (run_phase3.sh/runpod_phase1.sh는
# 스크립트 자체에 내장했지만, 아래 python 직접 호출은 셸에서 직접 잡아줘야 함)
export HF_HOME=/hf_cache
export MOAI_CHAT_CACHE_DIR=/workspace/hf_cache/chat_cache

# 0) Ablation-C 오염된 6개 조건 재실행 — 최우선. 명령어는 위 "[긴급]" 항목 참고.
tmux new -s ablation
cd /workspace/Masters_degree && source .venv/bin/activate
# (위 [긴급] 항목의 rm -rf 6줄 + bash scripts/run_phase2_ablation.sh 실행)
# Ctrl+B, D로 detach 후 방치. 진행 확인: ls results/phase2_finetune/ablation_c_*/train_result.json | wc -l

# 1) Cross-dataset CF 최초 실행 (미검증 신규 기능 — 에러 나면 바로 diagnose)
# --max_samples 500: full test set이면 PathVQA(6,719개)가 24회 반복되어 ~16시간
# 소요 추정(보조 지표라 과함) → 500으로 제한해 ~2-3시간으로 단축(사용자 확정, 2026-07-19)
python scripts/measure_cross_dataset_cf.py \
  --config_dir configs/models --phase2_dir results/phase2_finetune \
  --phase1_summary results/phase1_baseline/phase1_summary.csv --seeds 42 123 456 \
  --max_samples 500

# 2) Phase 1 재실행 (RQ1 McNemar/Cochran's Q, WCA 임상분석용 sample 단위 데이터 복구)
rm -f results/phase1_baseline/phase1_summary.csv results/phase1_baseline/phase1_intermediate.json
python -m src.baseline.run_all --output_dir results/phase1_baseline --data_dir data --batch_size 8

# 3) Mixed-Effects Model 포함 Phase 2 재분석
uv pip install statsmodels pandas
python scripts/analyze_phase2.py --phase1_dir results/phase1_baseline --phase2_dir results/phase2_finetune --base_seed 42
```

그 다음 순서:
- Ablation-C 6개 조건 재실행 완료 후 target modules 최적 조합 재계산(이전 "minimal 1등"은 컨파운드로 신뢰 불가)
- ratio=1.0, rank=64, target=(재계산 예정) 조합을 `configs/finetune/base_qlora.yaml`에 반영할지 결정 — Phase 3는 기본적으로 이 파일을 그대로 재사용함
- Phase 3(`scripts/run_phase3.sh`) 실행 전 `ANTHROPIC_API_KEY` 환경변수 필수(없으면 스크립트가 preflight에서 즉시 중단)
- Ablation 39/39 완료될 때까지 기다렸다가 결과 확인
- Phase 1 재실행 완료 후: `python scripts/analyze_phase1.py --results_dir results/phase1_baseline --seed 42` (RQ1)
- `python scripts/analyze_clinical.py --results_dir results/phase1_baseline --dataset pathvqa --seed 42` (WCA 임상분석)
- Phase 3 HPO: 아직 미착수. ~$78, ~200 GPU시간 규모 — Ablation까지 끝난 뒤 예산/시간 재확인 후 착수.
- **비용 대안 검토는 Ablation 완료 후로 보류(사용자 확정, 2026-07-22)**: 누적 지출 $120 초과로 비용 절감 방향을 논의함. 건국대 자체 중앙 HPC는 확인 안 됨(랩 단위 GPU서버만 존재). KISTI 국가슈퍼컴퓨팅센터(뉴론 GPU 클러스터)가 유력 후보 — 연구비 확보 어려운 과제 대상 무상지원 트랙 있음, `enables.ksc.re.kr`/`www.ksc.re.kr`에서 최신 공모 확인 필요(제 쪽에서 인증서 오류로 실시간 검증은 못 함). 참고 문서: `docs/ENVIRONMENT_SETUP.md`(새 컴퓨터 환경 요구사항), `docs/GPU_RENTAL_INQUIRY_CHECKLIST.md`(대여처 문의 체크리스트). Ablation 39/39 끝나면 이어서 검토.

## 알아둘 것

- **캐시 디스크 분산 배치는 스크립트마다 개별 설정해야 함**: `HF_HOME=/hf_cache`, `MOAI_CHAT_CACHE_DIR=/workspace/hf_cache/chat_cache`는 공용 함수가 아니라 각 실행 스크립트에 개별적으로 export돼 있음. 2026-07-22 전수 점검 결과 `run_phase2_main.sh`/`run_phase2_ablation.sh`는 있었지만 `run_phase3.sh`/`runpod_phase1.sh`엔 빠져있어서 추가함(커밋 확인은 위 git pull 안내 참고). `runpod_phase1_gemma4.sh`(deprecated, 미수정)와 `.sh` 래퍼 없이 직접 실행하는 `measure_cross_dataset_cf.py`/`run_all.py` 같은 bare python 명령은 **셸에서 직접 export해야 함**(위 최우선 작업 블록에 이미 추가함). 새 실행 스크립트를 추가하거나 복사해서 만들 때 이 export 누락 여부를 반드시 확인할 것 — 누락되면 `$HOME=/workspace`인 이 컨테이너에서는 조용히 `/workspace/.cache`로 캐시가 새며, 몇 시간 뒤 quota 초과로 전체 파이프라인이 죽는 형태로만 드러남(초기 증상은 무관해 보이는 `AttributeError` 등으로 나타날 수 있어 진단이 어려움).
- **`df -h /workspace`는 신뢰 불가**: `/workspace`는 `mfs#eu-cz-1.runpod.net` 네트워크 볼륨이라 `df -h`가 리전 전체 풀 용량(851T)을 보여줌. 실제 이 pod의 quota 확인은 `du -h --max-depth=1 /workspace`로 해야 함(`-s`와 `--max-depth`는 동시 사용 불가, `du: warning`만 뜨고 결과 없음).
- **unsloth 어댑터 호환성 미검증**: `measure_cross_dataset_cf.py`가 `PeftModel.from_pretrained`로 어댑터를 불러오는데, unsloth로 학습한 qwen3-vl-2b/qwen25-vl-3b 어댑터가 문제없이 로드되는지 실증 안 됨. 에러 나면 즉시 보고.
- **비용 민감**: 이미 $40+ 사용. RunPod 대시보드에서 지출 한도(spending limit) 설정 권장.
- **로컬↔pod 작업 방식**: Claude Code(로컬)는 이 노트북/PC에만 직접 접근 가능. RunPod pod는 SSH 직접 접속 없이 사용자가 웹 터미널에서 명령 실행 후 결과를 복사해서 붙여넣는 방식.
- 상세 이력은 `docs/RUNPOD_GUIDE.md`와 (로컬 `.claude` 메모리가 있는 컴퓨터에서는) auto-memory `runpod-experiment-status.md` 참고.
