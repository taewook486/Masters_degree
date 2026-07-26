# 다음 세션 시작점 (마지막 갱신: 2026-07-26)

이 파일은 컴퓨터가 바뀌어도(로컬 `~/.claude` 메모리는 컴퓨터별로 따로 저장되어 동기화되지 않음)
`git pull` 한 번이면 항상 최신 상태로 받아지도록, 다음에 할 일을 저장소에 직접 남겨둔 것입니다.

## 현재 상태

- **2026-07-26 세션 완료 — 우선순위 ①②③(Cross-dataset CF → Phase 1 재실행 → Phase 2 재분석) 전부 끝남** ✅
  - Ablation-C 컨파운드 6조건 재실행 완료 + target modules 최적 조합 확정: **rank=64, alpha=128, target=full(all-linear)** → `configs/finetune/base_qlora.yaml`에 반영, push 완료(커밋 `f52080b`). ratio=1.0/rank=64/target=full은 각각 축 하나씩만 바꿔가며 독립 검증한 것으로, 세 값을 동시에 적용한 조합 자체는 미검증(한계점으로 기록해둘 것).
  - Cross-dataset CF 72/72 조건 측정 완료(커밋 `20be0eb`). 도중 `transformers==5.5.0`의 `TokenizersBackend` 리팩터링으로 `bert-score`가 쓰는 `build_inputs_with_special_tokens`가 빠진 버그 발견·수정(커밋 `121af8e`, `src/evaluate/metrics.py`) — roberta-large/biobert 둘 다 실측 검증됨.
  - Phase 1 재실행을 "36개 전부"가 아니라 **백업 복원(27개, `_pre_bertscore/`) + gemma4-e2b 9→3개만 신규 실행**으로 대체해 GPU 시간 절감(커밋 `25f9f9f`). `scripts/rescore_phase1.py`가 원본에 bertscore 없는 조건에서 빈 값을 남기던 버그도 수정(커밋 `71bb476`). RQ1(McNemar/Cochran's Q), 임상분석(WCA, pathvqa), RQ2(Mixed-Effects Model) 전부 완료·백업(커밋 `8ec9fed`~`ad850b3`).
  - **RQ2 해석 주의**: 아래 참고.
- **RQ2 Mixed-Effects Model 해석 주의 (2026-07-26, 논문 작성 시 반드시 반영)** — `results/phase2_finetune/phase2_rq2_analysis.md`의 MEM(`accuracy ~ condition + dataset`, group=seed) 고정효과는 **p=0.3629로 유의하지 않고 ICC(seed)=0.0**으로 나오는데, 이건 계산 오류가 아니라 **MEM이 4개 모델을 구분 없이 합쳐서 추정**하기 때문임. 실제로는 모델별 3중 검증(paired t-test + BCa Bootstrap Cohen's d + Wilcoxon)에서 효과가 **모델마다 정반대**로 나타남 — qwen25-vl-3b(d=+2.65, 유의) / qwen3-vl-2b(d=+1.62, 유의)는 파인튜닝이 확실히 도움됐고, smolvlm2-2b(d=-2.28, 유의)는 오히려 나빠졌고, gemma4-e2b(d=-0.65, 비유의)도 부정 방향임. 이질적(heterogeneous) 효과가 pooled 평균에서 상쇄되며 MEM이 "효과 없음"으로 보이는 것 — **RQ2 결론은 모델별 3중 검증 결과를 1차 근거로 쓰고, MEM pooled 결과는 "모델 구분 없는 전체 효과는 유의하지 않음(모델간 이질성 때문)"이라는 보조 설명으로만 인용할 것.** 코드는 수정하지 않기로 결정함(2026-07-26 확정) — MEM 수식에 model 고정효과/상호작용을 추가하는 개선은 보류.
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

Cross-dataset CF, Phase 1 재실행(+RQ1/WCA), Phase 2 재분석(RQ2)까지 전부 끝났음(위 "2026-07-26 세션 완료" 참고). 이제 진짜 남은 건 이 3가지:

```bash
git pull   # 7a37e42 이상 확인

export HF_HOME=/hf_cache
export MOAI_CHAT_CACHE_DIR=/workspace/hf_cache/chat_cache
```

1. **[주의, 아직 미확인] GitHub PAT 토큰 무효화 여부** — 2026-07-25 세션 중 `git remote -v` 결과가 노출됐던 건. GitHub → Settings → Developer settings → Personal access tokens에서 살아있는지 확인, 살아있으면 즉시 revoke 후 재발급.
2. **Phase 3 HPO 착수** — 아직 미착수. `scripts/run_phase3.sh` 실행 전 `ANTHROPIC_API_KEY` 환경변수 필수(없으면 preflight에서 즉시 중단). ~$78, ~200 GPU시간 규모로 사전 추정해뒀음 — 예산 재확인 후 착수. base_qlora.yaml이 이미 최적 조합(rank=64/target=full)으로 갱신돼 있으니 Phase 3는 이 설정을 그대로 재사용하면 됨.
3. **비용 대안 검토(KISTI 등)** — 누적 지출 $120+ 관련, Ablation 끝나면 이어서 검토하기로 보류해뒀던 것. 건국대 중앙 HPC는 없음(랩 단위 GPU서버만). KISTI 국가슈퍼컴퓨팅센터(뉴론 GPU 클러스터) 무상지원 트랙이 유력 후보 — `enables.ksc.re.kr`/`www.ksc.re.kr`에서 최신 공모 확인 필요(인증서 오류로 원격 실시간 검증은 못 함). 참고: `docs/ENVIRONMENT_SETUP.md`, `docs/GPU_RENTAL_INQUIRY_CHECKLIST.md`.

## 알아둘 것

- **캐시 디스크 분산 배치는 스크립트마다 개별 설정해야 함**: `HF_HOME=/hf_cache`, `MOAI_CHAT_CACHE_DIR=/workspace/hf_cache/chat_cache`는 공용 함수가 아니라 각 실행 스크립트에 개별적으로 export돼 있음. 2026-07-22 전수 점검 결과 `run_phase2_main.sh`/`run_phase2_ablation.sh`는 있었지만 `run_phase3.sh`/`runpod_phase1.sh`엔 빠져있어서 추가함(커밋 확인은 위 git pull 안내 참고). `runpod_phase1_gemma4.sh`(deprecated, 미수정)와 `.sh` 래퍼 없이 직접 실행하는 `measure_cross_dataset_cf.py`/`run_all.py` 같은 bare python 명령은 **셸에서 직접 export해야 함**(위 최우선 작업 블록에 이미 추가함). 새 실행 스크립트를 추가하거나 복사해서 만들 때 이 export 누락 여부를 반드시 확인할 것 — 누락되면 `$HOME=/workspace`인 이 컨테이너에서는 조용히 `/workspace/.cache`로 캐시가 새며, 몇 시간 뒤 quota 초과로 전체 파이프라인이 죽는 형태로만 드러남(초기 증상은 무관해 보이는 `AttributeError` 등으로 나타날 수 있어 진단이 어려움).
- **`df -h /workspace`는 신뢰 불가**: `/workspace`는 `mfs#eu-cz-1.runpod.net` 네트워크 볼륨이라 `df -h`가 리전 전체 풀 용량(851T)을 보여줌. 실제 이 pod의 quota 확인은 `du -h --max-depth=1 /workspace`로 해야 함(`-s`와 `--max-depth`는 동시 사용 불가, `du: warning`만 뜨고 결과 없음).
- **unsloth 어댑터 호환성 — 검증 완료(2026-07-26)**: `measure_cross_dataset_cf.py`의 `PeftModel.from_pretrained` 어댑터 로드, 72/72 조건 전부 정상 동작 확인됨. 더 이상 미검증 아님.
- **비용 민감**: 이미 $40+ 사용. RunPod 대시보드에서 지출 한도(spending limit) 설정 권장.
- **로컬↔pod 작업 방식**: Claude Code(로컬)는 이 노트북/PC에만 직접 접근 가능. RunPod pod는 SSH 직접 접속 없이 사용자가 웹 터미널에서 명령 실행 후 결과를 복사해서 붙여넣는 방식.
- **`python3`가 venv를 안 가리킬 수 있음(2026-07-26 실증)**: 세션 중간에 venv activate가 풀리면 `python3`가 `/usr/bin/python3`(시스템, 패키지 없음)로 잡히면서 멀쩡한 패키지가 `ModuleNotFoundError`로 보이는 헛다리짚기가 발생함. `which python3`로 `.venv` 경로인지 먼저 확인하거나, 아예 **`uv run python ...`으로 통일**하면 셸 상태와 무관하게 항상 올바른 프로젝트 환경을 씀 — 앞으로는 이걸 기본으로 쓸 것.
- **`uv sync`의 하드링크가 이 네트워크 볼륨(`/workspace`)에서 가끔 깨짐(2026-07-26 실증)**: `nvidia-cusparselt-cu12`, `nvidia-nvshmem-cu12`, `scipy`가 각각 "설치됨"으로 기록돼 있는데 실제 파일은 로드 안 되는 증상이 반복됨(패키지 하나씩 `uv sync --reinstall-package <pkg>`로 개별 복구 가능하지만 계속 재발할 수 있음). 근본적으로는 `UV_LINK_MODE=copy uv sync --reinstall`로 하드링크 대신 실제 복사를 강제하는 게 더 안정적임 — 다음에 또 이런 `ModuleNotFoundError`/`ImportError: lib*.so`류가 나오면 이걸 먼저 시도할 것.
- **bert-score + `transformers==5.5.0` 호환성 버그(2026-07-26 수정)**: `transformers` 5.5.0에서 토크나이저가 새 `TokenizersBackend`로 리팩터링되며 `build_inputs_with_special_tokens`가 빠짐 — `bert_score` 0.3.13이 이 메서드를 직접 호출해 `AttributeError`로 죽음. `src/evaluate/metrics.py`의 `_patch_tokenizers_backend_special_tokens`(커밋 `121af8e`)로 공유 베이스 클래스에 호환 shim을 패치해뒀음. 만약 다른 bert-score 계열 스크립트에서 비슷한 에러가 또 나면, 이미 고쳐져 있는지부터(`git log -- src/evaluate/metrics.py`) 확인.
- **`git` index.lock이 이 저장소(`/mnt/d/...` WSL 마운트)에서 종종 stale하게 남음**: 느린 DrvFs 때문에 `git status`류가 오래 걸리다 index.lock을 남기고, 다음 git 명령이 "Another git process seems to be running"로 막히는 경우가 반복됨. `ps aux | grep git`+`lsof <lockfile>`로 실제 홀더가 없는 걸 확인한 뒤에만 `rm -f .git/index.lock`으로 지울 것(무작정 지우지 말 것).
- 상세 이력은 `docs/RUNPOD_GUIDE.md`와 (로컬 `.claude` 메모리가 있는 컴퓨터에서는) auto-memory `runpod-experiment-status.md` 참고.
