# 다음 세션 시작점 (마지막 갱신: 2026-07-19)

이 파일은 컴퓨터가 바뀌어도(로컬 `~/.claude` 메모리는 컴퓨터별로 따로 저장되어 동기화되지 않음)
`git pull` 한 번이면 항상 최신 상태로 받아지도록, 다음에 할 일을 저장소에 직접 남겨둔 것입니다.

## 현재 상태

- **Phase 2 Main: 36/36 전 조건 완료** ✅
- 오늘(07-18~19) 세션에서 disk quota/캐시 손상 버그 4건 + eval-split 회귀 1건 수정 (커밋 981bd6d~52b318a)
- Table 4.2b(B) cross-dataset CF 신규 구현·푸시 완료 (커밋 335c808) — **아직 pod에서 실행 안 함, 미검증**
- SSH 개인키(`runpod.ppk`) gitignore 보호 완료 (커밋 c570c64)

## 다음 세션 최우선 작업 (pod에서 실행)

```bash
git pull   # 오늘 푸시한 커밋들 받기

# 1) Cross-dataset CF 최초 실행 (미검증 신규 기능 — 에러 나면 바로 diagnose)
python scripts/measure_cross_dataset_cf.py \
  --config_dir configs/models --phase2_dir results/phase2_finetune \
  --phase1_summary results/phase1_baseline/phase1_summary.csv --seeds 42 123 456

# 2) Phase 1 재실행 (RQ1 McNemar/Cochran's Q, WCA 임상분석용 sample 단위 데이터 복구)
rm -f results/phase1_baseline/phase1_summary.csv results/phase1_baseline/phase1_intermediate.json
python -m src.baseline.run_all --output_dir results/phase1_baseline --data_dir data --batch_size 8

# 3) Mixed-Effects Model 포함 Phase 2 재분석
uv pip install statsmodels pandas
python scripts/analyze_phase2.py --phase1_dir results/phase1_baseline --phase2_dir results/phase2_finetune --base_seed 42
```

그 다음 순서:
- Phase 1 재실행 완료 후: `python scripts/analyze_phase1.py --results_dir results/phase1_baseline --seed 42` (RQ1)
- `python scripts/analyze_clinical.py --results_dir results/phase1_baseline --dataset pathvqa --seed 42` (WCA 임상분석)
- Phase 2 Ablation: `scripts/run_phase2_ablation.sh` (best model 지정 필요, `analyze_phase2.py` 결과로 결정)
- Phase 3 HPO: 아직 미착수. ~$78, ~200 GPU시간 규모 — 예산/시간 재확인 후 착수.

## 알아둘 것

- **unsloth 어댑터 호환성 미검증**: `measure_cross_dataset_cf.py`가 `PeftModel.from_pretrained`로 어댑터를 불러오는데, unsloth로 학습한 qwen3-vl-2b/qwen25-vl-3b 어댑터가 문제없이 로드되는지 실증 안 됨. 에러 나면 즉시 보고.
- **비용 민감**: 이미 $40+ 사용. RunPod 대시보드에서 지출 한도(spending limit) 설정 권장.
- **로컬↔pod 작업 방식**: Claude Code(로컬)는 이 노트북/PC에만 직접 접근 가능. RunPod pod는 SSH 직접 접속 없이 사용자가 웹 터미널에서 명령 실행 후 결과를 복사해서 붙여넣는 방식.
- 상세 이력은 `docs/RUNPOD_GUIDE.md`와 (로컬 `.claude` 메모리가 있는 컴퓨터에서는) auto-memory `runpod-experiment-status.md` 참고.
