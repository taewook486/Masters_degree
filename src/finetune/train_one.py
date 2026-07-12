"""단일 조건 QLoRA 학습 진입점 (서브프로세스 격리용).

각 (model, dataset, seed) 조건을 독립 프로세스에서 실행한다. 이유:

unsloth를 import하면 trl.SFTTrainer를 전역 몽키패치한다. 한 프로세스 안에서
unsloth 모델(qwen)과 non-unsloth 모델(smolvlm2)을 함께 돌리면, unsloth 패치가
standard backend의 native VLM collation까지 가로채 학습이 실패한다
('supply input_ids, but you provided [images, ...]').

서브프로세스로 격리하면:
  - standard 모델: MOAI_SKIP_UNSLOTH=1을 설정 → train_qlora가 unsloth를 아예
    import하지 않음 → 순수 trl SFTTrainer + DataCollatorForVisionLanguageModeling.
  - unsloth 모델(qwen): env 미설정 → train_qlora 상단에서 unsloth를 transformers
    보다 먼저 import → eos_token 수정 + unsloth backend 정상.
  - 조건마다 프로세스가 종료되어 GPU/CPU 메모리도 완전히 해제된다.

run_phase2.py가 이 스크립트를 subprocess로 호출한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

# train_qlora._UNSLOTH_SUPPORTED_PATTERNS와 동일해야 한다. 여기서는 무거운 import
# (transformers/unsloth) 없이 yaml만 읽어 backend를 판정하기 위해 패턴을 복제한다.
_UNSLOTH_SUPPORTED_PATTERNS = ["qwen2.5-vl", "qwen3-vl", "qwen2-vl", "gemma-4", "gemma-3n"]


def _model_supports_unsloth(model_id: str) -> bool:
    mid = model_id.lower()
    return any(pat in mid for pat in _UNSLOTH_SUPPORTED_PATTERNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-condition QLoRA training (subprocess-isolated)")
    parser.add_argument("--model_config_path", required=True)
    parser.add_argument("--finetune_config_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--subset_ratio", type=float, default=None)
    parser.add_argument("--measure_cf", action="store_true")
    parser.add_argument("--base_vqav2_json", default=None, help="Path to pre-computed base VQAv2 result JSON")
    args = parser.parse_args()

    # 1) yaml만 읽어 backend 판정 (transformers/unsloth import 전에)
    with open(args.model_config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    model_id = raw.get("model_id", "")
    use_unsloth = _model_supports_unsloth(model_id)

    # 2) standard 모델이면 unsloth 전역 패치를 막는다 (train_qlora import 전에 설정해야 함)
    if not use_unsloth:
        os.environ["MOAI_SKIP_UNSLOTH"] = "1"

    # 3) 이제 train_qlora import (여기서 env에 따라 unsloth import 여부가 결정된다)
    from src.finetune.train_qlora import train_qlora

    base_vqav2_result = None
    if args.base_vqav2_json and os.path.exists(args.base_vqav2_json):
        with open(args.base_vqav2_json, encoding="utf-8") as f:
            base_vqav2_result = json.load(f)

    # train_qlora가 output_dir/train_result.json을 자체 기록한다. 부모(_train_condition)는
    # 서브프로세스 정상 종료 후 그 파일을 읽는다 → 별도 직렬화 불필요.
    train_qlora(
        model_config_path=args.model_config_path,
        finetune_config_path=args.finetune_config_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        seed=args.seed,
        data_dir=args.data_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_test_samples=args.max_test_samples,
        subset_ratio=args.subset_ratio,
        measure_cf=args.measure_cf,
        base_vqav2_result=base_vqav2_result,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
