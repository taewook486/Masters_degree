"""범용 VLM 모델 호환성 테스트.

새 모델 추가 시 기존 평가 파이프라인과의 호환성을 단계별로 검증:
1. Config 로딩
2. transformers 클래스 해석
3. Processor 로드 + chat_template 확인
4. 모델 로드 + 단일 샘플 추론 (GPU 필요)

사용법:
    python scripts/test_model_compat.py configs/models/gemma4_e2b.yaml
    python scripts/test_model_compat.py configs/models/gemma4_e2b.yaml --skip-inference
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch


def step1_config_loading(config_path: str) -> dict:
    """Step 1: YAML config 로딩 테스트."""
    from src.baseline.model_loader import load_config

    config = load_config(config_path)

    required_fields = ["model_name", "model_id", "model_class", "prompt_format"]
    for field in required_fields:
        assert hasattr(config, field), f"필수 필드 누락: {field}"

    print(f"  model_name:   {config.model_name}")
    print(f"  model_id:     {config.model_id}")
    print(f"  model_class:  {config.model_class}")
    print(f"  torch_dtype:  {config.get('torch_dtype', 'auto')}")
    print(f"  prompt_format: {config.prompt_format}")

    vision_info = config.get("requires_vision_info_processing", False)
    print(f"  requires_vision_info_processing: {vision_info}")

    return config


def step2_class_resolution(config: dict) -> None:
    """Step 2: transformers 클래스 해석 테스트."""
    import transformers

    model_class_name = config.model_class
    model_cls = getattr(transformers, model_class_name, None)
    processor_cls = getattr(transformers, config.get("processor_class", "AutoProcessor"), None)

    assert model_cls is not None, f"{model_class_name} not found in transformers {transformers.__version__}"
    assert processor_cls is not None, "AutoProcessor not found"

    print(f"  transformers: {transformers.__version__}")
    print(f"  model_cls:    {model_cls}")
    print(f"  processor_cls: {processor_cls}")


def step3_processor_and_template(config: dict) -> None:
    """Step 3: Processor 로드 + chat_template 동작 확인."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=config.get("trust_remote_code", False),
    )

    # chat_template 테스트
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is in this image?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert isinstance(text, str) and len(text) > 0, "chat_template 출력이 비어있음"
    print(f"  chat_template output (first 200 chars): {text[:200]}")

    # pad_token 확인
    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id
    print(f"  pad_token_id: {pad_id}, eos_token_id: {eos_id}")


def step4_model_load_and_inference(config_path: str) -> None:
    """Step 4: 모델 로드 + 단일 샘플 추론 (GPU 필요)."""
    from PIL import Image

    from src.baseline.model_loader import generate_answer, load_config, load_model, unload_model

    config = load_config(config_path)
    model, processor = load_model(config)

    try:
        # 간단한 테스트 이미지 생성 (빨간 사각형)
        img = Image.new("RGB", (224, 224), color=(255, 0, 0))

        answer = generate_answer(
            model, processor, img,
            "What color is this image?",
            config,
        )
        print(f"  answer: {answer}")

        # VRAM 사용량
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  peak VRAM: {vram_mb:.0f} MB")
    finally:
        unload_model(model, processor)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VLM 모델 호환성 테스트 (범용)",
    )
    parser.add_argument("config_path", help="모델 config YAML 경로 (예: configs/models/gemma4_e2b.yaml)")
    parser.add_argument("--skip-inference", action="store_true", help="Step 4 (모델 로드 + 추론) 건너뛰기")
    args = parser.parse_args()

    steps = [
        ("Step 1: Config loading", lambda: step1_config_loading(args.config_path)),
        ("Step 2: Class resolution", lambda: step2_class_resolution(config)),
        ("Step 3: Processor + chat_template", lambda: step3_processor_and_template(config)),
    ]

    if not args.skip_inference:
        steps.append(
            ("Step 4: Model load + inference", lambda: step4_model_load_and_inference(args.config_path))
        )

    config = None
    passed = 0
    for name, fn in steps:
        print(f"\n{'='*60}")
        print(f"{name}")
        print("=" * 60)
        try:
            result = fn()
            if name == "Step 1: Config loading":
                config = result
            print(f"  -> PASS")
            passed += 1
        except Exception as e:
            print(f"  -> FAIL: {e}")
            traceback.print_exc()
            if name == "Step 1: Config loading":
                # config 없으면 나머지 스텝 진행 불가
                print("\n  Config 로딩 실패로 나머지 테스트 중단")
                break

    total = len(steps)
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} steps passed")
    print("=" * 60)

    if passed == total:
        print(f"\n이 모델은 Phase 1 평가 파이프라인과 호환됩니다.")
        print(f"다음 단계:")
        print(f"  1. 평가 실행: python scripts/run_phase1_single.py {args.config_path}")
        print(f"  2. RunPod:   bash scripts/runpod_phase1.sh --config {args.config_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
