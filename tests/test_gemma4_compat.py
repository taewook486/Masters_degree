"""Gemma 4 E2B 호환성 테스트.

기존 평가 파이프라인과의 호환성을 단계별로 검증:
1. Config 로딩
2. transformers 클래스 해석
3. Processor 로드 + chat_template 확인
4. 모델 로드 + 단일 샘플 추론 (GPU 필요)
"""

from __future__ import annotations

import sys
import traceback

import torch


def step1_config_loading() -> dict:
    """Step 1: YAML config 로딩 테스트."""
    from src.baseline.model_loader import load_config

    config = load_config("configs/models/gemma4_e2b.yaml")
    assert config.model_name == "gemma4-e2b"
    assert config.model_id == "google/gemma-4-E2B-it"
    assert config.model_class == "Gemma4ForConditionalGeneration"
    assert config.prompt_format == "chat_template"
    assert config.get("requires_vision_info_processing", False) is False
    print(f"  model_name: {config.model_name}")
    print(f"  model_id: {config.model_id}")
    print(f"  model_class: {config.model_class}")
    print(f"  torch_dtype: {config.torch_dtype}")
    print(f"  prompt_format: {config.prompt_format}")
    return config


def step2_class_resolution() -> None:
    """Step 2: transformers 클래스 해석 테스트."""
    import transformers

    model_cls = getattr(transformers, "Gemma4ForConditionalGeneration", None)
    processor_cls = getattr(transformers, "AutoProcessor", None)
    assert model_cls is not None, "Gemma4ForConditionalGeneration not found"
    assert processor_cls is not None, "AutoProcessor not found"
    print(f"  model_cls: {model_cls}")
    print(f"  processor_cls: {processor_cls}")


def step3_processor_and_template() -> None:
    """Step 3: Processor 로드 + chat_template 동작 확인."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")

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
    assert isinstance(text, str) and len(text) > 0
    print(f"  chat_template output (first 200 chars): {text[:200]}")

    # pad_token 확인
    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id
    print(f"  pad_token_id: {pad_id}, eos_token_id: {eos_id}")


def step4_model_load_and_inference() -> None:
    """Step 4: 모델 로드 + 단일 샘플 추론 (GPU 필요)."""
    from PIL import Image

    from src.baseline.model_loader import generate_answer, load_config, load_model, unload_model

    config = load_config("configs/models/gemma4_e2b.yaml")
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
    steps = [
        ("Step 1: Config loading", step1_config_loading),
        ("Step 2: Class resolution", step2_class_resolution),
        ("Step 3: Processor + chat_template", step3_processor_and_template),
        ("Step 4: Model load + inference", step4_model_load_and_inference),
    ]

    passed = 0
    for name, fn in steps:
        print(f"\n{'='*60}")
        print(f"{name}")
        print("=" * 60)
        try:
            fn()
            print(f"  -> PASS")
            passed += 1
        except Exception as e:
            print(f"  -> FAIL: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{len(steps)} steps passed")
    print("=" * 60)
    return 0 if passed == len(steps) else 1


if __name__ == "__main__":
    sys.exit(main())
