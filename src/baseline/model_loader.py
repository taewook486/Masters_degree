"""Unified model loading and inference for VLM zero-shot evaluation."""

from __future__ import annotations

import contextlib
import gc
import logging
from typing import Any

import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from PIL import Image

logger = logging.getLogger(__name__)

MEDICAL_PROMPT = (
    "You are a medical AI assistant. "
    "Look at this medical image and answer the following question.\n"
    "Question: {question}\n"
    "Answer concisely."
)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_config(config_path: str) -> DictConfig:
    """Load model config from YAML."""
    return OmegaConf.load(config_path)


@contextlib.contextmanager
def _florence2_compat():
    """Temporarily patch PretrainedConfig for Florence-2 + transformers 5.x.

    Florence-2's remote config accesses self.forced_bos_token_id before
    super().__init__() sets it.  transformers 5.x raises AttributeError
    for unset attributes.  This patches __getattribute__ to return None
    instead for those specific attributes.
    """
    from transformers.configuration_utils import PretrainedConfig

    original = PretrainedConfig.__getattribute__

    def _patched(self, key):
        try:
            return original(self, key)
        except AttributeError:
            if key in ("forced_bos_token_id", "forced_eos_token_id"):
                return None
            raise

    PretrainedConfig.__getattribute__ = _patched
    try:
        yield
    finally:
        PretrainedConfig.__getattribute__ = original


def _florence2_fix_missing_weights(model: Any, model_id: str) -> None:
    """Reload bias weights that transformers 5.x fails to map for Florence-2.

    transformers 5.x meta-device init drops many bias parameters during
    weight materialisation.  We detect all-zero params and reload them
    from the safetensors checkpoint.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    ckpt_path = hf_hub_download(model_id, filename="model.safetensors")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    fixed = 0
    with safe_open(ckpt_path, framework="pt", device=str(device)) as sf:
        ckpt_keys = set(sf.keys())
        for name, param in model.named_parameters():
            if param.numel() > 1 and param.abs().sum() == 0 and name in ckpt_keys:
                loaded = sf.get_tensor(name).to(dtype=dtype)
                param.data.copy_(loaded)
                fixed += 1

    logger.info(f"Florence-2: reloaded {fixed} missing weight tensors from checkpoint")


def load_model(config: DictConfig) -> tuple[Any, Any]:
    """Load model and processor based on config.

    Returns:
        Tuple of (model, processor).
    """
    model_id = config.model_id
    model_class_name = config.model_class
    processor_class_name = config.processor_class
    torch_dtype = DTYPE_MAP.get(config.torch_dtype, torch.float16)
    trust_remote_code = config.get("trust_remote_code", False)
    is_florence2 = "florence" in model_id.lower()

    logger.info(f"Loading model: {model_id} ({model_class_name})")

    # Get model class dynamically from transformers
    model_cls = getattr(transformers, model_class_name)
    processor_cls = getattr(transformers, processor_class_name)

    # Load model
    device_map = config.get("device_map", "auto")
    model_kwargs = {
        "torch_dtype": torch_dtype,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = config.attn_implementation

    # Florence-2: disable meta-device init to ensure all weights load correctly
    if is_florence2:
        model_kwargs["low_cpu_mem_usage"] = False

    # Florence-2 needs a compatibility patch for transformers 5.x
    _compat = _florence2_compat if is_florence2 else contextlib.nullcontext
    with _compat():
        model = model_cls.from_pretrained(model_id, **model_kwargs)

    # Florence-2: fix weight loading broken in transformers 5.x
    # 1. embed_tokens and lm_head must be tied to shared weight
    # 2. Many bias parameters fail to load (stay zero), must patch from checkpoint
    if is_florence2:
        lm = model.language_model
        if hasattr(lm.model, "shared"):
            lm.model.encoder.embed_tokens = lm.model.shared
            lm.model.decoder.embed_tokens = lm.model.shared
            if hasattr(lm, "lm_head"):
                lm.lm_head.weight = lm.model.shared.weight

        # Reload missing bias weights from checkpoint
        _florence2_fix_missing_weights(model, model_id)
        logger.info("Florence-2: patched weight tying + reloaded missing biases")

    # If no device_map, manually move to GPU
    if device_map is None and torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()

    # Load processor
    processor_kwargs = {}
    if trust_remote_code:
        processor_kwargs["trust_remote_code"] = True
    if config.get("processor_kwargs"):
        processor_kwargs.update(OmegaConf.to_container(config.processor_kwargs))

    with _compat():
        processor = processor_cls.from_pretrained(model_id, **processor_kwargs)

    logger.info(f"Model loaded on {model.device}, dtype={torch_dtype}")
    return model, processor


def generate_answer(
    model: Any,
    processor: Any,
    image: Image.Image,
    question: str,
    config: DictConfig,
) -> str:
    """Generate an answer from a VLM given an image and question.

    Routes to the correct prompt/inference path based on config.prompt_format.
    """
    prompt_format = config.get("prompt_format", "chat_template")
    gen_kwargs = OmegaConf.to_container(config.generation) if config.get("generation") else {}

    if prompt_format == "chat_template":
        return _generate_chat_template(model, processor, image, question, config, gen_kwargs)
    elif prompt_format == "direct_question":
        return _generate_direct_question(model, processor, image, question, config, gen_kwargs)
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format}")


def _generate_chat_template(
    model: Any,
    processor: Any,
    image: Image.Image,
    question: str,
    config: DictConfig,
    gen_kwargs: dict,
) -> str:
    """Generate answer using chat template (Qwen, SmolVLM, Gemma)."""
    prompt_text = MEDICAL_PROMPT.format(question=question)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    requires_vision_info = config.get("requires_vision_info_processing", False)

    if requires_vision_info:
        return _generate_qwen_style(model, processor, image, messages, gen_kwargs)
    else:
        return _generate_standard_chat(model, processor, image, messages, gen_kwargs)


def _generate_qwen_style(
    model: Any,
    processor: Any,
    image: Image.Image,
    messages: list[dict],
    gen_kwargs: dict,
) -> str:
    """Generate using Qwen-style VLM with process_vision_info."""
    from qwen_vl_utils import process_vision_info

    # Qwen expects the image in the message content
    messages[0]["content"][0] = {"type": "image", "image": image}

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Trim input tokens from output
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated_ids[0][input_len:]
    return processor.decode(output_ids, skip_special_tokens=True).strip()


def _generate_standard_chat(
    model: Any,
    processor: Any,
    image: Image.Image,
    messages: list[dict],
    gen_kwargs: dict,
) -> str:
    """Generate using standard chat template (SmolVLM, etc.)."""
    # Step 1: Build prompt text from chat template
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )

    # Step 2: Tokenize text + image separately
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    output_ids = generated_ids[0][input_len:]
    return processor.decode(output_ids, skip_special_tokens=True).strip()


def _generate_direct_question(
    model: Any,
    processor: Any,
    image: Image.Image,
    question: str,
    config: DictConfig,
    gen_kwargs: dict,
) -> str:
    """Generate using direct question prompt (Florence-2).

    Florence-2 uses task-specific prefixes, not free-form text.
    For VQA we use the <VQA> task token followed by the question.
    Output is post-processed via processor.post_process_generation().
    """
    torch_dtype = DTYPE_MAP.get(config.torch_dtype, torch.float16)
    task_prompt = "<VQA>"
    prompt = task_prompt + question

    # Florence-2's DaViT requires square images
    orig_size = (image.width, image.height)
    if image.width != image.height:
        size = max(image.width, image.height)
        image = image.resize((size, size))

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Cast pixel_values to the model's dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Florence-2 requires post-processing with skip_special_tokens=False
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=orig_size,
    )
    # Result is a dict like {"<VQA>": "answer text"}
    answer = parsed.get(task_prompt, generated_text)
    return answer.strip() if isinstance(answer, str) else str(answer).strip()


def unload_model(model: Any, processor: Any = None) -> None:
    """Unload model from GPU and free memory."""
    del model
    if processor is not None:
        del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded, GPU memory cleared")


# ---------------------------------------------------------------------------
# Batch inference (Fix 1: 2-4x throughput via batched generation)
# ---------------------------------------------------------------------------

def generate_answers_batch(
    model: Any,
    processor: Any,
    images: list,
    questions: list[str],
    config: DictConfig,
) -> list[str]:
    """Generate answers for a mini-batch of image-question pairs.

    Callers are responsible for chunking large datasets into batches.
    Falls back to per-sample inference for unsupported prompt formats.

    Args:
        model: Loaded VLM model.
        processor: Loaded processor.
        images: List of PIL Images for this batch.
        questions: List of question strings for this batch.
        config: Model DictConfig.

    Returns:
        List of answer strings, one per sample.
    """
    prompt_format = config.get("prompt_format", "chat_template")
    gen_kwargs = OmegaConf.to_container(config.generation) if config.get("generation") else {}

    if prompt_format == "chat_template":
        requires_vision_info = config.get("requires_vision_info_processing", False)
        if requires_vision_info:
            return _generate_qwen_style_batch(model, processor, images, questions, gen_kwargs)
        return _generate_standard_chat_batch(model, processor, images, questions, gen_kwargs)

    # Unsupported batch format (e.g. Florence-2 direct_question): fall back to single
    return [generate_answer(model, processor, img, q, config) for img, q in zip(images, questions)]


def _generate_qwen_style_batch(
    model: Any,
    processor: Any,
    images: list,
    questions: list[str],
    gen_kwargs: dict,
) -> list[str]:
    """Batch generation for Qwen-style VLMs (Qwen2.5-VL, Qwen3-VL)."""
    from qwen_vl_utils import process_vision_info

    texts: list[str] = []
    all_image_inputs: list = []

    for img, question in zip(images, questions):
        prompt_text = MEDICAL_PROMPT.format(question=question)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        texts.append(text)
        all_image_inputs.extend(image_inputs)

    # Ensure pad_token is set (Qwen uses eos_token as pad_token)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # Ensure left-padding for generation (required for correct batch trimming)
    orig_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        inputs = processor(
            text=texts,
            images=all_image_inputs,
            padding=True,
            return_tensors="pt",
        )
    finally:
        processor.tokenizer.padding_side = orig_padding_side

    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            pad_token_id=processor.tokenizer.pad_token_id,
            **gen_kwargs,
        )

    answers = []
    for i in range(len(images)):
        output_ids = generated_ids[i][input_len:]
        answers.append(processor.decode(output_ids, skip_special_tokens=True).strip())
    return answers


def _generate_standard_chat_batch(
    model: Any,
    processor: Any,
    images: list,
    questions: list[str],
    gen_kwargs: dict,
) -> list[str]:
    """Batch generation for standard chat template VLMs (SmolVLM2)."""
    texts: list[str] = []
    for question in questions:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": MEDICAL_PROMPT.format(question=question)},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        texts.append(text)

    # SmolVLM2 (Idefics3-based) requires images as list-of-lists: one list per sample.
    # A flat list is misinterpreted as multiple images for a single sample.
    images_nested = [[img] for img in images]

    # Ensure pad_token is set (SmolVLM2 may use eos_token as pad_token)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    orig_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        inputs = processor(text=texts, images=images_nested, return_tensors="pt", padding=True)
    finally:
        processor.tokenizer.padding_side = orig_padding_side

    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            pad_token_id=processor.tokenizer.pad_token_id,
            **gen_kwargs,
        )

    answers = []
    for i in range(len(images)):
        output_ids = generated_ids[i][input_len:]
        answers.append(processor.decode(output_ids, skip_special_tokens=True).strip())
    return answers
