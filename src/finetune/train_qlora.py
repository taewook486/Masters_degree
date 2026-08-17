"""QLoRA fine-tuning of VLMs for medical VQA.

Two backends with automatic selection:
  1. Unsloth (preferred): 2-5x faster, ~60% less VRAM. Used for Qwen2.5-VL, Qwen3-VL.
  2. Standard HF PEFT + TRL: fallback for unsupported models (SmolVLM2, etc.).

Hardware target: RTX 5060 Ti (16GB VRAM).

Usage:
    python -m src.finetune.train_qlora \
        --model_config configs/models/qwen25_vl_3b.yaml \
        --finetune_config configs/finetune/base_qlora.yaml \
        --dataset pathvqa \
        --output_dir results/phase2_finetune/qwen25-vl-3b_pathvqa_seed42 \
        --seed 42
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import os

# [HARD] Unsloth must be imported BEFORE transformers/trl/peft. Otherwise unsloth's
# runtime patching overwrites the tokenizer eos_token/pad_token to the placeholder
# '<EOS_TOKEN>', which trl 0.24 SFTTrainer then rejects as "not in vocabulary"
# (unslothai/unsloth#2797). Import order is the documented fix.
#
# BUT unsloth import also monkey-patches trl.SFTTrainer GLOBALLY, which hijacks the
# standard (non-unsloth) backend's native VLM collation. So we import unsloth ONLY when
# the caller has NOT set MOAI_SKIP_UNSLOTH=1. train_one.py sets this env for standard
# models to keep them on pure trl; unsloth models (qwen) leave it unset so the eos fix
# and unsloth backend work. Phase 3/direct callers (env unset) keep the eos fix intact.
if os.environ.get("MOAI_SKIP_UNSLOTH") == "1":
    unsloth = None
else:
    try:
        import unsloth  # noqa: F401
    except ImportError:
        unsloth = None

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

from datasets.exceptions import DatasetGenerationError

from src.baseline.evaluate_zero_shot import evaluate_with_loaded_model
from src.baseline.model_loader import DTYPE_MAP, load_config
from src.finetune.prepare_data import prepare_chat_dataset, prepare_qwen_chat_dataset
from src.utils.logging_config import setup_logging
from src.utils.seed import set_seed
from src.utils.vram_monitor import get_vram_usage, reset_peak_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time budget callback for HPO
# ---------------------------------------------------------------------------


class TimeBudgetCallback(TrainerCallback):
    """Stop training when time budget is exceeded."""

    def __init__(self, budget_min: float):
        self.budget_sec = budget_min * 60
        self.budget_min = budget_min
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed >= self.budget_sec:
            logger.info(
                f"Time budget exceeded ({elapsed / 60:.1f}min >= "
                f"{self.budget_min:.1f}min). Stopping at step {state.global_step}."
            )
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# Unsloth detection & model compatibility
# ---------------------------------------------------------------------------

# Gemma4-E2B는 standard backend로 처리한다. unsloth가 google/gemma-4-E2B-it를 미지원하고
# ("not supported in your current Unsloth version"), PEFT의 Gemma4ClippableLinear 거부
# (huggingface/peft#3129)는 _load_model_standard에서 LoRA 타깃을 실제 nn.Linear(텍스트
# 모델)로 한정해 해결한다(vision/audio 타워의 ClippableLinear 자동 제외).
_UNSLOTH_SUPPORTED_PATTERNS = ["qwen2.5-vl", "qwen3-vl", "qwen2-vl"]


def _unsloth_available() -> bool:
    """Check if Unsloth is installed (and not explicitly skipped)."""
    if os.environ.get("MOAI_SKIP_UNSLOTH") == "1":
        return False
    try:
        from unsloth import FastVisionModel  # noqa: F401
        return True
    except ImportError:
        return False


def _model_supports_unsloth(model_id: str) -> bool:
    """Check if model is in Unsloth's supported VLM list."""
    model_id_lower = model_id.lower()
    return any(pat in model_id_lower for pat in _UNSLOTH_SUPPORTED_PATTERNS)


def _should_use_unsloth(model_id: str, force_standard: bool = False) -> bool:
    """Determine whether to use Unsloth backend.

    Pattern match FIRST (cheap, no import) so that non-unsloth models never trigger
    `_unsloth_available()`, which would import unsloth and globally patch trl.SFTTrainer.
    """
    if force_standard:
        return False
    if not _model_supports_unsloth(model_id):
        return False
    return _unsloth_available()


# ---------------------------------------------------------------------------
# Unsloth backend
# ---------------------------------------------------------------------------

def _load_model_unsloth(
    model_config: DictConfig,
    ft_config: DictConfig,
) -> tuple[Any, Any, dict]:
    """Load model via Unsloth FastVisionModel for QLoRA training.

    Returns:
        Tuple of (model, processor, lora_info_dict).
    """
    from unsloth import FastVisionModel

    model_id = model_config.model_id
    lora = ft_config.lora
    torch_dtype = DTYPE_MAP.get(model_config.torch_dtype, torch.float16)

    logger.info(f"[Unsloth] Loading {model_id} with 4-bit quantization")

    model, processor = FastVisionModel.from_pretrained(
        model_id,
        dtype=torch_dtype,
        load_in_4bit=ft_config.quantization.get("load_in_4bit", True),
        use_gradient_checkpointing="unsloth",
    )

    # Resolve target modules
    target_modules = list(lora.target_modules)
    if target_modules == ["minimal"]:
        target_modules = ["q_proj", "v_proj"]
    elif target_modules == ["medium"]:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif target_modules == ["full"] or target_modules == ["all_linear"]:
        target_modules = "all-linear"

    # Apply LoRA via Unsloth
    model = FastVisionModel.get_peft_model(
        model,
        r=lora.get("rank", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=target_modules,
        bias=lora.get("bias", "none"),
        finetune_vision_layers=True,
        finetune_language_layers=True,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0
    logger.info(
        f"[Unsloth] LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
        f"({trainable_pct:.2f}%)"
    )

    # Ensure pad_token is set
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    lora_info = {
        "rank": lora.get("rank", 16),
        "alpha": lora.get("alpha", 32),
        "dropout": lora.get("dropout", 0.05),
        "target_modules": target_modules if isinstance(target_modules, list) else target_modules,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(trainable_pct, 2),
    }

    return model, processor, lora_info


def _build_trainer_unsloth(
    model: Any,
    processor: Any,
    model_config: DictConfig,
    ft_config: DictConfig,
    train_ds: Any,
    eval_ds: Any,
    output_dir: str,
    seed: int,
    model_name: str,
    dataset_name: str,
) -> SFTTrainer:
    """Build SFTTrainer with Unsloth's optimized data collator."""
    from unsloth import UnslothVisionDataCollator

    t = ft_config.training
    output_path = Path(output_dir) / "checkpoints"

    # v0.2: Support max_steps (Phase 3) or num_train_epochs (Phase 2)
    max_steps_val = t.get("max_steps", -1)
    epochs_val = t.get("num_train_epochs", 3) if max_steps_val <= 0 else 1
    save_eval_strategy = "steps" if max_steps_val > 0 else t.get("save_strategy", "epoch")

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs_val,
        max_steps=max_steps_val,
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=t.get("learning_rate", 2e-4),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.01),
        optim=t.get("optim", "paged_adamw_8bit"),
        # 모델 dtype에 맞춰 정밀도 자동 설정 (bf16 모델에 fp16 강제 시 crash: smolvlm2/gemma4)
        fp16=DTYPE_MAP.get(model_config.torch_dtype, torch.float16) != torch.bfloat16,
        bf16=DTYPE_MAP.get(model_config.torch_dtype, torch.float16) == torch.bfloat16,
        logging_steps=t.get("logging_steps", 10),
        save_strategy=save_eval_strategy,
        # eval_loss는 어디에서도 소비되지 않는다 — results.tsv 컬럼에도, TrialResult
        # 필드에도, 논문에도 없고 HPO 에이전트도 보지 않는다. 그런데 검증셋 6,259건
        # 전량을 도느라 trial당 약 11.6분이 든다(실측 647~718초, 하이퍼파라미터와
        # 무관한 고정 비용으로 train_runtime의 41.8%). save_strategy와 변수를
        # 공유하던 것을 분리해 평가만 끈다. 가중치와 val_accuracy에는 영향이 없다
        # (정확도는 학습 후 별도 500샘플 평가에서 산출).
        eval_strategy="no",
        # max_steps 사용 시 끝에서 1회만 save (미지정 시 logging_steps=10마다 → 낭비).
        # epoch 전략일 땐 무시됨.
        save_steps=max_steps_val if max_steps_val > 0 else 500,
        eval_steps=max_steps_val if max_steps_val > 0 else 500,
        # epoch 전략(Ablation C 등, 3에폭)에서 체크포인트가 누적돼 디스크 quota를 채우는 것을 방지.
        # 최종 가중치는 학습 후 별도로 adapter/에 저장되므로 checkpoints/는 최신 1개만 있으면 충분.
        save_total_limit=1,
        seed=seed,
        report_to="wandb",
        run_name=f"{model_name}_{dataset_name}_seed{seed}_unsloth",
        remove_unused_columns=False,
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,  # Unsloth VLM: None to avoid truncating image tokens (trl 0.24: max_seq_length→max_length)
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=UnslothVisionDataCollator(model, processor),
        processing_class=processor.tokenizer,
    )

    return trainer


# ---------------------------------------------------------------------------
# Standard HF PEFT backend (fallback)
# ---------------------------------------------------------------------------

def _build_bnb_config(ft_config: DictConfig, compute_dtype_override: Any = None) -> BitsAndBytesConfig:
    """Build BitsAndBytesConfig from finetune YAML.

    compute_dtype_override: 지정 시 bnb 4bit 연산 dtype을 강제(모델 dtype과 정합).
    """
    q = ft_config.quantization
    compute_dtype = compute_dtype_override or DTYPE_MAP.get(
        q.get("bnb_4bit_compute_dtype", "float16"), torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=q.get("load_in_4bit", True),
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
    )


def _load_model_standard(
    model_config: DictConfig,
    ft_config: DictConfig,
) -> tuple[Any, Any, dict]:
    """Load model with standard HF PEFT QLoRA (fallback for non-Unsloth models).

    Returns:
        Tuple of (model, processor, lora_info_dict).
    """
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model_id = model_config.model_id
    model_class_name = model_config.model_class
    processor_class_name = model_config.processor_class
    trust_remote_code = model_config.get("trust_remote_code", False)
    torch_dtype = DTYPE_MAP.get(model_config.torch_dtype, torch.float16)
    lora = ft_config.lora

    logger.info(f"[Standard PEFT] Loading {model_id} with 4-bit quantization")

    model_cls = getattr(transformers, model_class_name)
    processor_cls = getattr(transformers, processor_class_name)

    bnb_config = _build_bnb_config(ft_config, compute_dtype_override=torch_dtype)

    model_kwargs: dict[str, Any] = {
        "quantization_config": bnb_config,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if model_config.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_config.attn_implementation

    model = model_cls.from_pretrained(model_id, **model_kwargs)

    # device_map="auto"가 모델을 여러 GPU에 나눠 올렸을 때(model-parallel), HF Trainer는
    # 기본적으로 model.model_parallel 플래그를 못 찾으면 "이미 병렬화됐다"는 걸 인식 못 하고
    # torch.cuda.device_count()>1을 보고 별도로 nn.DataParallel로 한 번 더 감싸려 한다.
    # 그러면 "module must have its parameters and buffers on device cuda:0 ... found on
    # cuda:1" 충돌이 난다(모델이 이미 여러 GPU에 나뉘어 있는데 DataParallel은 원본이 단일
    # 디바이스에 있길 요구). 표준 해결책(QLoRA 멀티GPU 튜토리얼에서 통용): 로드 직후 이 두
    # 플래그를 세팅해 Trainer가 DataParallel로 재래핑하지 않도록 한다.
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # prepare_model_for_kbit_training은 4bit(Params4bit)가 아닌 모든 fp16/bf16 파라미터를
    # 블랑켓으로 fp32 업캐스트한다(의도는 LayerNorm 안정화지만 실제로는 전체를 훑음).
    # bnb 4bit 양자화는 nn.Linear만 바꾸고 nn.Embedding은 안 건드려 bf16으로 남는데,
    # Gemma4는 거대한 vocab 임베딩을 2개(embed_tokens + embed_tokens_per_layer) 쓴다.
    # 이 임베딩들은 LoRA 타깃이 아니라 완전히 frozen이라 fp32로 올릴 이유가 없는데,
    # 업캐스트 시 단일 텐서가 ~8.75GiB를 요구해 GPU당 16GB 카드에서 OOM
    # (model-parallel 분산으로도 해결 안 됨 — in-place 캐스트라 파라미터가 있는
    # GPU에서 그대로 부족). OOM은 peft 함수 "내부"의 캐스트 순간 발생하므로 함수
    # 호출 후 되돌리는 방식은 늦다 — 호출 전에 해당 임베딩을 CPU로 옮겨 fp32
    # 업캐스트가 CPU RAM(124GB, 여유 충분)에서 일어나게 하고, 끝난 뒤 원래
    # device·dtype으로 되돌린다.
    _frozen_embeds = [m for m in model.modules() if isinstance(m, torch.nn.Embedding)]
    _embed_devices = {id(m): m.weight.device for m in _frozen_embeds}
    for _emb in _frozen_embeds:
        _emb.weight.data = _emb.weight.data.to("cpu")

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=ft_config.training.get("gradient_checkpointing", True),
    )

    for _emb in _frozen_embeds:
        _emb.weight.data = _emb.weight.data.to(device=_embed_devices[id(_emb)], dtype=torch_dtype)

    # Resolve target modules
    target_modules = list(lora.target_modules)
    if target_modules == ["minimal"]:
        target_modules = ["q_proj", "v_proj"]
    elif target_modules == ["medium"]:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif target_modules == ["full"] or target_modules == ["all_linear"]:
        target_modules = "all-linear"

    # 접미사 리스트(q_proj 등)를 실제 nn.Linear 모듈의 전체 경로로 해석한다.
    # Gemma4는 vision/audio 타워에 nn.Linear가 아닌 래퍼(Gemma4ClippableLinear)를 같은
    # 이름(q_proj/v_proj)으로 써서, 단순 접미사 매칭 시 PEFT가 이를 LoRA 대상으로 잡으려다
    # 거부한다(peft#3129). isinstance(nn.Linear) 필터로 텍스트 모델의 (4bit 포함) Linear만
    # 남긴다 — bnb Linear4bit는 nn.Linear 서브클래스라 유지되고 ClippableLinear는 제외된다.
    # (VLM QLoRA 표준: vision/audio 인코더 freeze, 언어 모델만 적응). "all-linear"는 PEFT가
    # 자체 처리하므로 건너뛴다.
    if isinstance(target_modules, list):
        _suffixes = set(target_modules)
        _resolved = [
            name for name, mod in model.named_modules()
            if isinstance(mod, torch.nn.Linear) and name.rsplit(".", 1)[-1] in _suffixes
        ]
        if _resolved:
            target_modules = _resolved
            logger.info(
                f"[Standard PEFT] Resolved {len(_resolved)} nn.Linear target modules "
                f"(non-Linear wrappers e.g. Gemma4ClippableLinear excluded)"
            )

    lora_config = LoraConfig(
        r=lora.get("rank", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=target_modules,
        bias=lora.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # VLM dtype 정합 패치: 학습 첫 forward에서 inputs_merger가
    #   image_embeds[image_mask] = image_hidden_states[...]  (modeling_smolvlm.py L516)
    # 를 수행하는데, autocast 하에서 vision post_layernorm이 float32를 출력해
    # image_hidden_states가 fp32가 되고 bf16 텍스트 임베딩(image_embeds)과 어긋나 실패한다
    # (RuntimeError: Index put ... BFloat16 dest, Float source).
    # get_image_features 출력을 모델 dtype으로 캐스트해 병합 dtype을 맞춘다.
    # [중요] SmolVLM은 get_image_features가 텐서가 아니라 BaseModelOutputWithPooling 객체를
    # 반환하고 forward는 .pooler_output을 쓴다(L631). Gemma4는 텐서를 반환한다. 두 경우를 모두
    # 처리: 텐서면 직접 캐스트, ModelOutput이면 pooler_output/last_hidden_state를 캐스트.
    for _submodule in model.modules():
        if hasattr(_submodule, "get_image_features") and hasattr(_submodule, "inputs_merger"):
            _orig_get_image_features = _submodule.get_image_features

            def _cast_image_features(*a, __orig=_orig_get_image_features, __dt=torch_dtype, **k):
                out = __orig(*a, **k)
                if isinstance(out, torch.Tensor):
                    return out.to(__dt)
                for _attr in ("pooler_output", "last_hidden_state"):
                    _v = getattr(out, _attr, None)
                    if isinstance(_v, torch.Tensor):
                        setattr(out, _attr, _v.to(__dt))
                return out

            _submodule.get_image_features = _cast_image_features
            logger.info("[Standard PEFT] Patched get_image_features to cast image features to model dtype (VLM merge fix)")
            break

    trainable_params, total_params = model.get_nb_trainable_parameters()
    trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0
    logger.info(
        f"[Standard PEFT] LoRA applied: {trainable_params:,} trainable / {total_params:,} total "
        f"({trainable_pct:.2f}%)"
    )

    # Load processor
    processor_kwargs = {}
    if trust_remote_code:
        processor_kwargs["trust_remote_code"] = True
    if model_config.get("processor_kwargs"):
        processor_kwargs.update(OmegaConf.to_container(model_config.processor_kwargs))

    processor = processor_cls.from_pretrained(model_id, **processor_kwargs)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    lora_info = {
        "rank": lora.get("rank", 16),
        "alpha": lora.get("alpha", 32),
        "dropout": lora.get("dropout", 0.05),
        "target_modules": target_modules if isinstance(target_modules, list) else target_modules,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(trainable_pct, 2),
    }

    return model, processor, lora_info


def _build_trainer_standard(
    model: Any,
    processor: Any,
    model_config: DictConfig,
    ft_config: DictConfig,
    train_ds: Any,
    eval_ds: Any,
    output_dir: str,
    seed: int,
    model_name: str,
    dataset_name: str,
) -> SFTTrainer:
    """Build SFTTrainer with standard HF collation."""
    t = ft_config.training
    max_seq_length = t.get("max_seq_length", 2048)
    output_path = Path(output_dir) / "checkpoints"

    # v0.2: Support max_steps (Phase 3) or num_train_epochs (Phase 2)
    max_steps_val = t.get("max_steps", -1)
    epochs_val = t.get("num_train_epochs", 3) if max_steps_val <= 0 else 1
    save_eval_strategy = "steps" if max_steps_val > 0 else t.get("save_strategy", "epoch")

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=epochs_val,
        max_steps=max_steps_val,
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=t.get("learning_rate", 2e-4),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.01),
        optim=t.get("optim", "paged_adamw_8bit"),
        # 모델 dtype에 맞춰 정밀도 자동 설정 (bf16 모델에 fp16 강제 시 crash: smolvlm2/gemma4)
        fp16=DTYPE_MAP.get(model_config.torch_dtype, torch.float16) != torch.bfloat16,
        bf16=DTYPE_MAP.get(model_config.torch_dtype, torch.float16) == torch.bfloat16,
        logging_steps=t.get("logging_steps", 10),
        save_strategy=save_eval_strategy,
        # eval_loss는 어디에서도 소비되지 않는다 — results.tsv 컬럼에도, TrialResult
        # 필드에도, 논문에도 없고 HPO 에이전트도 보지 않는다. 그런데 검증셋 6,259건
        # 전량을 도느라 trial당 약 11.6분이 든다(실측 647~718초, 하이퍼파라미터와
        # 무관한 고정 비용으로 train_runtime의 41.8%). save_strategy와 변수를
        # 공유하던 것을 분리해 평가만 끈다. 가중치와 val_accuracy에는 영향이 없다
        # (정확도는 학습 후 별도 500샘플 평가에서 산출).
        eval_strategy="no",
        # max_steps 사용 시 끝에서 1회만 save (미지정 시 logging_steps=10마다 → 낭비).
        # epoch 전략일 땐 무시됨.
        save_steps=max_steps_val if max_steps_val > 0 else 500,
        eval_steps=max_steps_val if max_steps_val > 0 else 500,
        # epoch 전략(Ablation C 등, 3에폭)에서 체크포인트가 누적돼 디스크 quota를 채우는 것을 방지.
        # 최종 가중치는 학습 후 별도로 adapter/에 저장되므로 checkpoints/는 최신 1개만 있으면 충분.
        save_total_limit=1,
        seed=seed,
        report_to="wandb",
        run_name=f"{model_name}_{dataset_name}_seed{seed}_peft",
        remove_unused_columns=False,
        max_length=max_seq_length,  # trl 0.24: max_seq_length→max_length
    )

    # trl 0.24 native VLM 경로: data_collator=None → 데이터셋에 "images" 컬럼이 있으면
    # SFTTrainer가 자동으로 DataCollatorForVisionLanguageModeling(processor)을 생성하고
    # vision 데이터셋은 prepare를 자동 skip한다. 커스텀 collate_fn을 넘기던 방식은
    # trl 0.24에서 raw 컬럼이 tokenizer.pad()로 넘어가 실패했음("supply input_ids").
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,  # ProcessorMixin → trl이 VLM으로 인식
    )

    return trainer


# ---------------------------------------------------------------------------
# Main training function (unified entry point)
# ---------------------------------------------------------------------------

def train_qlora(
    model_config_path: str,
    finetune_config_path: str,
    dataset_name: str,
    output_dir: str,
    seed: int = 42,
    data_dir: str = "data",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
    subset_ratio: float | None = None,
    eval_after_training: bool = True,
    force_standard: bool = False,
    time_budget_min: float | None = None,
    measure_cf: bool = False,
    base_vqav2_result: dict | None = None,
) -> dict:
    """Run QLoRA fine-tuning for one model on one dataset.

    Automatically selects Unsloth backend for supported models (Qwen VL),
    falls back to standard HF PEFT for others (SmolVLM2).

    Args:
        model_config_path: Path to model config YAML.
        finetune_config_path: Path to finetune config YAML.
        dataset_name: One of "pathvqa", "slake", "vqa_rad".
        output_dir: Directory to save adapter weights and results.
        seed: Random seed for reproducibility.
        data_dir: Base directory for datasets.
        max_train_samples: Limit training samples (for debugging).
        max_eval_samples: Limit validation samples used for epoch eval /
            best-checkpoint selection. 최종 보고 지표는 학습 후 test셋 평가라
            이 값은 결과 수치에 영향을 주지 않고 학습 중 평가 비용만 줄인다.
        subset_ratio: Use fraction of training data (Ablation A).
        eval_after_training: Run evaluation on test set after training.
        force_standard: Force standard PEFT backend even for Unsloth-supported models.
        time_budget_min: Max training time in minutes (None = no limit).
        measure_cf: If True, measure Catastrophic Forgetting on VQAv2 after training.
        base_vqav2_result: Pre-computed base model VQAv2 result (for CF).

    Returns:
        Dictionary with training metrics and evaluation results.
    """
    set_seed(seed)
    reset_peak_stats()
    train_start = time.time()

    model_config = load_config(model_config_path)
    ft_config = OmegaConf.load(finetune_config_path)
    model_name = model_config.model_name
    model_id = model_config.model_id

    # Select backend
    use_unsloth = _should_use_unsloth(model_id, force_standard)
    backend_label = "Unsloth" if use_unsloth else "Standard PEFT"
    logger.info(f"=== QLoRA Training [{backend_label}]: {model_name} on {dataset_name} (seed={seed}) ===")

    # Load model + LoRA
    if use_unsloth:
        model, processor, lora_info = _load_model_unsloth(model_config, ft_config)
    else:
        model, processor, lora_info = _load_model_standard(model_config, ft_config)

    # Prepare datasets
    is_qwen = model_config.get("requires_vision_info_processing", False)
    prepare_fn = prepare_qwen_chat_dataset if is_qwen else prepare_chat_dataset

    train_ds = prepare_fn(
        dataset_name, split="train", data_dir=data_dir,
        max_samples=max_train_samples, subset_ratio=subset_ratio,
    )

    try:
        eval_ds = prepare_fn(
            dataset_name, split="validation", data_dir=data_dir,
            max_samples=max_eval_samples,
        )
    except (ValueError, KeyError, DatasetGenerationError) as e:
        # Dataset.from_generator()로 스트리밍 빌드(997238f)한 이후, split 없음 같은
        # ValueError가 제너레이터 내부에서 나면 datasets가 DatasetGenerationError로
        # 감싸서 던진다 — 원래 잡던 ValueError/KeyError가 더는 안 잡혀 조용히 죽는
        # 회귀였다(Phase 2 Main 2026-07-19 vqa_rad 실제 재현). __cause__를 까서
        # 진짜 "split 없음"(ValueError/KeyError)일 때만 대체하고, 그 외(디스크 풀 등
        # 진짜 에러)는 숨기지 않고 그대로 재발생시킨다.
        cause = e.__cause__ if isinstance(e, DatasetGenerationError) else e
        if not isinstance(cause, (ValueError, KeyError)):
            raise
        logger.info(f"{dataset_name} has no validation split; using last 10% of train")
        n_eval = max(50, len(train_ds) // 10)
        if max_eval_samples is not None:
            n_eval = min(n_eval, max_eval_samples)
        # train 크기를 넘지 않도록 clamp (소량 스모크에서 n_eval > len(train) 방지)
        n_eval = min(n_eval, max(1, len(train_ds) // 2))
        eval_ds = train_ds.select(range(len(train_ds) - n_eval, len(train_ds)))
        train_ds = train_ds.select(range(len(train_ds) - n_eval))

    logger.info(f"Train: {len(train_ds)} samples, Eval: {len(eval_ds)} samples")

    # Build trainer
    if use_unsloth:
        trainer = _build_trainer_unsloth(
            model, processor, model_config, ft_config,
            train_ds, eval_ds, output_dir, seed, model_name, dataset_name,
        )
    else:
        trainer = _build_trainer_standard(
            model, processor, model_config, ft_config,
            train_ds, eval_ds, output_dir, seed, model_name, dataset_name,
        )

    # Add time budget callback if specified
    if time_budget_min and time_budget_min > 0:
        trainer.add_callback(TimeBudgetCallback(time_budget_min))
        logger.info(f"Time budget: {time_budget_min:.1f} minutes")

    # Train
    logger.info(f"Starting training with {backend_label} backend...")
    train_result = trainer.train()
    train_time_min = (time.time() - train_start) / 60

    # Save adapter weights
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    adapter_path = output_path / "adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    logger.info(f"Adapter saved to {adapter_path}")

    # 학습용 raw 체크포인트(옵티마이저 상태 포함, adapter/보다 훨씬 큼)는
    # 최종 가중치가 adapter/에 이미 저장됐으므로 더 이상 필요 없음 → 디스크 quota 방지를 위해 삭제.
    checkpoints_path = output_path / "checkpoints"
    if checkpoints_path.exists():
        shutil.rmtree(checkpoints_path, ignore_errors=True)
        logger.info(f"Removed redundant checkpoints dir: {checkpoints_path}")

    # Collect training metrics
    vram = get_vram_usage()

    result = {
        "metadata": {
            "model_name": model_name,
            "model_id": model_id,
            "dataset": dataset_name,
            "seed": seed,
            "subset_ratio": subset_ratio,
            "backend": backend_label,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "lora_config": {
            "rank": lora_info["rank"],
            "alpha": lora_info["alpha"],
            "dropout": lora_info["dropout"],
            "target_modules": (
                list(lora_info["target_modules"])
                if isinstance(lora_info["target_modules"], (list, tuple))
                else lora_info["target_modules"]
            ),
        },
        "training": {
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds),
            "train_loss": train_result.metrics.get("train_loss"),
            "train_runtime_sec": train_result.metrics.get("train_runtime"),
            "train_time_min": round(train_time_min, 1),
            "trainable_params": lora_info["trainable_params"],
            "total_params": lora_info["total_params"],
            "trainable_pct": lora_info["trainable_pct"],
            "peak_vram_mb": vram["peak_mb"],
        },
    }

    # Post-training evaluation on test set
    if eval_after_training:
        logger.info("Running post-training evaluation on test set...")
        if use_unsloth:
            from unsloth import FastVisionModel
            FastVisionModel.for_inference(model)
        else:
            model = model.merge_and_unload()
        model.eval()

        # standard backend는 prepare_model_for_kbit_training이 lm_head/norm을 fp32로 두어
        # (LoRA 병합 후에도 유지) autocast 없는 generation에서 bf16 hidden state와 충돌한다
        # (F.linear: expected scalar type Float but found BFloat16). 학습은 Trainer의 autocast
        # 덕에 통과하므로, 평가·CF generation도 동일하게 autocast로 감싼다.
        # unsloth 경로는 FastVisionModel.for_inference가 dtype을 정리하므로 감싸지 않는다.
        _compute_dtype = DTYPE_MAP.get(model_config.torch_dtype, torch.float16)
        _eval_ctx = (
            contextlib.nullcontext() if use_unsloth
            else torch.autocast("cuda", dtype=_compute_dtype)
        )
        with _eval_ctx:
            eval_summary = evaluate_with_loaded_model(
                model=model,
                processor=processor,
                config=model_config,
                dataset_name=dataset_name,
                output_dir=str(output_path),
                seed=seed,
                data_dir=data_dir,
                batch_size=4,
                max_samples=max_test_samples,  # 스모크: 소량으로 제한 (None=full test, 논문용)
                # REQ-EM-001: Phase 2는 BioBERT 이중 BERTScore를 opt-in으로 활성화한다
                # (§4.4). roberta-large는 REQ-EM-004에 따라 항상 primary로 유지된다.
                bertscore_models=["roberta-large", "dmis-lab/biobert-v1.1"],
            )
            result["eval_summary"] = eval_summary

            # Catastrophic Forgetting measurement (v0.2)
            if measure_cf and base_vqav2_result is not None:
                try:
                    from src.evaluate.catastrophic_forgetting import run_cf_measurement

                    cf_result = run_cf_measurement(
                        model=model,
                        processor=processor,
                        config=model_config,
                        base_vqav2_result=base_vqav2_result,
                        output_dir=output_dir,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        seed=seed,
                        data_dir=data_dir,
                    )
                    result["catastrophic_forgetting"] = cf_result.get("summary", {})
                except Exception as e:
                    logger.warning(f"[CF] Measurement failed: {e}")

    # Save result
    result_file = output_path / "train_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Training result saved to {result_file}")

    # Cleanup
    del model, processor, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for medical VQA")
    parser.add_argument("--model_config", required=True, help="Path to model config YAML")
    parser.add_argument(
        "--finetune_config", default="configs/finetune/base_qlora.yaml",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["pathvqa", "slake", "vqa_rad"],
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument(
        "--subset_ratio", type=float, default=None,
        help="Fraction of training data (Ablation A: 0.05, 0.1, 0.25, 0.5, 1.0)",
    )
    parser.add_argument("--no_eval", action="store_true", help="Skip post-training eval")
    parser.add_argument(
        "--force_standard", action="store_true",
        help="Force standard HF PEFT even if Unsloth is available",
    )
    args = parser.parse_args()

    setup_logging(log_dir=args.output_dir, experiment_name="train_qlora")

    train_qlora(
        model_config_path=args.model_config,
        finetune_config_path=args.finetune_config,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        seed=args.seed,
        data_dir=args.data_dir,
        max_train_samples=args.max_train_samples,
        subset_ratio=args.subset_ratio,
        eval_after_training=not args.no_eval,
        force_standard=args.force_standard,
    )


if __name__ == "__main__":
    main()
