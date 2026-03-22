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
import gc
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src.baseline.evaluate_zero_shot import evaluate_with_loaded_model
from src.baseline.model_loader import DTYPE_MAP, load_config
from src.finetune.prepare_data import prepare_chat_dataset, prepare_qwen_chat_dataset
from src.utils.seed import set_seed
from src.utils.vram_monitor import get_vram_usage, reset_peak_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unsloth detection & model compatibility
# ---------------------------------------------------------------------------

_UNSLOTH_SUPPORTED_PATTERNS = ["qwen2.5-vl", "qwen3-vl", "qwen2-vl"]


def _unsloth_available() -> bool:
    """Check if Unsloth is installed."""
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
    """Determine whether to use Unsloth backend."""
    if force_standard:
        return False
    return _unsloth_available() and _model_supports_unsloth(model_id)


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

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=t.get("num_train_epochs", 3),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=t.get("learning_rate", 2e-4),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.01),
        optim=t.get("optim", "paged_adamw_8bit"),
        fp16=t.get("fp16", True),
        bf16=t.get("bf16", False),
        logging_steps=t.get("logging_steps", 10),
        save_strategy=t.get("save_strategy", "epoch"),
        eval_strategy=ft_config.evaluation.get("eval_strategy", "epoch"),
        seed=seed,
        report_to="wandb",
        run_name=f"{model_name}_{dataset_name}_seed{seed}_unsloth",
        remove_unused_columns=False,
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=None,  # Unsloth VLM: None to avoid truncating image tokens
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

def _build_bnb_config(ft_config: DictConfig) -> BitsAndBytesConfig:
    """Build BitsAndBytesConfig from finetune YAML."""
    q = ft_config.quantization
    compute_dtype = DTYPE_MAP.get(q.get("bnb_4bit_compute_dtype", "float16"), torch.float16)
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

    bnb_config = _build_bnb_config(ft_config)

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
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=ft_config.training.get("gradient_checkpointing", True),
    )

    # Resolve target modules
    target_modules = list(lora.target_modules)
    if target_modules == ["minimal"]:
        target_modules = ["q_proj", "v_proj"]
    elif target_modules == ["medium"]:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif target_modules == ["full"] or target_modules == ["all_linear"]:
        target_modules = "all-linear"

    lora_config = LoraConfig(
        r=lora.get("rank", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        target_modules=target_modules,
        bias=lora.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

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


def _build_collate_fn(processor: Any, model_config: DictConfig, max_seq_length: int):
    """Build data collation function for standard PEFT backend."""
    is_qwen = model_config.get("requires_vision_info_processing", False)

    def collate_fn(examples: list[dict]) -> dict:
        texts = []
        images = []

        for ex in examples:
            msgs = ex["messages"]

            if is_qwen:
                text = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False,
                )
                from qwen_vl_utils import process_vision_info
                img_inputs, _ = process_vision_info(msgs)
                images.extend(img_inputs)
            else:
                text = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False,
                )
                images.append(ex["image"])

            texts.append(text)

        batch = processor(
            text=texts,
            images=images if images else None,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

    return collate_fn


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

    collate_fn = _build_collate_fn(processor, model_config, max_seq_length)

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=t.get("num_train_epochs", 3),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=t.get("learning_rate", 2e-4),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.01),
        optim=t.get("optim", "paged_adamw_8bit"),
        fp16=t.get("fp16", True),
        bf16=t.get("bf16", False),
        logging_steps=t.get("logging_steps", 10),
        save_strategy=t.get("save_strategy", "epoch"),
        eval_strategy=ft_config.evaluation.get("eval_strategy", "epoch"),
        seed=seed,
        report_to="wandb",
        run_name=f"{model_name}_{dataset_name}_seed{seed}_peft",
        remove_unused_columns=False,
        dataset_text_field=None,
        max_seq_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
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
    subset_ratio: float | None = None,
    eval_after_training: bool = True,
    force_standard: bool = False,
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
        subset_ratio: Use fraction of training data (Ablation A).
        eval_after_training: Run evaluation on test set after training.
        force_standard: Force standard PEFT backend even for Unsloth-supported models.

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
        eval_ds = prepare_fn(dataset_name, split="validation", data_dir=data_dir)
    except (ValueError, KeyError):
        logger.info(f"{dataset_name} has no validation split; using last 10% of train")
        n_eval = max(50, len(train_ds) // 10)
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

        eval_summary = evaluate_with_loaded_model(
            model=model,
            processor=processor,
            config=model_config,
            dataset_name=dataset_name,
            output_dir=str(output_path),
            seed=seed,
            data_dir=data_dir,
            batch_size=4,
        )
        result["eval_summary"] = eval_summary

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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

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
