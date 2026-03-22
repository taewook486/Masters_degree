"""Convert medical VQA datasets to chat-format for SFT training.

Each sample is converted to a conversation turn:
  User: <image> + medical prompt + question
  Assistant: answer

Supports both Qwen-style (with vision info) and standard chat template formats.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

from src.data.dataset import load_medical_vqa_dataset

logger = logging.getLogger(__name__)

MEDICAL_PROMPT = (
    "You are a medical AI assistant. "
    "Look at this medical image and answer the following question.\n"
    "Question: {question}\n"
    "Answer concisely."
)


def prepare_chat_dataset(
    dataset_name: str,
    split: str,
    data_dir: str = "data",
    max_samples: int | None = None,
    subset_ratio: float | None = None,
) -> Dataset:
    """Load a medical VQA dataset and convert to HuggingFace Dataset with chat columns.

    Args:
        dataset_name: One of "pathvqa", "slake", "vqa_rad".
        split: Dataset split ("train", "validation", "test").
        data_dir: Base directory where datasets are stored.
        max_samples: Hard limit on number of samples (for debugging).
        subset_ratio: Fraction of data to use (0.0-1.0), for Ablation Study A.

    Returns:
        HuggingFace Dataset with columns: image, question, answer, question_type, messages.
    """
    samples = load_medical_vqa_dataset(dataset_name, split=split, data_dir=data_dir)

    if subset_ratio is not None:
        n = max(1, int(len(samples) * subset_ratio))
        samples = samples[:n]
        logger.info(f"Subset ratio {subset_ratio}: using {n}/{len(samples)} samples")

    if max_samples is not None:
        samples = samples[:max_samples]

    records: list[dict[str, Any]] = []
    for s in samples:
        prompt_text = MEDICAL_PROMPT.format(question=s.question)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": s.answer},
                ],
            },
        ]
        records.append({
            "image": s.image,
            "question": s.question,
            "answer": s.answer,
            "question_type": s.question_type,
            "messages": messages,
        })

    ds = Dataset.from_list(records)
    logger.info(
        f"Prepared {dataset_name}/{split}: {len(ds)} samples for SFT training"
    )
    return ds


def prepare_qwen_chat_dataset(
    dataset_name: str,
    split: str,
    data_dir: str = "data",
    max_samples: int | None = None,
    subset_ratio: float | None = None,
) -> Dataset:
    """Prepare dataset with Qwen-style messages (image object in content).

    Qwen VL models expect the image directly in the message content dict,
    not as a separate column processed via apply_chat_template.

    Returns:
        HuggingFace Dataset with columns: image, messages (Qwen format).
    """
    samples = load_medical_vqa_dataset(dataset_name, split=split, data_dir=data_dir)

    if subset_ratio is not None:
        n = max(1, int(len(samples) * subset_ratio))
        samples = samples[:n]
        logger.info(f"Subset ratio {subset_ratio}: using {n}/{len(samples)} samples")

    if max_samples is not None:
        samples = samples[:max_samples]

    records: list[dict[str, Any]] = []
    for s in samples:
        prompt_text = MEDICAL_PROMPT.format(question=s.question)
        # Qwen expects {"type": "image", "image": <PIL.Image>} in content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": s.image},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": s.answer},
                ],
            },
        ]
        records.append({
            "image": s.image,
            "messages": messages,
        })

    ds = Dataset.from_list(records)
    logger.info(
        f"Prepared {dataset_name}/{split} (Qwen format): {len(ds)} samples"
    )
    return ds
