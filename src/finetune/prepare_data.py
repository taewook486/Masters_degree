"""Convert medical VQA datasets to chat-format for SFT training.

Each sample is converted to a conversation turn:
  User: <image> + medical prompt + question
  Assistant: answer

Supports both Qwen-style (with vision info) and standard chat template formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from src.data.dataset import load_medical_vqa_dataset
from src.utils.constants import MEDICAL_PROMPT

logger = logging.getLogger(__name__)


def _chat_cache_dir(
    data_dir: str,
    fmt: str,
    dataset_name: str,
    split: str,
    max_samples: int | None,
    subset_ratio: float | None,
) -> Path:
    """준비된(chat 포맷) 데이터셋의 디스크 캐시 경로.

    조건마다 이미지 19k개를 디코딩→재인코딩(Dataset.from_list)하면 조건당 ~30분이라
    36조건 Main이 비현실적이 된다. (dataset, split, format, samples, ratio)로 키를 만들어
    한 번만 빌드하고 이후 load_from_disk(mmap)로 즉시 로드 + 학습 중 배치별 lazy 디코딩.
    """
    parts = [fmt, dataset_name, split]
    if subset_ratio is not None:
        parts.append(f"sub{subset_ratio}")
    if max_samples is not None:
        parts.append(f"max{max_samples}")
    return Path(data_dir) / "_chat_cache" / "_".join(parts)


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
    cache_dir = _chat_cache_dir(data_dir, "std", dataset_name, split, max_samples, subset_ratio)
    if cache_dir.exists():
        logger.info(f"[chat-cache] load {dataset_name}/{split} (std) from {cache_dir}")
        return load_from_disk(str(cache_dir))

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
            # trl 0.24 native VLM collator(DataCollatorForVisionLanguageModeling)는
            # "images"(복수 리스트) 컬럼을 읽는다. messages content는 이미 리스트라
            # prepare_multimodal_messages가 no-op → 이미지 토큰 이중 삽입 없음.
            # 단수 "image"는 collator가 안 읽으므로 제거(대용량 train셋 이미지 중복 저장/RAM 방지).
            "images": [s.image],
            "question": s.question,
            "answer": s.answer,
            "question_type": s.question_type,
            "messages": messages,
        })

    ds = Dataset.from_list(records)
    logger.info(
        f"Prepared {dataset_name}/{split}: {len(ds)} samples for SFT training"
    )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_dir))
    logger.info(f"[chat-cache] saved {dataset_name}/{split} (std) → {cache_dir}")
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
    cache_dir = _chat_cache_dir(data_dir, "qwen", dataset_name, split, max_samples, subset_ratio)
    if cache_dir.exists():
        logger.info(f"[chat-cache] load {dataset_name}/{split} (qwen) from {cache_dir}")
        return load_from_disk(str(cache_dir))

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
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(cache_dir))
    logger.info(f"[chat-cache] saved {dataset_name}/{split} (qwen) → {cache_dir}")
    return ds
