"""Unified medical VQA dataset loading interface."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from datasets import load_from_disk
from PIL import Image

logger = logging.getLogger(__name__)

VALID_DATASETS = {"pathvqa", "slake", "vqa_rad"}


@dataclass
class VQASample:
    """A single VQA sample in standardized format."""

    image: Image.Image
    question: str
    answer: str
    question_type: str  # "open" or "closed"


def classify_question_type(
    answer: str,
    dataset_name: str,
    answer_type: str | None = None,
) -> str:
    """Classify a QA pair as 'open' or 'closed'.

    For SLAKE: uses the native answer_type field.
    For PathVQA/VQA-RAD: yes/no heuristic on the answer string.
    """
    if dataset_name == "slake" and answer_type is not None:
        return "closed" if answer_type.upper() == "CLOSED" else "open"

    normalized = answer.strip().lower()
    return "closed" if normalized in {"yes", "no"} else "open"


def _load_pathvqa(data_dir: str, split: str) -> list[VQASample]:
    """Load PathVQA dataset."""
    ds_path = Path(data_dir) / "pathvqa"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"PathVQA not found at {ds_path}. Run: python -m src.data.download --dataset pathvqa"
        )

    ds = load_from_disk(str(ds_path))
    if split not in ds:
        raise ValueError(f"PathVQA split '{split}' not found. Available: {list(ds.keys())}")

    samples = []
    for row in ds[split]:
        samples.append(
            VQASample(
                image=row["image"].convert("RGB"),
                question=row["question"],
                answer=row["answer"],
                question_type=classify_question_type(row["answer"], "pathvqa"),
            )
        )
    return samples


def _load_slake(data_dir: str, split: str) -> list[VQASample]:
    """Load SLAKE dataset (English-only, columns: image, question, answer)."""
    ds_path = Path(data_dir) / "slake"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"SLAKE not found at {ds_path}. Run: python -m src.data.download --dataset slake"
        )

    ds = load_from_disk(str(ds_path))
    if split not in ds:
        raise ValueError(f"SLAKE split '{split}' not found. Available: {list(ds.keys())}")

    samples = []
    for row in ds[split]:
        samples.append(
            VQASample(
                image=row["image"].convert("RGB"),
                question=row["question"],
                answer=row["answer"],
                question_type=classify_question_type(row["answer"], "slake"),
            )
        )
    return samples


def _load_vqa_rad(data_dir: str, split: str) -> list[VQASample]:
    """Load VQA-RAD dataset."""
    ds_path = Path(data_dir) / "vqa_rad"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"VQA-RAD not found at {ds_path}. Run: python -m src.data.download --dataset vqa_rad"
        )

    ds = load_from_disk(str(ds_path))

    if split == "validation":
        raise ValueError(
            "VQA-RAD has no validation split. Available: train, test"
        )
    if split not in ds:
        raise ValueError(f"VQA-RAD split '{split}' not found. Available: {list(ds.keys())}")

    samples = []
    for row in ds[split]:
        samples.append(
            VQASample(
                image=row["image"].convert("RGB"),
                question=row["question"],
                answer=row["answer"],
                question_type=classify_question_type(row["answer"], "vqa_rad"),
            )
        )
    return samples


_LOADERS = {
    "pathvqa": _load_pathvqa,
    "slake": _load_slake,
    "vqa_rad": _load_vqa_rad,
}


def _iter_pathvqa(data_dir: str, split: str) -> Iterator[VQASample]:
    """Stream PathVQA dataset one sample at a time (memory-safe for large train splits)."""
    ds_path = Path(data_dir) / "pathvqa"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"PathVQA not found at {ds_path}. Run: python -m src.data.download --dataset pathvqa"
        )
    ds = load_from_disk(str(ds_path))
    if split not in ds:
        raise ValueError(f"PathVQA split '{split}' not found. Available: {list(ds.keys())}")
    for row in ds[split]:
        yield VQASample(
            image=row["image"].convert("RGB"),
            question=row["question"],
            answer=row["answer"],
            question_type=classify_question_type(row["answer"], "pathvqa"),
        )


def _iter_slake(data_dir: str, split: str) -> Iterator[VQASample]:
    """Stream SLAKE dataset one sample at a time."""
    ds_path = Path(data_dir) / "slake"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"SLAKE not found at {ds_path}. Run: python -m src.data.download --dataset slake"
        )
    ds = load_from_disk(str(ds_path))
    if split not in ds:
        raise ValueError(f"SLAKE split '{split}' not found. Available: {list(ds.keys())}")
    for row in ds[split]:
        yield VQASample(
            image=row["image"].convert("RGB"),
            question=row["question"],
            answer=row["answer"],
            question_type=classify_question_type(row["answer"], "slake"),
        )


def _iter_vqa_rad(data_dir: str, split: str) -> Iterator[VQASample]:
    """Stream VQA-RAD dataset one sample at a time."""
    ds_path = Path(data_dir) / "vqa_rad"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"VQA-RAD not found at {ds_path}. Run: python -m src.data.download --dataset vqa_rad"
        )
    ds = load_from_disk(str(ds_path))
    if split == "validation":
        raise ValueError("VQA-RAD has no validation split. Available: train, test")
    if split not in ds:
        raise ValueError(f"VQA-RAD split '{split}' not found. Available: {list(ds.keys())}")
    for row in ds[split]:
        yield VQASample(
            image=row["image"].convert("RGB"),
            question=row["question"],
            answer=row["answer"],
            question_type=classify_question_type(row["answer"], "vqa_rad"),
        )


_ITER_LOADERS = {
    "pathvqa": _iter_pathvqa,
    "slake": _iter_slake,
    "vqa_rad": _iter_vqa_rad,
}


def iter_medical_vqa_dataset(
    dataset_name: str,
    split: str = "test",
    data_dir: str = "data",
) -> Iterator[VQASample]:
    """Stream a medical VQA dataset one sample at a time (no full-list materialization).

    Large train splits (e.g. PathVQA train = 19,654 images) decoded via
    load_medical_vqa_dataset() hold every PIL image in one Python list at once,
    which exceeded the container's cgroup memory limit during Phase 2 Main
    (SIGKILL). Use this streaming variant for full-dataset chat-cache building;
    load_medical_vqa_dataset() remains unchanged for existing callers (evaluation
    on smaller/capped splits).
    """
    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {VALID_DATASETS}")
    yield from _ITER_LOADERS[dataset_name](data_dir, split)


def dataset_length(dataset_name: str, split: str, data_dir: str = "data") -> int:
    """Row count for a dataset split without decoding any images (cheap Arrow metadata read)."""
    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {VALID_DATASETS}")
    ds_path = Path(data_dir) / dataset_name
    ds = load_from_disk(str(ds_path))
    if split not in ds:
        raise ValueError(f"{dataset_name} split '{split}' not found. Available: {list(ds.keys())}")
    return ds[split].num_rows


def load_medical_vqa_dataset(
    dataset_name: str,
    split: str = "test",
    data_dir: str = "data",
) -> list[VQASample]:
    """Load a medical VQA dataset in standardized format.

    Args:
        dataset_name: One of "pathvqa", "slake", "vqa_rad".
        split: Dataset split ("train", "validation", "test").
        data_dir: Base directory where datasets are saved.

    Returns:
        List of VQASample objects.
    """
    if dataset_name not in VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {VALID_DATASETS}"
        )

    loader = _LOADERS[dataset_name]
    samples = loader(data_dir, split)

    open_count = sum(1 for s in samples if s.question_type == "open")
    closed_count = sum(1 for s in samples if s.question_type == "closed")
    logger.info(
        f"Loaded {dataset_name}/{split}: {len(samples)} samples "
        f"(open={open_count}, closed={closed_count})"
    )

    return samples
