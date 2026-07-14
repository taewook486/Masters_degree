"""Download medical VQA datasets from HuggingFace Hub."""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "pathvqa": {
        "hf_id": "flaviagiammarino/path-vqa",
        "description": "PathVQA - Pathology VQA (4,998 images, 32,799 QA pairs)",
    },
    "slake": {
        "hf_id": "mdwiratathya/SLAKE-vqa-english",
        "description": "SLAKE - English Medical VQA (642 images, ~7K QA pairs)",
    },
    "vqa_rad": {
        "hf_id": "flaviagiammarino/vqa-rad",
        "description": "VQA-RAD - Radiology VQA (315 images, 2,248 QA pairs)",
    },
}


def download_dataset(
    dataset_name: str,
    save_dir: str = "data",
    force: bool = False,
) -> Path:
    """Download a single dataset from HuggingFace and save to disk.

    Args:
        dataset_name: One of "pathvqa", "slake", "vqa_rad".
        save_dir: Base directory for saving datasets.
        force: Re-download even if already exists.

    Returns:
        Path to the saved dataset directory.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    info = DATASET_REGISTRY[dataset_name]
    save_path = Path(save_dir) / dataset_name

    if save_path.exists() and not force:
        logger.info(f"Dataset '{dataset_name}' already exists at {save_path}, skipping.")
        return save_path

    logger.info(f"Downloading {info['description']}...")
    ds = load_dataset(info["hf_id"])

    ds.save_to_disk(str(save_path))
    logger.info(f"Saved {dataset_name} to {save_path}")

    for split_name, split_ds in ds.items():
        logger.info(f"  {split_name}: {len(split_ds)} samples")

    return save_path


def download_all_datasets(
    save_dir: str = "data",
    force: bool = False,
) -> None:
    """Download all 3 medical VQA datasets."""
    for name in DATASET_REGISTRY:
        download_dataset(name, save_dir=save_dir, force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download medical VQA datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY.keys()),
        help="Download a specific dataset (default: all)",
    )
    parser.add_argument(
        "--save_dir",
        default="data",
        help="Base directory for saving datasets (default: data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already exists",
    )
    args = parser.parse_args()

    setup_logging(log_dir="logs", experiment_name="download_datasets")

    if args.dataset:
        download_dataset(args.dataset, save_dir=args.save_dir, force=args.force)
    else:
        download_all_datasets(save_dir=args.save_dir, force=args.force)


if __name__ == "__main__":
    main()
