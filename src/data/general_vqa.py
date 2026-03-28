"""VQAv2 validation subset loader for Catastrophic Forgetting measurement.

THESIS_PROPOSAL v0.2:
  - VQAv2 validation subset (2,000 samples, balanced sampling)
  - Used as a control group for CF measurement across all 9 Phase 2 conditions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset, load_from_disk
from PIL import Image

logger = logging.getLogger(__name__)

VQAV2_HF_ID = "lmms-lab/VQAv2"
VQAV2_SUBSET_SIZE = 2000
VQAV2_SPLIT = "validation"


@dataclass
class VQAv2Sample:
    """A single VQAv2 sample."""

    image: Image.Image
    question: str
    answer: str  # Most common answer from annotators
    question_type: str  # "open" or "closed"


def _classify_vqav2_type(answer_type: str) -> str:
    """Classify VQAv2 question as open or closed based on answer_type field."""
    return "closed" if answer_type == "yes/no" else "open"


def _extract_best_answer(row: dict) -> str:
    """Extract the most common annotator answer from a VQAv2 row."""
    answers = row.get("answers", [])
    if answers:
        answer_texts = [a["answer"] for a in answers if isinstance(a, dict) and "answer" in a]
        if answer_texts:
            from collections import Counter
            return Counter(answer_texts).most_common(1)[0][0]
    return str(row.get("multiple_choice_answer", ""))


def download_vqav2_subset(
    save_dir: str = "data",
    n_samples: int = VQAV2_SUBSET_SIZE,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """Download and save a balanced VQAv2 validation subset.

    Balanced sampling: equal proportion of open/closed questions.

    Args:
        save_dir: Base directory for saving.
        n_samples: Total number of samples (split equally between open/closed).
        seed: Random seed for reproducible sampling.
        force: Re-download even if already exists.

    Returns:
        Path to the saved subset directory.
    """
    save_path = Path(save_dir) / "vqav2_subset"

    if save_path.exists() and not force:
        logger.info(f"VQAv2 subset already exists at {save_path}, skipping.")
        return save_path

    import random
    from datasets import Dataset as HFDataset

    logger.info("Downloading VQAv2 validation (streaming mode)...")
    ds = load_dataset(VQAV2_HF_ID, split=VQAV2_SPLIT, streaming=True)

    n_per_type = n_samples // 2
    max_candidates = n_per_type * 5  # 5x buffer for randomness

    open_rows: list[dict] = []
    closed_rows: list[dict] = []
    total_seen = 0

    for row in ds:
        total_seen += 1
        answer_type = row.get("answer_type", "")
        if answer_type == "yes/no":
            if len(closed_rows) < max_candidates:
                closed_rows.append(row)
        else:
            if len(open_rows) < max_candidates:
                open_rows.append(row)

        if (len(closed_rows) >= max_candidates
                and len(open_rows) >= max_candidates):
            break

        if total_seen % 5000 == 0:
            logger.info(
                f"  streamed {total_seen} rows "
                f"(open={len(open_rows)}, closed={len(closed_rows)})"
            )

    logger.info(
        f"Collected candidates from {total_seen} rows: "
        f"open={len(open_rows)}, closed={len(closed_rows)}"
    )

    rng = random.Random(seed)
    sampled_closed = rng.sample(
        closed_rows, min(n_per_type, len(closed_rows)),
    )
    sampled_open = rng.sample(
        open_rows, min(n_per_type, len(open_rows)),
    )

    all_samples = sampled_closed + sampled_open
    subset = HFDataset.from_list(all_samples)
    subset.save_to_disk(str(save_path))
    logger.info(
        f"Saved VQAv2 subset to {save_path}: {len(subset)} samples "
        f"(closed={len(sampled_closed)}, open={len(sampled_open)})"
    )

    return save_path


def load_vqav2_subset(
    data_dir: str = "data",
    max_samples: int | None = None,
) -> list[VQAv2Sample]:
    """Load the VQAv2 validation subset for CF measurement.

    Args:
        data_dir: Base directory where datasets are saved.
        max_samples: Limit number of samples (for debugging).

    Returns:
        List of VQAv2Sample objects.
    """
    ds_path = Path(data_dir) / "vqav2_subset"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"VQAv2 subset not found at {ds_path}. "
            "Run: python -m src.data.general_vqa --download"
        )

    ds = load_from_disk(str(ds_path))
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples = []
    for row in ds:
        answer = _extract_best_answer(row)
        answer_type = row.get("answer_type", "other")

        samples.append(
            VQAv2Sample(
                image=row["image"].convert("RGB"),
                question=row["question"],
                answer=answer,
                question_type=_classify_vqav2_type(answer_type),
            )
        )

    open_count = sum(1 for s in samples if s.question_type == "open")
    closed_count = sum(1 for s in samples if s.question_type == "closed")
    logger.info(
        f"Loaded VQAv2 subset: {len(samples)} samples "
        f"(open={open_count}, closed={closed_count})"
    )

    return samples


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="VQAv2 subset management")
    parser.add_argument("--download", action="store_true", help="Download VQAv2 subset")
    parser.add_argument("--save_dir", default="data")
    parser.add_argument("--n_samples", type=int, default=VQAV2_SUBSET_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.download:
        download_vqav2_subset(
            save_dir=args.save_dir,
            n_samples=args.n_samples,
            seed=args.seed,
            force=args.force,
        )
    else:
        samples = load_vqav2_subset(data_dir=args.save_dir)
        logger.info(f"Loaded {len(samples)} VQAv2 samples successfully")


if __name__ == "__main__":
    main()
