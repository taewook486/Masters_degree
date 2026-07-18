"""Convert medical VQA datasets to chat-format for SFT training.

Each sample is converted to a conversation turn:
  User: <image> + medical prompt + question
  Assistant: answer

Supports both Qwen-style (with vision info) and standard chat template formats.
"""

from __future__ import annotations

import itertools
import logging
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, load_from_disk

from src.data.dataset import VQASample, dataset_length, iter_medical_vqa_dataset
from src.utils.constants import MEDICAL_PROMPT

logger = logging.getLogger(__name__)


def _atomic_save_to_disk(ds: Dataset, final_dir: Path) -> None:
    """save_to_disk를 임시 경로에 쓴 뒤 성공 시에만 원자적으로 최종 경로로 rename한다.

    디스크 풀(ENOSPC) 등으로 save_to_disk가 중간에 끊기면, final_dir에 "폴더는
    있지만 내용은 불완전한" 캐시가 남는다. 이후 실행은 cache_dir.exists()만 보고
    load_from_disk를 시도해 FileNotFoundError로 영구 반복 실패한다(Phase 2 Main
    2026-07-18 실제 재현: 원본 컨테이너 디스크가 꽉 찬 순간 빌드 중이던
    qwen_pathvqa_train 캐시가 이렇게 깨진 채 남았다). 임시 경로 rename은 같은
    파일시스템 내에서 원자적이므로, 중간에 죽어도 final_dir은 "아예 없음" 상태를
    유지해 다음 실행이 정상적으로 재빌드를 시도한다.
    """
    tmp_dir = final_dir.parent / f".tmp-{final_dir.name}-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    ds.save_to_disk(str(tmp_dir))
    os.replace(tmp_dir, final_dir)


def _bounded_samples(
    dataset_name: str,
    split: str,
    data_dir: str,
    max_samples: int | None,
    subset_ratio: float | None,
) -> Iterator[VQASample]:
    """subset_ratio/max_samples를 반영한 샘플 스트림.

    dataset_length()는 이미지를 디코딩하지 않는 가벼운 Arrow 메타데이터 조회라,
    subset_ratio 계산에 전체 리스트 materialize가 필요 없다.
    """
    n = None
    if subset_ratio is not None:
        total = dataset_length(dataset_name, split, data_dir=data_dir)
        n = max(1, int(total * subset_ratio))
        logger.info(f"Subset ratio {subset_ratio}: using {n}/{total} samples")
    if max_samples is not None:
        n = max_samples if n is None else min(n, max_samples)

    it = iter_medical_vqa_dataset(dataset_name, split=split, data_dir=data_dir)
    return itertools.islice(it, n) if n is not None else it


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
    # 캐시는 이미지를 재저장하므로 용량이 큰다. /workspace 볼륨(quota)이 아니라
    # 여유 있는 컨테이너 디스크에 두도록 MOAI_CHAT_CACHE_DIR로 재지정 가능
    # (run_phase2_main.sh가 /hf_cache/chat_cache로 설정). 미설정 시 data_dir 하위.
    base = os.environ.get("MOAI_CHAT_CACHE_DIR") or str(Path(data_dir) / "_chat_cache")
    return Path(base) / "_".join(parts)


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

    # load_medical_vqa_dataset()는 전체 split을 파이썬 리스트로 한번에 materialize한다.
    # PathVQA train(19,654 이미지)처럼 큰 split은 이게 컨테이너 cgroup 메모리 상한을
    # 넘겨 SIGKILL로 죽는다(Phase 2 Main 2026-07-17 실제 재현). Dataset.from_generator로
    # 배치 단위 스트리밍 빌드해 이미지 전체를 메모리에 동시에 들고 있지 않게 한다.
    def _records() -> Iterator[dict]:
        for s in _bounded_samples(dataset_name, split, data_dir, max_samples, subset_ratio):
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
            yield {
                # trl 0.24 native VLM collator(DataCollatorForVisionLanguageModeling)는
                # "images"(복수 리스트) 컬럼을 읽는다. messages content는 이미 리스트라
                # prepare_multimodal_messages가 no-op → 이미지 토큰 이중 삽입 없음.
                # 단수 "image"는 collator가 안 읽으므로 제거(대용량 train셋 이미지 중복 저장/RAM 방지).
                "images": [s.image],
                "question": s.question,
                "answer": s.answer,
                "question_type": s.question_type,
                "messages": messages,
            }

    ds = Dataset.from_generator(_records)
    logger.info(
        f"Prepared {dataset_name}/{split}: {len(ds)} samples for SFT training"
    )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_to_disk(ds, cache_dir)
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

    # prepare_chat_dataset과 동일한 이유로 스트리밍 빌드(전체 리스트 materialize 회피).
    def _records() -> Iterator[dict]:
        for s in _bounded_samples(dataset_name, split, data_dir, max_samples, subset_ratio):
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
            yield {
                "image": s.image,
                "messages": messages,
            }

    ds = Dataset.from_generator(_records)
    logger.info(
        f"Prepared {dataset_name}/{split} (Qwen format): {len(ds)} samples"
    )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_to_disk(ds, cache_dir)
    logger.info(f"[chat-cache] saved {dataset_name}/{split} (qwen) → {cache_dir}")
    return ds
