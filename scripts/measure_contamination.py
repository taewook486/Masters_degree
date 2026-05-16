"""Min-K% Probability를 이용한 데이터 오염(Data Contamination) 측정.

동료 심사 IV번 지적("Data Contamination 완전 무시")에 대응한 능동적 통제 절차.

이론적 배경:
    Sample이 사전훈련 데이터에 포함되었다면, 모델은 해당 sample의 token에
    대해 비훈련 sample 대비 평균적으로 높은 log-probability를 부여한다.
    Min-K%는 정답 텍스트의 하위 K% token의 평균 log-prob를 contamination
    indicator로 사용한다.

참조: Shi et al., "Detecting Pretraining Data from Large Language Models",
    NAACL 2024.

사용법:
    # 단일 모델 + 데이터셋
    python scripts/measure_contamination.py \\
        --config configs/models/qwen3_vl_2b.yaml \\
        --dataset pathvqa --k_percent 20

    # 결과 통합 분석 (Phase 1 결과 산출 후)
    python scripts/measure_contamination.py --aggregate
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.baseline.model_loader import load_config, load_model, unload_model
from src.data.dataset import load_medical_vqa_dataset
from src.utils.logging_config import setup_logging
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


@dataclass
class MinKResult:
    """단일 sample의 Min-K% 측정 결과."""
    sample_id: int
    question: str
    answer: str
    minK_score: float
    num_tokens: int


def compute_min_k_score(
    model: Any,
    processor: Any,
    image: Any,
    question: str,
    answer: str,
    config: Any,
    k_percent: int = 20,
) -> tuple[float, int]:
    """단일 sample에 대해 정답 텍스트의 Min-K% log-probability를 계산한다.

    Args:
        model: 로드된 VLM 모델.
        processor: 모델 processor.
        image: PIL Image.
        question: 질문 텍스트.
        answer: 정답 텍스트 (Min-K% 측정 대상).
        config: 모델 config (prompt_format 등).
        k_percent: 하위 K% (기본 20).

    Returns:
        (minK_score, num_tokens) 튜플. score는 NaN/Inf 방지된 평균 log-prob.
    """
    # 프롬프트 구성 (Phase 1과 동일 형식)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {question}\nAnswer concisely."},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + answer

    # 텍스트 + 이미지 인코딩
    inputs = processor(
        text=full_text,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    # 정답 부분만 따로 인코딩하여 토큰 개수 파악
    answer_tokens = processor.tokenizer(
        answer, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    n_answer_tokens = len(answer_tokens)
    if n_answer_tokens == 0:
        return 0.0, 0

    # 모델 forward (no_grad)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # shape: (1, seq_len, vocab_size)

    # 정답 토큰의 log-prob 추출
    # logits[:, i-1]은 position i 토큰의 분포를 예측
    input_ids = inputs["input_ids"][0]
    seq_len = input_ids.shape[0]

    # 정답 토큰은 마지막 n_answer_tokens 위치
    answer_start = seq_len - n_answer_tokens
    if answer_start < 1:
        return 0.0, 0

    log_probs: list[float] = []
    log_softmax = torch.nn.functional.log_softmax
    for pos in range(answer_start, seq_len):
        # 다음 토큰 분포 예측은 logits[pos - 1]
        token_logits = logits[0, pos - 1]
        token_log_probs = log_softmax(token_logits, dim=-1)
        target_token = input_ids[pos].item()
        log_probs.append(float(token_log_probs[target_token]))

    if not log_probs:
        return 0.0, 0

    # 하위 K%만 추출하여 평균
    log_probs.sort()
    k_count = max(1, int(len(log_probs) * k_percent / 100))
    bottom_k = log_probs[:k_count]
    minK_score = sum(bottom_k) / len(bottom_k)

    return float(minK_score), n_answer_tokens


def measure_single_condition(
    config_path: str,
    dataset_name: str,
    output_dir: str,
    k_percent: int = 20,
    max_samples: int | None = None,
    seed: int = 42,
    data_dir: str = "data",
) -> dict:
    """단일 (모델 × 데이터셋) 조합에 대한 Min-K% 측정.

    Args:
        config_path: 모델 config YAML.
        dataset_name: pathvqa, slake, vqa_rad 중 하나.
        output_dir: 결과 저장 디렉토리.
        k_percent: Min-K% 파라미터.
        max_samples: 디버그용 샘플 수 제한.
        seed: 재현성 시드.
        data_dir: 데이터셋 디렉토리.

    Returns:
        요약 통계 딕셔너리.
    """
    set_seed(seed)
    config = load_config(config_path)
    model_name = config.model_name

    logger.info(f"[Contamination] {model_name} × {dataset_name} (K={k_percent}%)")

    samples = load_medical_vqa_dataset(dataset_name, split="test", data_dir=data_dir)
    if max_samples is not None:
        samples = samples[:max_samples]

    model, processor = load_model(config)
    results: list[MinKResult] = []
    try:
        for idx, sample in enumerate(tqdm(samples, desc=f"{model_name}/{dataset_name}")):
            try:
                score, n_tokens = compute_min_k_score(
                    model=model,
                    processor=processor,
                    image=sample.image,
                    question=sample.question,
                    answer=sample.answer,
                    config=config,
                    k_percent=k_percent,
                )
                results.append(MinKResult(
                    sample_id=idx,
                    question=sample.question,
                    answer=sample.answer,
                    minK_score=round(score, 4),
                    num_tokens=n_tokens,
                ))
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM at sample {idx}, skipping")
                torch.cuda.empty_cache()
            except (RuntimeError, ValueError) as e:
                logger.warning(f"Error at sample {idx}: {e}")
    finally:
        unload_model(model, processor)

    # 요약 통계
    valid_scores = [r.minK_score for r in results if r.num_tokens > 0]
    summary = {
        "model_name": model_name,
        "dataset": dataset_name,
        "k_percent": k_percent,
        "num_samples": len(results),
        "valid_samples": len(valid_scores),
        "mean_minK": round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0,
        "median_minK": round(sorted(valid_scores)[len(valid_scores) // 2], 4) if valid_scores else 0.0,
        "min_minK": round(min(valid_scores), 4) if valid_scores else 0.0,
        "max_minK": round(max(valid_scores), 4) if valid_scores else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"{model_name}_{dataset_name}_minK{k_percent}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "per_sample": [
                {
                    "sample_id": r.sample_id,
                    "question": r.question,
                    "answer": r.answer,
                    "minK_score": r.minK_score,
                    "num_tokens": r.num_tokens,
                }
                for r in results
            ],
        }, f, indent=2, ensure_ascii=False)

    logger.info(
        f"  -> mean={summary['mean_minK']:.4f}, "
        f"median={summary['median_minK']:.4f}, "
        f"range=[{summary['min_minK']:.4f}, {summary['max_minK']:.4f}]"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Min-K% Probability 데이터 오염 측정")
    parser.add_argument("--config", required=True, help="모델 config YAML 경로")
    parser.add_argument("--dataset", required=True, choices=["pathvqa", "slake", "vqa_rad"])
    parser.add_argument("--output_dir", default="results/contamination")
    parser.add_argument("--k_percent", type=int, default=20, help="Min-K% 파라미터")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    setup_logging(log_dir=args.output_dir, experiment_name="contamination")

    measure_single_condition(
        config_path=args.config,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        k_percent=args.k_percent,
        max_samples=args.max_samples,
        seed=args.seed,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
