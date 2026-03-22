"""LLM agent for autoresearch-style hyperparameter suggestion.

Calls the Anthropic API (Claude) to propose next HPO config
based on experiment history. Parses JSON response into a config dict.

Requires ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# Search space bounds for validation
_VALID_RANKS = {4, 8, 16, 32, 64}
_VALID_BATCH_SIZES = {1, 2, 4}
_VALID_GRAD_ACCUM = {4, 8, 16}
_VALID_TARGETS = {"minimal", "medium", "full"}
_VALID_EPOCHS = {1, 2, 3, 5}


def ask_agent_for_config(program: str, history_text: str) -> dict:
    """Ask the LLM agent to suggest the next hyperparameter config.

    Args:
        program: The agent's system prompt (from program.md).
        history_text: Human-readable summary of past trials.

    Returns:
        Dict of validated hyperparameters.

    Raises:
        RuntimeError: If API call fails or response cannot be parsed.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. "
            "Set it in your environment or .env file."
        )

    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic package not installed. "
            "Run: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)

    user_message = (
        f"Here are the results from previous experiments:\n\n"
        f"{history_text}\n\n"
        f"Based on this history, suggest the next hyperparameter configuration. "
        f"Respond with ONLY a valid JSON object containing these keys: "
        f"lora_rank, lora_alpha, learning_rate, batch_size, grad_accum_steps, "
        f"warmup_ratio, weight_decay, lora_targets, epochs"
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=program,
        messages=[{"role": "user", "content": user_message}],
        temperature=0.7,  # Some creativity for exploration
    )

    raw_text = response.content[0].text.strip()
    config = _parse_config(raw_text)
    config = _validate_config(config)
    return config


def _parse_config(raw_text: str) -> dict:
    """Extract JSON from the agent's response (may have markdown fences)."""
    # Try direct JSON parse
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... }
    match = re.search(r"\{[^{}]+\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise RuntimeError(f"Could not parse JSON from agent response: {raw_text[:200]}")


def _validate_config(config: dict) -> dict:
    """Validate and clamp hyperparameters to search space bounds."""
    rank = config.get("lora_rank", 16)
    if rank not in _VALID_RANKS:
        rank = min(_VALID_RANKS, key=lambda x: abs(x - rank))
        logger.warning(f"Clamped lora_rank to {rank}")

    alpha = config.get("lora_alpha", rank * 2)
    # Ensure alpha is rank * {1, 2, 4}
    valid_alphas = [rank * r for r in [1, 2, 4]]
    if alpha not in valid_alphas:
        alpha = min(valid_alphas, key=lambda x: abs(x - alpha))
        logger.warning(f"Clamped lora_alpha to {alpha}")

    lr = float(config.get("learning_rate", 2e-4))
    lr = max(1e-5, min(5e-4, lr))

    bs = config.get("batch_size", 1)
    if bs not in _VALID_BATCH_SIZES:
        bs = min(_VALID_BATCH_SIZES, key=lambda x: abs(x - bs))

    ga = config.get("grad_accum_steps", 8)
    if ga not in _VALID_GRAD_ACCUM:
        ga = min(_VALID_GRAD_ACCUM, key=lambda x: abs(x - ga))

    wu = float(config.get("warmup_ratio", 0.03))
    wu = max(0.0, min(0.1, wu))

    wd = float(config.get("weight_decay", 0.01))
    wd = max(0.0, min(0.1, wd))

    targets = config.get("lora_targets", "minimal")
    if targets not in _VALID_TARGETS:
        targets = "minimal"
        logger.warning("Invalid lora_targets, defaulting to 'minimal'")

    epochs = config.get("epochs", 3)
    if epochs not in _VALID_EPOCHS:
        epochs = min(_VALID_EPOCHS, key=lambda x: abs(x - epochs))

    return {
        "lora_rank": rank,
        "lora_alpha": alpha,
        "learning_rate": round(lr, 6),
        "batch_size": bs,
        "grad_accum_steps": ga,
        "warmup_ratio": round(wu, 4),
        "weight_decay": round(wd, 4),
        "lora_targets": targets,
        "epochs": epochs,
    }
