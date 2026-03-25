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
import time

logger = logging.getLogger(__name__)

# Retry settings for API resilience
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds, exponential backoff

# Search space bounds for validation
_VALID_RANKS = {4, 8, 16, 32, 64}
_VALID_BATCH_SIZES = {1, 2, 4}
_VALID_GRAD_ACCUM = {4, 8, 16}
_VALID_TARGETS = {"minimal", "medium", "full"}
_VALID_EPOCHS = {1, 2, 3, 5}


def ask_agent_for_config(
    program: str,
    history_text: str,
    trial_number: int = 0,
    total_trials: int = 40,
) -> tuple[dict, str]:
    """Ask the LLM agent to suggest the next hyperparameter config.

    Args:
        program: The agent's system prompt (from program.md).
        history_text: Human-readable summary of past trials.
        trial_number: Current trial index (0-based), used for temp scheduling.
        total_trials: Total planned trials, used for temp scheduling.

    Returns:
        Tuple of (validated hyperparameters dict, raw agent response text).

    Raises:
        RuntimeError: If API call fails after all retries.
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

    # Temperature scheduling: explore early (1.0), exploit later (0.3)
    progress = trial_number / max(total_trials - 1, 1)
    temperature = round(1.0 - 0.7 * progress, 2)

    user_message = _build_user_message(history_text, trial_number, total_trials)

    # Retry with exponential backoff
    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=program,
                messages=[{"role": "user", "content": user_message}],
                temperature=temperature,
            )
            raw_text = response.content[0].text.strip()
            config = _parse_config(raw_text)
            config = _validate_config(config)
            logger.info(
                f"Trial {trial_number}/{total_trials}, "
                f"temp={temperature}, config={config}"
            )
            return config, raw_text
        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"API attempt {attempt + 1}/{_MAX_RETRIES} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"All {_MAX_RETRIES} API attempts failed: {e}")

    raise RuntimeError(f"Agent API failed after {_MAX_RETRIES} retries: {last_error}")


def _build_user_message(
    history_text: str, trial_number: int, total_trials: int
) -> str:
    """Build structured user message with analysis guidance."""
    progress = trial_number / max(total_trials, 1)

    if progress < 0.25:
        phase_hint = (
            "EXPLORATION PHASE: Prioritize diverse configurations. "
            "Try varied rank, learning rate, and target module combinations "
            "to map the search space broadly."
        )
    elif progress < 0.75:
        phase_hint = (
            "TRANSITION PHASE: Balance exploration with exploitation. "
            "Focus on promising regions from top results, "
            "but still test unexplored parameter combinations."
        )
    else:
        phase_hint = (
            "EXPLOITATION PHASE: Fine-tune around the best configurations. "
            "Make small adjustments to top-performing hyperparameters "
            "to maximize accuracy."
        )

    return (
        f"## Experiment Status\n"
        f"- Trial: {trial_number + 1} / {total_trials}\n"
        f"- Phase: {phase_hint}\n\n"
        f"## Previous Results\n\n"
        f"{history_text}\n\n"
        f"## Analysis Instructions\n"
        f"Before suggesting, analyze:\n"
        f"1. Which hyperparameter changes correlated with accuracy gains?\n"
        f"2. Are there diminishing returns on any parameter (e.g., rank)?\n"
        f"3. What parameter combinations remain unexplored?\n\n"
        f"## Response Format\n"
        f"Respond with ONLY a valid JSON object containing these keys: "
        f"lora_rank, lora_alpha, learning_rate, batch_size, grad_accum_steps, "
        f"warmup_ratio, weight_decay, lora_targets, epochs"
    )


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
