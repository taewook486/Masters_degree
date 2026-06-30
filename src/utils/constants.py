"""프로젝트 전체에서 사용하는 상수 정의."""

from __future__ import annotations

MEDICAL_PROMPT = (
    "You are a medical AI assistant. "
    "Look at this medical image and answer the following question.\n"
    "Question: {question}\n"
    "Answer concisely."
)
