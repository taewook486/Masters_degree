"""VQAv2 data loader 테스트 (REQ-010).

general_vqa.py의 VQAv2Sample, _classify_vqav2_type, _extract_best_answer,
load_vqav2_subset, download_vqav2_subset 함수를 테스트한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# datasets가 설치되어 있지 않으므로 mock으로 대체한다.
# torch/transformers는 src.data.general_vqa에서 사용하지 않으므로 모킹하지 않는다.
_mock_datasets = MagicMock()
sys.modules.setdefault("datasets", _mock_datasets)

from src.data.general_vqa import (  # noqa: E402, I001
    VQAv2Sample,
    _classify_vqav2_type,
    _extract_best_answer,
)


# ---------------------------------------------------------------------------
# _classify_vqav2_type 테스트
# ---------------------------------------------------------------------------


def test_classify_vqav2_type_yes_no_returns_closed():
    """answer_type이 'yes/no'이면 'closed'를 반환해야 한다."""
    assert _classify_vqav2_type("yes/no") == "closed"


def test_classify_vqav2_type_other_returns_open():
    """answer_type이 'yes/no'가 아니면 'open'을 반환해야 한다."""
    assert _classify_vqav2_type("number") == "open"
    assert _classify_vqav2_type("other") == "open"
    assert _classify_vqav2_type("") == "open"


# ---------------------------------------------------------------------------
# _extract_best_answer 테스트
# ---------------------------------------------------------------------------


def test_extract_best_answer_picks_most_common():
    """annotator 답변 중 가장 빈번한 답변을 반환해야 한다."""
    row = {
        "answers": [
            {"answer": "cat"},
            {"answer": "dog"},
            {"answer": "cat"},
            {"answer": "cat"},
        ],
    }
    assert _extract_best_answer(row) == "cat"


def test_extract_best_answer_falls_back_to_multiple_choice():
    """answers가 비어있으면 multiple_choice_answer로 폴백해야 한다."""
    row = {"answers": [], "multiple_choice_answer": "yes"}
    assert _extract_best_answer(row) == "yes"


def test_extract_best_answer_empty_row():
    """answers 키가 없으면 multiple_choice_answer 폴백 또는 빈 문자열."""
    row = {}
    assert _extract_best_answer(row) == ""


def test_extract_best_answer_invalid_answer_entries():
    """answers에 dict가 아닌 항목이 있어도 정상 처리해야 한다."""
    row = {
        "answers": ["not_a_dict", {"answer": "valid"}, 42],
        "multiple_choice_answer": "fallback",
    }
    assert _extract_best_answer(row) == "valid"


# ---------------------------------------------------------------------------
# VQAv2Sample dataclass 테스트
# ---------------------------------------------------------------------------


def test_vqav2_sample_fields():
    """VQAv2Sample이 올바른 필드를 가지고 있는지 확인한다."""
    mock_image = MagicMock()
    sample = VQAv2Sample(
        image=mock_image,
        question="What color is the sky?",
        answer="blue",
        question_type="open",
    )
    assert sample.image is mock_image
    assert sample.question == "What color is the sky?"
    assert sample.answer == "blue"
    assert sample.question_type == "open"


def test_vqav2_sample_question_type_values():
    """question_type이 'open' 또는 'closed' 값을 가질 수 있다."""
    mock_image = MagicMock()
    for qt in ("open", "closed"):
        sample = VQAv2Sample(
            image=mock_image, question="Q?", answer="A", question_type=qt,
        )
        assert sample.question_type in ("open", "closed")


# ---------------------------------------------------------------------------
# load_vqav2_subset 테스트
# ---------------------------------------------------------------------------


def test_load_vqav2_subset_raises_on_missing_dir(tmp_path: Path):
    """데이터 디렉토리가 없으면 FileNotFoundError를 발생시켜야 한다."""
    import pytest

    from src.data.general_vqa import load_vqav2_subset

    with pytest.raises(FileNotFoundError, match="VQAv2 subset not found"):
        load_vqav2_subset(data_dir=str(tmp_path / "nonexistent"))


def test_load_vqav2_subset_loads_samples(tmp_path: Path):
    """저장된 데이터셋을 로드하여 VQAv2Sample 리스트를 반환해야 한다."""
    subset_dir = tmp_path / "vqav2_subset"
    subset_dir.mkdir()

    mock_image = MagicMock()
    mock_image.convert.return_value = mock_image

    mock_row_open = {
        "image": mock_image,
        "question": "What is this?",
        "answers": [{"answer": "cat"}, {"answer": "cat"}],
        "answer_type": "other",
    }
    mock_row_closed = {
        "image": mock_image,
        "question": "Is this a dog?",
        "answers": [{"answer": "yes"}],
        "answer_type": "yes/no",
    }

    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(
        return_value=iter([mock_row_open, mock_row_closed]),
    )
    mock_ds.__len__ = MagicMock(return_value=2)

    with patch("src.data.general_vqa.load_from_disk", return_value=mock_ds):
        from src.data.general_vqa import load_vqav2_subset

        samples = load_vqav2_subset(data_dir=str(tmp_path))

    assert len(samples) == 2
    assert samples[0].question_type == "open"
    assert samples[0].answer == "cat"
    assert samples[1].question_type == "closed"
    assert samples[1].answer == "yes"


def test_load_vqav2_subset_max_samples(tmp_path: Path):
    """max_samples 인자가 로드되는 샘플 수를 제한해야 한다."""
    subset_dir = tmp_path / "vqav2_subset"
    subset_dir.mkdir()

    mock_image = MagicMock()
    mock_image.convert.return_value = mock_image
    rows = [
        {
            "image": mock_image,
            "question": f"Q{i}?",
            "answers": [{"answer": f"a{i}"}],
            "answer_type": "other",
        }
        for i in range(5)
    ]

    mock_ds = MagicMock()
    # select(range(2)) 호출 시 처음 2개만 반환
    selected = MagicMock()
    selected.__iter__ = MagicMock(return_value=iter(rows[:2]))
    selected.__len__ = MagicMock(return_value=2)
    mock_ds.select.return_value = selected
    mock_ds.__len__ = MagicMock(return_value=5)

    with patch("src.data.general_vqa.load_from_disk", return_value=mock_ds):
        from src.data.general_vqa import load_vqav2_subset

        samples = load_vqav2_subset(data_dir=str(tmp_path), max_samples=2)

    assert len(samples) == 2
    mock_ds.select.assert_called_once()


# ---------------------------------------------------------------------------
# download_vqav2_subset 테스트
# ---------------------------------------------------------------------------


def test_download_vqav2_subset_skips_existing(tmp_path: Path):
    """이미 존재하는 디렉토리가 있으면 다운로드를 건너뛰어야 한다."""
    subset_dir = tmp_path / "vqav2_subset"
    subset_dir.mkdir()

    from src.data.general_vqa import download_vqav2_subset

    result = download_vqav2_subset(save_dir=str(tmp_path), force=False)
    assert result == subset_dir
