"""
Unit tests for FigureProcessor.
"""

from pathlib import Path
from PIL import Image
import pymupdf
import pytest

from case_translator.figure_processor import FigureProcessor
from case_translator.models import BoundingBox, ContentBlock, ElementType


def test_caption_regex_matching():
    captions = [
        ("Figure 1. Intraoral photograph showing fracture.", 1),
        ("Figure 2: Periapical radiograph.", 2),
        ("Fig. 3. Follow-up after 12 months.", 3),
        ("Fig 4: Clinical examination.", 4),
        ("شکل ۱. تصویر دندان.", 1),
    ]
    for text, expected_num in captions:
        match = FigureProcessor.CAPTION_REGEX.search(text)
        assert match is not None, f"Failed to match: {text}"
        assert int(match.group(1)) == expected_num


def test_link_captions_to_figures(tmp_path):
    processor = FigureProcessor(assets_dir=tmp_path)

    fig1 = ContentBlock(
        id="fig1",
        type=ElementType.FIGURE,
        content="[Figure 1]",
        bbox=BoundingBox(50, 100, 250, 300),
        page_number=1,
        metadata={"figure_id": "fig1", "figure_num": 1}
    )

    cap1 = ContentBlock(
        id="cap1",
        type=ElementType.PARAGRAPH,
        content="Figure 1. Intraoral photograph showing the fractured crown.",
        bbox=BoundingBox(50, 310, 250, 340),
        page_number=1,
        metadata={}
    )

    processor.link_captions_to_figures([fig1], [cap1])

    assert cap1.type == ElementType.FIGURE_CAPTION
    assert cap1.metadata.get("figure_id") == "fig1"
    assert fig1.metadata.get("caption_id") == "cap1"
    assert "Figure 1" in fig1.metadata.get("caption", "")
