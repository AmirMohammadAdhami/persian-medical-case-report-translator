"""
Unit tests for document data models.
"""

import pytest
from case_translator.models import BoundingBox, ContentBlock, Document, ElementType, Page


def test_bounding_box_geometry():
    bbox = BoundingBox(x0=50.0, y0=100.0, x1=250.0, y1=300.0)
    assert bbox.width == 200.0
    assert bbox.height == 200.0
    assert bbox.center_x == 150.0
    assert bbox.center_y == 200.0
    assert bbox.as_tuple() == (50.0, 100.0, 250.0, 300.0)


def test_content_block_translatable():
    bbox = BoundingBox(0, 0, 100, 50)
    block_para = ContentBlock(
        id="b1",
        type=ElementType.PARAGRAPH,
        content="Clinical examination revealed a tooth fracture.",
        bbox=bbox,
        page_number=1
    )
    assert block_para.is_translatable is True

    block_fig = ContentBlock(
        id="fig1",
        type=ElementType.FIGURE,
        content="[Figure 1]",
        bbox=bbox,
        page_number=1
    )
    assert block_fig.is_translatable is False


def test_document_ordered_blocks():
    bbox = BoundingBox(0, 0, 100, 50)
    b1 = ContentBlock("b1", ElementType.TITLE, "Title", bbox, 1)
    b2 = ContentBlock("b2", ElementType.PARAGRAPH, "Paragraph 1", bbox, 1)
    b3 = ContentBlock("b3", ElementType.FIGURE, "[Figure 1]", bbox, 1)
    b4 = ContentBlock("b4", ElementType.PARAGRAPH, "Paragraph 2", bbox, 2)

    doc = Document(pages=[
        Page(page_number=1, width=600, height=800, blocks=[b1, b2, b3]),
        Page(page_number=2, width=600, height=800, blocks=[b4])
    ])

    ordered = doc.get_ordered_blocks()
    assert [b.id for b in ordered] == ["b1", "b2", "b3", "b4"]

    figures = doc.get_figures()
    assert len(figures) == 1
    assert figures[0].id == "b3"
