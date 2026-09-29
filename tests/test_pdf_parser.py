"""
Unit tests for the PDFParser module.
"""

from pathlib import Path
import pytest
from case_translator.models import BoundingBox, ContentBlock, Document, ElementType, Page
from case_translator.pdf_parser import PDFParser


def test_join_lines_with_dehyphenation(tmp_path):
    parser = PDFParser(assets_dir=tmp_path)
    lines = [
        "Trauma to the oral and maxillofacial re-",
        "gion occurs frequently, and treat-",
        "ment is essential."
    ]
    result = parser._join_lines(lines)
    assert "maxillofacial region" in result
    assert "treatment is essential" in result


def test_reorder_blocks_for_reading_two_columns(tmp_path):
    parser = PDFParser(assets_dir=tmp_path)
    page_width = 600.0
    page_height = 800.0

    # Header full width
    header = ContentBlock("head", ElementType.TITLE, "Title", BoundingBox(50, 50, 550, 100), 1)

    # Column 1 blocks (x ~ 50 to 280)
    c1_b1 = ContentBlock("c1_b1", ElementType.PARAGRAPH, "Col 1 Para 1", BoundingBox(50, 150, 280, 250), 1)
    c1_b2 = ContentBlock("c1_b2", ElementType.PARAGRAPH, "Col 1 Para 2", BoundingBox(50, 260, 280, 360), 1)

    # Column 2 blocks (x ~ 320 to 550)
    c2_b1 = ContentBlock("c2_b1", ElementType.PARAGRAPH, "Col 2 Para 1", BoundingBox(320, 150, 550, 250), 1)
    c2_b2 = ContentBlock("c2_b2", ElementType.PARAGRAPH, "Col 2 Para 2", BoundingBox(320, 260, 550, 360), 1)

    # Naive vertical order would mix c1_b1, c2_b1, c1_b2, c2_b2
    shuffled = [c2_b1, c1_b2, header, c2_b2, c1_b1]

    ordered = parser._reorder_blocks_for_reading(shuffled, page_width, page_height)
    ordered_ids = [b.id for b in ordered]

    # Expected human reading order:
    # 1. Full-width header
    # 2. Column 1: c1_b1, then c1_b2
    # 3. Column 2: c2_b1, then c2_b2
    assert ordered_ids == ["head", "c1_b1", "c1_b2", "c2_b1", "c2_b2"]


def test_parse_sample_pdf(tmp_path):
    pdf_path = Path("sample_case_report.pdf")
    if not pdf_path.exists():
        pytest.skip("sample_case_report.pdf not found")

    parser = PDFParser(assets_dir=tmp_path / "assets")
    doc = parser.parse(pdf_path)

    assert len(doc.pages) == 2
    assert "Complicated Crown-Root Fracture" in doc.metadata.get("title", "")

    ordered = doc.get_ordered_blocks()
    types = [b.type for b in ordered]

    assert ElementType.TITLE in types
    assert ElementType.HEADING in types
    assert ElementType.PARAGRAPH in types
    assert ElementType.FIGURE in types
    assert ElementType.FIGURE_CAPTION in types

    # Check that figures are in the assets
    figures = doc.get_figures()
    assert len(figures) == 2
    for fig in figures:
        img_path = Path(fig.metadata["image_path"])
        assert img_path.exists()


class TestRunningHeaderSuppression:
    """
    Journal banners / running heads must not leak into the translated body.

    Regression: "Case Reports in Dentistry / Hindawi Publishing Corporation"
    used to be rendered as the first <p> of the article, because its bbox sat
    just below the old hardcoded 35-point header threshold.
    """

    def test_journal_banner_is_not_a_paragraph(self):
        parser = PDFParser(assets_dir=Path("output/assets"))
        doc = parser.parse(Path("sample_case_report.pdf"))

        bodies = [
            b.content.lower()
            for page in doc.pages
            for b in page.blocks
            if b.type == ElementType.PARAGRAPH
        ]
        assert not any("hindawi" in text for text in bodies), (
            "publisher banner leaked into the article body"
        )
        assert not any("case reports in dentistry" in text for text in bodies)

    def test_first_body_block_is_real_content(self):
        parser = PDFParser(assets_dir=Path("output/assets"))
        doc = parser.parse(Path("sample_case_report.pdf"))

        first = doc.pages[0].blocks[0]
        assert first.type in (ElementType.TITLE, ElementType.HEADING, ElementType.PARAGRAPH)
        assert "hindawi" not in first.content.lower()

    def test_header_band_uses_page_ratio_not_fixed_points(self):
        # A block 42pt from the top of a 842pt page is inside the 9% band.
        parser = PDFParser(assets_dir=Path("output/assets"))
        assert parser.HEADER_FOOTER_BAND_RATIO > 0
        band = 842.0 * parser.HEADER_FOOTER_BAND_RATIO
        assert 42.4 <= band, "42pt banner must fall inside the header band"

    def test_repeated_page_edge_text_is_suppressed(self):
        """Text repeated in the page-edge band across pages becomes OTHER."""
        parser = PDFParser(assets_dir=Path("output/assets"))
        doc = parser.parse(Path("sample_case_report.pdf"))

        repeated_banner = [
            b for page in doc.pages for b in page.blocks
            if "hindawi" in b.content.lower()
        ]
        for block in repeated_banner:
            assert block.type == ElementType.OTHER


class TestReferenceNumberExtraction:
    """
    Reference numbering must be parsed out before translation.

    Regression: numbering was left inside the translated text, so the model
    dropped it ("[1]") or shuffled it mid-sentence ("2000. [2] Baratieri...").
    Worse, the split regex used an unbounded \\d+, so a publication year like
    "2000." was treated as a numbered entry and torn away from the previous
    reference.
    """

    def test_bracket_marker(self):
        assert PDFParser._extract_reference_number("[3] Simonsen RJ. Text") == (
            3,
            "Simonsen RJ. Text",
        )

    def test_dot_marker(self):
        assert PDFParser._extract_reference_number("2. Baratieri LN. Text") == (
            2,
            "Baratieri LN. Text",
        )

    def test_year_is_not_a_marker(self):
        """'2000.' is a publication year, not a reference number."""
        text = "2000. Baratieri LN. Text"
        assert PDFParser._extract_reference_number(text) == (None, text)

    def test_year_not_split_as_new_reference(self):
        ref_re = PDFParser.REFERENCE_ITEM_REGEX
        assert ref_re.search("2000. Baratieri LN. Text") is None
        assert ref_re.search("[2] Baratieri LN. Text") is not None

    def test_no_marker_is_left_untouched(self):
        text = "Andreasen JO. Essentials of Traumatic Dental Injuries."
        assert PDFParser._extract_reference_number(text) == (None, text)


class TestReferenceRuns:
    """
    Reference list items must be identified so the pipeline can keep them in
    English (journal names, DOIs, volume/issue formatting).

    A run must start at the first reference on a page: numbered citation markers
    like "[2]" are also parsed as references but can appear mid-paragraph, so a
    lone marker must not open a bibliography interval.
    """

    @staticmethod
    def _page(block_types):
        bbox = BoundingBox(0, 0, 10, 10)
        blocks = [
            ContentBlock(f"b{i}", t, f"text {i}", bbox, 1)
            for i, t in enumerate(block_types)
        ]
        return Page(page_number=1, width=100, height=100, blocks=blocks)

    def test_bibliography_run_is_marked(self):
        doc = Document(pages=[self._page([
            ElementType.PARAGRAPH,
            ElementType.REFERENCE,
            ElementType.REFERENCE,
        ])])
        PDFParser.mark_reference_runs(doc)
        run_ids = doc.pages[0].metadata["ref_run_ids"]
        assert run_ids == {"b1", "b2"}

    def test_mid_paragraph_marker_is_not_a_run(self):
        """Regression: a lone [2] inside prose must not open a run."""
        doc = Document(pages=[self._page([
            ElementType.PARAGRAPH,
            ElementType.REFERENCE,
            ElementType.PARAGRAPH,
        ])])
        PDFParser.mark_reference_runs(doc)
        assert doc.pages[0].metadata["ref_run_ids"] == set()

    def test_run_ends_at_a_non_reference_block(self):
        doc = Document(pages=[self._page([
            ElementType.REFERENCE,
            ElementType.REFERENCE,
            ElementType.HEADING,
            ElementType.REFERENCE,
            ElementType.REFERENCE,
        ])])
        PDFParser.mark_reference_runs(doc)
        assert doc.pages[0].metadata["ref_run_ids"] == {"b0", "b1", "b3", "b4"}


class TestReferenceRuns:
    """
    Reference list items must be identified so the pipeline can keep them in
    English (journal names, DOIs, volume/issue formatting).

    A run must start at the first reference on a page: numbered citation markers
    like "[2]" are also parsed as references but can appear mid-paragraph, so a
    lone marker must not open a bibliography interval.
    """

    @staticmethod
    def _page(block_types):
        bbox = BoundingBox(0, 0, 10, 10)
        blocks = [
            ContentBlock(f"b{i}", t, f"text {i}", bbox, 1)
            for i, t in enumerate(block_types)
        ]
        return Page(page_number=1, width=100, height=100, blocks=blocks)

    def test_bibliography_run_is_marked(self):
        doc = Document(pages=[self._page([
            ElementType.PARAGRAPH,
            ElementType.REFERENCE,
            ElementType.REFERENCE,
        ])])
        PDFParser.mark_reference_runs(doc)
        run_ids = doc.pages[0].metadata["ref_run_ids"]
        assert run_ids == {"b1", "b2"}

    def test_mid_paragraph_marker_is_not_a_run(self):
        """Regression: a lone [2] inside prose must not open a run."""
        doc = Document(pages=[self._page([
            ElementType.PARAGRAPH,
            ElementType.REFERENCE,
            ElementType.PARAGRAPH,
        ])])
        PDFParser.mark_reference_runs(doc)
        assert doc.pages[0].metadata["ref_run_ids"] == set()

    def test_run_ends_at_a_non_reference_block(self):
        doc = Document(pages=[self._page([
            ElementType.REFERENCE,
            ElementType.REFERENCE,
            ElementType.HEADING,
            ElementType.REFERENCE,
            ElementType.REFERENCE,
        ])])
        PDFParser.mark_reference_runs(doc)
        assert doc.pages[0].metadata["ref_run_ids"] == {"b0", "b1", "b3", "b4"}
