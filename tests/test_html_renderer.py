"""
Unit tests for HTMLRenderer.
"""

from pathlib import Path
from case_translator.html_renderer import HTMLRenderer
from case_translator.models import BoundingBox, ContentBlock, Document, ElementType, Page


def test_bidi_text_formatting():
    """A parenthetical at the start of a block has no Persian term in front."""
    renderer = HTMLRenderer(output_dir=Path("tmp_out"))
    raw_text = "(radiolucent lesion) مشاهده شد."
    formatted = renderer.format_bidi_text(raw_text)
    assert '<span class="en-term" dir="ltr">(radiolucent lesion)</span>' in formatted
    assert "glossary-term" not in formatted


def test_html_render_flow(tmp_path):
    renderer = HTMLRenderer(output_dir=tmp_path)

    bbox = BoundingBox(0, 0, 100, 50)
    p1 = ContentBlock("p1", ElementType.PARAGRAPH, "Paragraph 1", bbox, 1, translated_content="پاراگراف اول")
    fig = ContentBlock("fig1", ElementType.FIGURE, "[Figure 1]", bbox, 1, metadata={
        "figure_id": "fig1",
        "figure_num": 1,
        "relative_path": "assets/figure-1.png",
        "caption": "Figure 1",
        "translated_caption": "شکل ۱. تصویر دندان"
    })
    p2 = ContentBlock("p2", ElementType.PARAGRAPH, "Paragraph 2", bbox, 1, translated_content="پاراگراف دوم")

    doc = Document(
        metadata={"title": "گزارش بالینی"},
        pages=[Page(page_number=1, width=600, height=800, blocks=[p1, fig, p2])]
    )

    out_file = renderer.render(doc)
    assert out_file.exists()

    content = out_file.read_text(encoding="utf-8")
    assert '<html lang="fa" dir="rtl">' in content
    assert 'assets/figure-1.png' in content
    assert 'شکل ۱. تصویر دندان' in content

    # Check reading flow: p1 must appear before fig1, and fig1 before p2
    idx_p1 = content.find("پاراگراف اول")
    idx_fig = content.find("figure-1.png")
    idx_p2 = content.find("پاراگراف دوم")

    assert idx_p1 < idx_fig < idx_p2, "Figure was not positioned in the exact reading flow between paragraphs!"


class TestReferenceNumberRendering:
    """
    The reference number is rendered by the renderer, never taken from the
    translated text — so it survives translation and bidi reordering.
    """

    @staticmethod
    def _block(rid, content, ref_number=None):
        return ContentBlock(
            id=rid,
            type=ElementType.REFERENCE,
            content=content,
            bbox=BoundingBox(0, 0, 10, 10),
            page_number=1,
            metadata={"ref_number": ref_number},
        )

    def test_number_is_emitted_from_metadata(self):
        renderer = HTMLRenderer(output_dir=Path("tmp_out"))
        html = renderer._render_body([self._block("r1", "آندریاسن JO", 1)])
        assert '<span class="ref-number" dir="ltr">[1]</span>' in html

    def test_number_falls_back_to_counter(self):
        """No parsed number -> fall back to a running counter, never blank."""
        renderer = HTMLRenderer(output_dir=Path("tmp_out"))
        html = renderer._render_body(
            [self._block("r1", "یک", None), self._block("r2", "دو", None)]
        )
        assert '<span class="ref-number" dir="ltr">[1]</span>' in html
        assert '<span class="ref-number" dir="ltr">[2]</span>' in html

    def test_translated_text_never_supplies_the_number(self):
        """A number left inside the text must not be treated as the marker."""
        renderer = HTMLRenderer(output_dir=Path("tmp_out"))
        html = renderer._render_body([self._block("r1", "۲۰۰۰. [2] باراتیری", 2)])
        assert '<span class="ref-number" dir="ltr">[2]</span>' in html
        assert html.count('class="ref-number"') == 1


class TestGlossaryMarkup:
    """
    "[Persian] ([English])" pairs become clickable glossary terms.

    The glossary scan must run BEFORE en-term wrapping: every clickable pair
    contains a parenthetical, so if en-term ran first it would consume the
    pair's English half and no pair would ever match.
    """

    @staticmethod
    def _renderer(tmp_path):
        return HTMLRenderer(output_dir=tmp_path)

    def test_pair_becomes_clickable(self, tmp_path):
        r = self._renderer(tmp_path)
        out = r.format_bidi_text("پست فایبر (Fiber Post) استفاده شد")
        assert 'class="glossary-term"' in out
        assert 'data-term="Fiber Post"' in out
        assert "پست فایبر" in out

    def test_multi_word_persian_term_not_clipped(self, tmp_path):
        """Regression: the term used to lose its first character."""
        r = self._renderer(tmp_path)
        out = r.format_bidi_text("شکستگی پیچیده تاج-ریشه (Complicated Crown-Root Fracture) درمان شد")
        assert ">شکستگی پیچیده تاج-ریشه<" in out
        assert "درمان شد" in out

    def test_preceding_word_is_preserved(self, tmp_path):
        r = self._renderer(tmp_path)
        out = r.format_bidi_text("مقدار (radiolucent) بود")
        assert out.startswith("مقدار") or ">مقدار<" in out.replace('class="glossary-term"', "")
        assert "بود" in out

    def test_conjunction_is_not_part_of_the_term(self, tmp_path):
        r = self._renderer(tmp_path)
        out = r.format_bidi_text("پست فایبر (Fiber Post) و رزین کامپوزیت (Composite Resin) استفاده شد")
        assert out.count('class="glossary-term"') == 2
        assert ">و <" not in out.split('data-term="Composite Resin"')[1].split(">")[1]

    def test_terms_are_recorded_once(self, tmp_path):
        r = self._renderer(tmp_path)
        r.format_bidi_text("پست فایبر (Fiber Post) و پست فایبر (Fiber Post)")
        assert r.glossary_terms == ["Fiber Post"]

    def test_notes_are_escaped_for_inline_script(self, tmp_path):
        """'</script>' in a note must not break out of the script block."""
        r = self._renderer(tmp_path)
        bbox = BoundingBox(0, 0, 10, 10)
        doc = Document(
            pages=[Page(page_number=1, width=100, height=100, blocks=[
                ContentBlock("p1", ElementType.PARAGRAPH, "x", bbox, 1, translated_content="متن")
            ])],
            metadata={"glossary_notes": {"evil": "a </script><b>b</b>"}},
        )
        out = r.render(doc, filename="notes.html").read_text(encoding="utf-8")
        assert "</script><b>" not in out
        assert "\\u003c" in out


    def test_notes_present_for_every_clickable_term(self, tmp_path):
        """
        Every clickable term must ship a note in the rendered HTML.

        The template only embeds an explanation for terms known *before*
        render(); if the notes dict were built from a stale or empty term list,
        the panel would silently fall back to its generic message for every
        term while the page still looked fine.
        """
        r = self._renderer(tmp_path)
        bbox = BoundingBox(0, 0, 10, 10)
        doc = Document(
            pages=[Page(page_number=1, width=100, height=100, blocks=[
                ContentBlock(
                    "p1", ElementType.PARAGRAPH, "x", bbox, 1,
                    translated_content="\u067e\u0633\u062a \u0641\u0627\u06cc\u0628\u0631 (Fiber Post) \u0627\u0633\u062a\u0641\u0627\u062f\u0647 \u0634\u062f",
                )
            ])],
            metadata={"glossary_notes": {"fiber post": "\u062a\u0648\u0636\u06cc\u062d"}},
        )
        out = r.render(doc, filename="coverage.html").read_text(encoding="utf-8")
        assert 'class="glossary-term"' in out
        payload = out.split("var TERM_NOTES", 1)[1]
        assert "fiber post" in payload
