"""
Regression tests for the defects reported against the initial implementation.

Each test names the bug it locks down, so a future refactor that reintroduces it
fails loudly instead of silently degrading the output.
"""

from pathlib import Path

import pytest

from case_translator.cache import TranslationCache
from case_translator.models import BoundingBox, ContentBlock, Document, ElementType, Page
from case_translator.pdf_parser import PDFParser
from case_translator.pipeline import CaseReportPipeline
from case_translator.translators import (
    GeminiTranslator,
    MockTranslator,
    OpenAITranslator,
    OpenCodeTranslator,
)
from case_translator.translators.base import (
    GLOSSARY_SYSTEM_PROMPT,
    NonRetryableTranslationError,
    RetryableTranslationError,
    classify_translation_error,
)
from case_translator.verification import compare_numbers


def _bbox(y0=0, y1=10, x0=0, x1=100):
    return BoundingBox(x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Bug 1: the glossary cache never existed, so prepare_terms() raised
# AttributeError and the pipeline swallowed it.
# ---------------------------------------------------------------------------


class TestTranslatorBaseInitialisation:
    def test_every_provider_calls_super_init(self):
        providers = [
            OpenCodeTranslator(api_key="sk-test"),
            OpenAITranslator(api_key="sk-test"),
            GeminiTranslator(api_key="AIza-test"),
            MockTranslator(),
        ]
        for provider in providers:
            assert isinstance(provider._term_translations, dict), type(provider).__name__
            assert provider.stats["requests"] == 0

    def test_prepare_terms_populates_the_cache(self):
        translator = MockTranslator()
        translator.prepare_terms(["Fiber Post", "Crown-Root Fracture"])
        assert translator.term_translation("Fiber Post")
        # Plural lookups must resolve to the same entry.
        assert translator.term_translation("fiber posts") == translator.term_translation("Fiber Post")

    def test_glossary_pass_uses_its_own_system_prompt(self):
        """
        The glossary payload is a numbered list; answering it with the medical
        prose prompt produced unusable output.
        """
        translator = MockTranslator()
        raw = translator._translate_with_system(GLOSSARY_SYSTEM_PROMPT, "1. Fiber Post\n2. Incisor")
        assert raw.startswith("1.")
        assert "پست فایبر" in raw

    def test_prime_glossary_failure_is_logged_not_raised(self, caplog):
        class Broken(MockTranslator):
            def _translate_with_system(self, system_prompt, text):
                raise RuntimeError("boom")

        pipeline = CaseReportPipeline(translator=Broken(), output_dir=Path("tmp_out"))
        with caplog.at_level("WARNING"):
            pipeline._prime_glossary()
        assert any("priming" in record.message.lower() for record in caplog.records)


class TestErrorClassification:
    def test_auth_error_is_not_retried(self):
        class AuthError(Exception):
            status_code = 401

        assert isinstance(
            classify_translation_error(AuthError("unauthorized")),
            NonRetryableTranslationError,
        )

    def test_connection_refused_is_not_retried(self):
        error = Exception("Connection error: [WinError 10061] connection refused")
        assert isinstance(classify_translation_error(error), NonRetryableTranslationError)

    def test_timeout_is_retried(self):
        error = Exception("Request timed out after 30 seconds")
        assert isinstance(classify_translation_error(error), RetryableTranslationError)

    def test_rate_limit_is_retried(self):
        class RateLimited(Exception):
            status_code = 429

        assert isinstance(
            classify_translation_error(RateLimited("too many requests")),
            RetryableTranslationError,
        )

    def test_openai_translator_fails_fast_on_auth(self):
        translator = OpenAITranslator(api_key="sk-test", max_retries=3)

        class AuthError(Exception):
            status_code = 401

        calls = []

        def boom(*args, **kwargs):
            calls.append(1)
            raise AuthError("incorrect api key")

        translator.client.chat.completions.create = boom
        with pytest.raises(NonRetryableTranslationError):
            translator.translate_text("Some text")
        assert len(calls) == 1, "a permanent error must not be retried"


# ---------------------------------------------------------------------------
# Bug 3: translation failures were silent, so a run "succeeded" while returning
# an English document.
# ---------------------------------------------------------------------------


class _FailingTranslator(MockTranslator):
    def translate_text(self, text: str) -> str:
        raise RuntimeError("gateway unreachable")


class _FlakyTranslator(MockTranslator):
    """Fails the first N blocks, then succeeds."""

    def __init__(self, fail_count: int):
        super().__init__()
        self.remaining = fail_count

    def translate_text(self, text: str) -> str:
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError("gateway unreachable")
        return f"ترجمه: {text}"


def _document_with_paragraphs(count: int) -> Document:
    blocks = [
        ContentBlock(
            id=f"p1_b{i}",
            type=ElementType.PARAGRAPH,
            content=f"Sentence number {i} about dental trauma.",
            bbox=_bbox(y0=i * 20, y1=i * 20 + 15),
            page_number=1,
        )
        for i in range(count)
    ]
    return Document(
        metadata={"title": "t", "total_pages": 1},
        pages=[Page(page_number=1, width=600, height=800, blocks=blocks)],
    )


class TestFailureReporting:
    def _run(self, translator, blocks=8, max_consecutive=3):
        pipeline = CaseReportPipeline(
            translator=translator,
            output_dir=Path("tmp_out"),
            workers=1,
            use_cache=False,
            max_consecutive_failures=max_consecutive,
        )
        report = pipeline._translate_document(_document_with_paragraphs(blocks))
        return pipeline, report

    def test_failures_are_counted(self):
        _, report = self._run(_FailingTranslator())
        assert report["failed"] > 0
        assert report["translated"] == 0
        assert report["failures"]

    def test_circuit_breaker_stops_early(self):
        _, report = self._run(_FailingTranslator(), blocks=20, max_consecutive=3)
        assert report["circuit_breaker_tripped"] is True
        assert report["skipped"] > 0
        # A dead endpoint must not cost one request per block.
        assert report["failed"] == 3

    def test_failed_blocks_are_marked(self):
        document = _document_with_paragraphs(4)
        pipeline = CaseReportPipeline(
            translator=_FailingTranslator(), output_dir=Path("tmp_out"), workers=1, use_cache=False
        )
        pipeline._translate_document(document)
        failed = [b for b in document.get_ordered_blocks() if b.metadata.get("translation_failed")]
        assert failed
        # The original text is preserved so nothing is lost from the document.
        assert failed[0].translated_content == failed[0].content

    def test_consecutive_counter_resets_on_success(self):
        _, report = self._run(_FlakyTranslator(fail_count=2), blocks=8, max_consecutive=3)
        assert report["circuit_breaker_tripped"] is False
        assert report["failed"] == 2
        assert report["translated"] == 6


# ---------------------------------------------------------------------------
# Bug 2: glossary fallback notes were built before render(), so they were dead
# code and every un-authored term fell back to the generic message.
# ---------------------------------------------------------------------------


class TestGlossaryNoteCoverage:
    def test_every_clickable_term_gets_a_note(self, tmp_path):
        from case_translator.html_renderer import HTMLRenderer

        renderer = HTMLRenderer(output_dir=tmp_path)
        blocks = [
            ContentBlock(
                id="p1",
                type=ElementType.PARAGRAPH,
                content="x",
                bbox=_bbox(),
                page_number=1,
                translated_content="پست فایبر (Fiber Post) و عرض بیولوژیک (Biologic Width) استفاده شد.",
            )
        ]
        document = Document(
            pages=[Page(page_number=1, width=100, height=100, blocks=blocks)],
            metadata={"glossary_notes": {}},
        )
        html = renderer.render(document, filename="notes.html").read_text(encoding="utf-8")

        assert 'data-term="Fiber Post"' in html
        assert 'data-term="Biologic Width"' in html
        # Both terms must carry an explanation in the embedded notes payload.
        payload = html.split("var TERM_NOTES", 1)[1]
        assert "fiber post" in payload
        assert "biologic width" in payload


# ---------------------------------------------------------------------------
# Cache: re-running must not pay for the same blocks twice.
# ---------------------------------------------------------------------------


class TestTranslationCache:
    def test_round_trip(self, tmp_path):
        cache = TranslationCache(tmp_path)
        key = cache.make_key("model", "prompt", "text")
        assert cache.get(key) is None
        cache.set(key, "ترجمه", model="model")
        assert cache.get(key) == "ترجمه"

    def test_key_covers_model_prompt_and_text(self):
        base = TranslationCache.make_key("m", "p", "t")
        assert TranslationCache.make_key("m2", "p", "t") != base
        assert TranslationCache.make_key("m", "p2", "t") != base
        assert TranslationCache.make_key("m", "p", "t2") != base

    def test_second_run_hits_the_cache(self, tmp_path):
        cache_dir = tmp_path / "cache"
        out = tmp_path / "out"

        first = CaseReportPipeline(
            translator=MockTranslator(), output_dir=out, cache_dir=cache_dir, workers=2
        )
        report_one = first._translate_document(_document_with_paragraphs(6))

        second = CaseReportPipeline(
            translator=MockTranslator(), output_dir=out, cache_dir=cache_dir, workers=2
        )
        report_two = second._translate_document(_document_with_paragraphs(6))

        assert report_one["cache"]["hits"] == 0
        assert report_two["cache"]["hits"] == 6


# ---------------------------------------------------------------------------
# Numbers and units must survive translation.
# ---------------------------------------------------------------------------


class TestNumericVerification:
    def test_detects_a_dropped_number(self):
        warnings = compare_numbers(
            "Follow-up at 3, 6 and 12 months.", "پیگیری در ۳ و ۶ ماه انجام شد."
        )
        assert warnings
        assert any("12" in w for w in warnings)

    def test_detects_persian_digits_as_equal(self):
        assert compare_numbers("The dose was 5 mg.", "دوز ۵ mg بود.") == []

    def test_detects_a_lost_unit(self):
        warnings = compare_numbers("A 3 mm gap was present.", "فاصله ۳ بود.")
        assert any("mm" in w for w in warnings)

    def test_clean_translation_reports_nothing(self):
        assert compare_numbers("Figure 1 shows 2 mm loss.", "شکل ۱ افت ۲ mm را نشان می‌دهد.") == []


# ---------------------------------------------------------------------------
# Bug 5: parser behaviour on real journal PDFs.
# ---------------------------------------------------------------------------


class TestLineReflow:
    """
    A "print to PDF" export puts every visual line in its own text block.
    Without reflow each line became its own translation request and the model
    saw sentence fragments.
    """

    @staticmethod
    def _line(text, y, size=25.0):
        return ContentBlock(
            id=f"b{y}",
            type=ElementType.PARAGRAPH,
            content=text,
            bbox=BoundingBox(27, y, 550, y + 30),
            page_number=1,
            metadata={"font_size": size},
        )

    def test_adjacent_lines_are_merged(self):
        parser = PDFParser(assets_dir=Path("tmp_out/assets"))
        blocks = [
            self._line("The use of photo-curable resin", 31),
            self._line("composite restorations is increasing,", 71),
            self._line("owing to improvements in strength.", 111),
        ]
        merged = parser._merge_line_fragments(blocks, 863.0)
        assert len(merged) == 1
        assert "resin composite restorations" in merged[0].content

    def test_sentence_end_starts_a_new_paragraph(self):
        parser = PDFParser(assets_dir=Path("tmp_out/assets"))
        blocks = [
            self._line("The cavity was prepared.", 31),
            self._line("Then the adhesive system was applied.", 71),
        ]
        merged = parser._merge_line_fragments(blocks, 863.0)
        assert len(merged) == 2

    def test_separate_columns_are_not_merged(self):
        parser = PDFParser(assets_dir=Path("tmp_out/assets"))
        left = self._line("Left column text without a full stop", 100)
        left.bbox = BoundingBox(45, 100, 290, 130)
        right = self._line("right column text", 100)
        right.bbox = BoundingBox(310, 100, 555, 130)
        merged = parser._merge_line_fragments([left, right], 800.0)
        assert len(merged) == 2

    def test_soft_hyphen_across_lines_is_rejoined(self):
        assert PDFParser._join_lines(["the pop\ue000", "ulation, with"]) == "the population, with"
        # A real hyphen before the break is part of the word and is kept.
        assert PDFParser._join_lines(["three-\ue000", "dimensional"]) == "three-dimensional"


class TestAuthorDetection:
    def test_byline_is_recognised(self):
        assert PDFParser._looks_like_author_line(
            "Hideki Suito1 | Yuuri Oku1 | Tomoyuki Kondo1 | Kumiko Kamada2"
        )

    def test_comma_separated_byline_is_recognised(self):
        assert PDFParser._looks_like_author_line("John A. Smith, MD, Jane Doe, PhD, Alan Reed")

    def test_title_is_not_a_byline(self):
        assert not PDFParser._looks_like_author_line(
            "Precautions When Extracting Impacted Third Molars With Impacted Enamel Pearls"
        )

    def test_sentence_is_not_a_byline(self):
        assert not PDFParser._looks_like_author_line(
            "Enamel pearls are present in 1%-9% of the population, with prevalence varying."
        )


class TestFrontmatterSuppression:
    @pytest.mark.parametrize("text", [
        "Correspondence: Hideki Suito (suito@example.edu)",
        "Received: 23 October 2025 | Accepted: 16 August 2026",
        "Keywords: cone-beam computed tomography | enamel pearl",
        "This is an open access article under the terms of the Creative Commons Attribution License.",
    ])
    def test_publisher_metadata_is_not_translated(self, text):
        assert PDFParser._looks_like_frontmatter(text)

    def test_ordinary_prose_is_not_frontmatter(self):
        assert not PDFParser._looks_like_frontmatter(
            "The patient was recalled after 3 months for a clinical evaluation."
        )

    def test_affiliation_is_not_translated(self):
        assert PDFParser._looks_like_affiliation(
            "1 Department of Oral and Maxillofacial Radiology, Graduate School of Biomedical Sciences"
        )


class TestReferenceSection:
    def test_everything_after_the_heading_becomes_a_reference(self):
        parser = PDFParser(assets_dir=Path("tmp_out/assets"))
        document = Document(pages=[Page(page_number=1, width=100, height=100, blocks=[
            ContentBlock("h", ElementType.HEADING, "References", _bbox(), 1),
            ContentBlock("b1", ElementType.PARAGRAPH,
                         "1. E. Kaminagakura, C. S. Salmon, \u201cPrevalence of Enamel Pearls,\u201d 2011.", _bbox(), 1),
            ContentBlock("b2", ElementType.PARAGRAPH,
                         "2. M. Zeichner-David, K. Oishi, \u201cRole of Hertwig's Sheath,\u201d 2003.", _bbox(), 1),
        ])])
        parser._classify_references(document)
        types = [b.type for b in document.pages[0].blocks]
        assert types == [ElementType.HEADING, ElementType.REFERENCE, ElementType.REFERENCE]
        assert document.pages[0].blocks[1].metadata["ref_number"] == 1
        assert document.pages[0].blocks[2].metadata["ref_number"] == 2

    def test_mid_block_markers_are_split(self):
        parser = PDFParser(assets_dir=Path("tmp_out/assets"))
        document = Document(pages=[Page(page_number=1, width=100, height=100, blocks=[
            ContentBlock("h", ElementType.HEADING, "References", _bbox(), 1),
            ContentBlock("b1", ElementType.PARAGRAPH,
                         "12. A. Arys, C. Philippart, \u201cRole of Cementum,\u201d 1989. "
                         "13. G. Tomov, E. Popova, \u201cEnamel Pearl,\u201d 2017.", _bbox(), 1),
        ])])
        parser._classify_references(document)
        refs = [b for b in document.pages[0].blocks if b.type == ElementType.REFERENCE]
        assert len(refs) == 2
        assert [r.metadata["ref_number"] for r in refs] == [12, 13]

    def test_doi_is_not_mistaken_for_entry_10(self):
        parser = PDFParser(assets_dir=Path("tmp_out/assets"))
        pieces = parser._split_reference_text(
            "10. S. Risnes, \u201cThe Prevalence,\u201d 1974, https://doi.org/10.1111/j.1600-0722.1974.tb00394.x."
        )
        assert pieces is not None
        assert [number for number, _ in pieces] == [10]

    def test_continuation_across_a_page_break_is_joined(self):
        parser = PDFParser(assets_dir=Path("tmp_out/assets"))
        document = Document(pages=[
            Page(page_number=1, width=100, height=100, blocks=[
                ContentBlock("h", ElementType.HEADING, "References", _bbox(), 1),
                ContentBlock("b1", ElementType.PARAGRAPH,
                             "1. Author A. \u201cA study of enamel pearls,\u201d Journal, 2011,", _bbox(), 1),
            ]),
            Page(page_number=2, width=100, height=100, blocks=[
                ContentBlock("b2", ElementType.PARAGRAPH,
                             "Bulletin Of The International Association, 11 (2017): 62-66.", _bbox(), 2),
            ]),
        ])
        parser._classify_references(document)
        refs = [b for page in document.pages for b in page.blocks if b.type == ElementType.REFERENCE]
        assert len(refs) == 1
        assert "62-66" in refs[0].content

    def test_reference_section_ends_at_the_next_heading(self):
        parser = PDFParser(assets_dir=Path("tmp_out/assets"))
        document = Document(pages=[Page(page_number=1, width=100, height=100, blocks=[
            ContentBlock("h", ElementType.HEADING, "References", _bbox(), 1),
            ContentBlock("b1", ElementType.PARAGRAPH, "1. Author A. Text, 2011.", _bbox(), 1),
            ContentBlock("h2", ElementType.HEADING, "Acknowledgements", _bbox(), 1),
            ContentBlock("b2", ElementType.PARAGRAPH, "We thank the staff of the clinic.", _bbox(), 1),
        ])])
        parser._classify_references(document)
        blocks = document.pages[0].blocks
        assert blocks[2].type == ElementType.HEADING
        assert blocks[3].type == ElementType.PARAGRAPH


class TestReferenceRunsIncludeWholeSection:
    def test_single_entry_page_is_still_kept_in_english(self):
        document = Document(pages=[Page(page_number=1, width=100, height=100, blocks=[
            ContentBlock("h", ElementType.HEADING, "References", _bbox(), 1),
            ContentBlock("r1", ElementType.REFERENCE, "1. Author A. Text, 2011.", _bbox(), 1),
        ])])
        PDFParser.mark_reference_runs(document)
        assert document.pages[0].metadata["ref_run_ids"] == set()
        PDFParser._mark_reference_section_runs(document)
        assert document.pages[0].metadata["ref_run_ids"] == {"r1"}


# ---------------------------------------------------------------------------
# Downloader hardening
# ---------------------------------------------------------------------------


class TestDownloaderHardening:
    def test_filenames_are_unique_per_url(self):
        from case_translator.downloader import ArticleDownloader

        first = ArticleDownloader._generate_filename("https://example.com/a/download")
        second = ArticleDownloader._generate_filename("https://example.com/b/download")
        assert first != second

    def test_filename_keeps_the_pdf_extension(self):
        from case_translator.downloader import ArticleDownloader

        assert ArticleDownloader._generate_filename("https://example.com/x/article").endswith(".pdf")


# ---------------------------------------------------------------------------
# Bridge security
# ---------------------------------------------------------------------------


class TestBridgeSecurity:
    def test_bridge_cli_can_parse_its_arguments(self, capsys):
        """
        Regression: the bridge's argparse defaults read OPENCODE_SERVER_PASSWORD
        and OPENCODE_BRIDGE_TOKEN from the environment, but the module never
        imported `os`, so `python -m case_translator.opencode_bridge` died with
        NameError before it could start.
        """
        from case_translator import opencode_bridge

        with pytest.raises(SystemExit) as exit_info:
            opencode_bridge.main(["--help"])
        assert exit_info.value.code == 0
        assert "--insecure-no-auth" in capsys.readouterr().out

    def test_bridge_rejects_cross_origin_and_missing_token(self, tmp_path):
        import threading
        import urllib.error
        import urllib.request

        from case_translator.opencode_bridge import build_server

        server = build_server(
            port=0,
            opencode_url="http://127.0.0.1:1",
            default_model="m",
            token="secret-token",
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]

            unauthenticated = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(unauthenticated, timeout=5)
            assert excinfo.value.code == 401

            authenticated = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/models",
                headers={"Authorization": "Bearer secret-token"},
            )
            with urllib.request.urlopen(authenticated, timeout=5) as response:
                assert response.status == 200
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_origin_header_is_refused(self):
        from case_translator.opencode_bridge import BridgeHandler

        handler = object.__new__(BridgeHandler)

        class _Headers(dict):
            def get(self, key, default=None):
                return dict.get(self, key, default)

        handler.headers = _Headers({"Origin": "https://evil.example"})
        assert handler._origin_allowed() is False
        handler.headers = _Headers({})
        assert handler._origin_allowed() is True

    def test_host_header_must_be_loopback(self):
        from case_translator.opencode_bridge import BridgeHandler

        handler = object.__new__(BridgeHandler)

        class _Headers(dict):
            def get(self, key, default=None):
                return dict.get(self, key, default)

        handler.headers = _Headers({"Host": "127.0.0.1:4097"})
        assert handler._host_allowed() is True
        handler.headers = _Headers({"Host": "attacker.example"})
        assert handler._host_allowed() is False
