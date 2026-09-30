"""
Translation pipeline orchestrating Downloader, PDFParser, Translators, and HTMLRenderer.
"""

import logging
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .cache import DEFAULT_CACHE_DIR, TranslationCache
from .downloader import ArticleDownloader
from .glossary import notes_index
from .html_renderer import HTMLRenderer
from .models import ContentBlock, Document, ElementType
from .pdf_parser import PDFParser
from .translators.base import (
    NonRetryableTranslationError,
    Translator,
    is_non_translation_response,
)
from .verification import compare_numbers

logger = logging.getLogger(__name__)


class CaseReportPipeline:
    """
    Coordinates the full end-to-end medical translation workflow:
    PDF Retrieval -> Structural Extraction -> Figure Processing -> Translation -> HTML Rendering
    """

    #: How many consecutive block failures end the translation phase early.
    #: A gateway that is down should cost a few seconds, not a full run: with
    #: retries and backoff, 27 failed blocks previously took minutes and then
    #: still produced a "successful" English document.
    DEFAULT_MAX_CONSECUTIVE_FAILURES = 5

    def __init__(
        self,
        translator: Translator,
        output_dir: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        workers: int = 4,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        bilingual: bool = False,
        embed_images: bool = False,
        clear_assets: bool = True,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    ):
        self.translator = translator
        self.output_dir = Path(output_dir)
        self.assets_dir = self.output_dir / "assets"
        self.progress_callback = progress_callback or (lambda step, total, msg: None)

        self.workers = max(1, int(workers))
        self.bilingual = bilingual
        self.embed_images = embed_images
        self.clear_assets = clear_assets
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))

        self.cache = TranslationCache(
            cache_dir=cache_dir or (self.output_dir.parent / DEFAULT_CACHE_DIR),
            enabled=use_cache,
        )

        self.downloader = ArticleDownloader()
        self.pdf_parser = PDFParser(assets_dir=self.assets_dir)
        self.html_renderer = HTMLRenderer(output_dir=self.output_dir)
        #: Filled in by run(); consumed by the CLI to print the final summary.
        self.last_report: Dict = {}

    # -- public entry point ------------------------------------------------

    def run(self, source: str) -> Path:
        """
        Executes the full pipeline for a given PDF source (local file or URL).
        Returns the path to the generated HTML document.
        """
        total_steps = 6

        # Step 1: Download or locate PDF
        self.progress_callback(1, total_steps, "Downloading / verifying PDF...")
        pdf_path = self.downloader.get_pdf(source)

        # Step 2: Extract document structure
        self.progress_callback(2, total_steps, "Extracting document structure & reading order...")
        if self.clear_assets:
            removed = self.pdf_parser.reset_assets()
            if removed:
                logger.info("Cleared %d stale asset file(s) from %s", removed, self.assets_dir)
        document = self.pdf_parser.parse(pdf_path)

        # Step 3: Extract figures (figures already extracted during parsing; report count)
        figures = document.get_figures()
        self.progress_callback(3, total_steps, f"Extracted {len(figures)} figure(s) to assets...")

        # Step 4: Translate text blocks
        self.progress_callback(4, total_steps, "Translating medical text into fluent Persian...")
        report = self._translate_document(document)

        # Step 5: Process figure captions & optional vision analysis
        self.progress_callback(5, total_steps, "Processing figure captions & visual analysis...")
        self._process_figures(figures, document.get_ordered_blocks(), report)

        # Step 6: Render HTML
        self.progress_callback(6, total_steps, "Building RTL Persian HTML document...")
        document.metadata["glossary_notes"] = notes_index()
        document.metadata["translation_report"] = report
        document.metadata["bilingual"] = self.bilingual
        document.metadata["embed_images"] = self.embed_images
        document.metadata["term_translations"] = dict(self.translator._term_translations)

        output_html = self.html_renderer.render(document, filename="article.html")

        self.last_report = report
        self._report_summary(report, output_html)
        return output_html

    # -- glossary priming --------------------------------------------------

    def _prime_glossary(self) -> None:
        """
        Warms the term cache so a recurring term is rendered identically
        everywhere. A failure here is survivable but must be visible: the old
        bare `except Exception: pass` hid a hard AttributeError that made the
        whole feature a no-op.
        """
        from .glossary import iter_known_terms

        try:
            self.translator.prepare_terms(iter_known_terms())
        except NonRetryableTranslationError as exc:
            logger.warning(
                "Glossary priming skipped: the provider rejected the request (%s).", exc
            )
        except Exception as exc:  # noqa: BLE001 - priming must never abort a run
            logger.warning("Glossary priming failed (%s: %s).", type(exc).__name__, exc)

    # -- block translation -------------------------------------------------

    def _translate_document(self, document: Document) -> Dict:
        """Translates every translatable block, in parallel, with a circuit breaker."""
        ordered_blocks = document.get_ordered_blocks()

        self._prime_glossary()

        # Reference blocks that sit inside a bibliography run are kept in the
        # original English (journal names, DOIs and volume/issue formatting must
        # keep their academic integrity). Numbered citation markers parsed out of
        # prose are NOT part of a run and are translated normally.
        reference_run_ids = set()
        for page in document.pages:
            reference_run_ids.update(page.metadata.get("ref_run_ids", set()))

        jobs: List[Tuple[ContentBlock, str]] = []
        table_blocks: List[ContentBlock] = []
        for block in ordered_blocks:
            if block.type == ElementType.FIGURE_CAPTION:
                continue  # handled in the figure pass
            if block.type == ElementType.TABLE:
                # Tables are translated cell by cell so the grid survives; see
                # _translate_tables().
                table_blocks.append(block)
                continue
            if block.type not in (
                ElementType.TITLE,
                ElementType.HEADING,
                ElementType.PARAGRAPH,
                ElementType.REFERENCE,
            ):
                continue
            if not block.content.strip():
                continue

            if block.type == ElementType.REFERENCE and block.id in reference_run_ids:
                # Bibliography entries bypass translation entirely.
                block.translated_content = block.content
                block.metadata["kept_original"] = True
                continue

            kind = "reference" if block.type == ElementType.REFERENCE else "text"
            jobs.append((block, kind))

        report = {
            "total": len(jobs),
            "translated": 0,
            "failed": 0,
            "skipped": 0,
            "kept_original": sum(
                1 for b in ordered_blocks if b.metadata.get("kept_original")
            ),
            "failures": [],
            "numeric_warnings": [],
            "circuit_breaker_tripped": False,
            "cache": self.cache.stats(),
            "stats": dict(self.translator.stats),
        }

        self._run_jobs(jobs, report)
        self._translate_tables(table_blocks, report)

        report["cache"] = self.cache.stats()
        report["stats"] = dict(self.translator.stats)
        return report

    def _run_jobs(self, jobs: List[Tuple[ContentBlock, str]], report: Dict) -> None:
        """
        Runs translation jobs with bounded parallelism and a circuit breaker.

        Concurrency: the previous implementation was strictly sequential, so a
        27-block article cost 27 round-trips of wall-clock latency. Blocks are
        independent, so a small pool is safe; the pool size is capped low
        because most gateways rate-limit aggressively.

        Circuit breaker: after N consecutive failures the remaining blocks are
        left untranslated and flagged, instead of hammering a dead endpoint and
        then silently emitting an English document that looks like a success.
        """
        if not jobs:
            return

        consecutive_failures = 0
        tripped = False
        pending = deque()

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            job_iter = iter(jobs)

            def _submit_next() -> bool:
                try:
                    block, kind = next(job_iter)
                except StopIteration:
                    return False
                pending.append((block, kind, pool.submit(self._translate_one, block, kind)))
                return True

            for _ in range(self.workers):
                if not _submit_next():
                    break

            while pending:
                block, kind, future = pending.popleft()

                if tripped:
                    future.cancel()
                    self._mark_untranslated(block, "circuit breaker open")
                    report["skipped"] += 1
                    continue

                try:
                    translation, warnings = future.result()
                except Exception as exc:  # noqa: BLE001 - one block must not kill the run
                    consecutive_failures += 1
                    self._mark_untranslated(block, f"{type(exc).__name__}: {exc}")
                    report["failed"] += 1
                    report["failures"].append(
                        {"id": block.id, "page": block.page_number, "error": str(exc)[:300]}
                    )
                    logger.warning(
                        "Block %s (page %s) could not be translated: %s",
                        block.id, block.page_number, exc,
                    )
                    if consecutive_failures >= self.max_consecutive_failures:
                        tripped = True
                        report["circuit_breaker_tripped"] = True
                        logger.error(
                            "Stopping translation after %d consecutive failures; "
                            "the remaining blocks are left in the original language.",
                            consecutive_failures,
                        )
                    else:
                        # Keep the pipeline fed while the endpoint still looks
                        # alive. Once the breaker trips we stop submitting, so a
                        # dead provider does not queue hundreds of doomed tasks.
                        _submit_next()
                    continue

                consecutive_failures = 0
                self._assign_result(block, kind, translation)
                report["translated"] += 1
                if warnings:
                    block.metadata["numeric_warnings"] = warnings
                    report["numeric_warnings"].append(
                        {"id": block.id, "page": block.page_number, "warnings": warnings}
                    )

                _submit_next()

            # Blocks that were never even submitted (because the breaker opened
            # first) still have to be reported: an untranslated block must be
            # visible in the document, not silently left empty.
            if tripped:
                for block, _kind in job_iter:
                    self._mark_untranslated(block, "circuit breaker open")
                    report["skipped"] += 1

    @staticmethod
    def _source_text_for(block: ContentBlock, kind: str) -> str:
        """The text a job actually translates (figures carry their caption in metadata)."""
        if kind == "caption" and block.type == ElementType.FIGURE:
            return block.metadata.get("caption", "") or ""
        return block.content

    def _assign_result(self, block: ContentBlock, kind: str, translation: str) -> None:
        if kind == "caption" and block.type == ElementType.FIGURE:
            block.metadata["translated_caption"] = translation
        else:
            block.translated_content = translation

    def _translate_one(self, block: ContentBlock, kind: str) -> Tuple[str, List[str]]:
        """Translates one block (cache-aware) and returns (text, numeric warnings)."""
        source = self._source_text_for(block, kind)

        model = getattr(self.translator, "model_name", "") or "default"
        cache_key = TranslationCache.make_key(model, kind, source)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.translator.stats["cache_hits"] += 1
            translation = cached
        else:
            if kind == "caption":
                translation = self.translator.translate_caption(source)
            elif kind == "reference":
                translation = self.translator.translate_reference(source)
            else:
                translation = self.translator.translate_text(source)
            self.cache.set(cache_key, translation, model=model, source_text=source[:400])

        # Guard: some chat models answer a bare heading with a conversational
        # "please provide the text" instead of a translation.
        if is_non_translation_response(source, translation):
            raise RuntimeError("model replied conversationally instead of translating")

        if not translation or not translation.strip():
            raise RuntimeError("model returned an empty translation")

        # Numeric fidelity: doses, sizes and dates must survive translation.
        # Headings are skipped — a bare label rarely carries a number worth
        # comparing, and one-character replies would produce noise.
        warnings: List[str] = []
        if kind == "text" and block.type in (ElementType.PARAGRAPH, ElementType.TITLE):
            warnings = compare_numbers(source, translation)

        return translation, warnings

    @staticmethod
    def _mark_untranslated(block: ContentBlock, reason: str) -> None:
        if block.type == ElementType.FIGURE:
            caption = block.metadata.get("caption", "")
            if caption:
                block.metadata["translated_caption"] = caption
        else:
            block.translated_content = block.content
        block.metadata["translation_failed"] = True
        block.metadata["translation_failure_reason"] = reason[:300]

    # -- tables ------------------------------------------------------------

    def _translate_tables(self, table_blocks: List[ContentBlock], report: Dict) -> None:
        """
        Translates table cells individually and rebuilds the grid.

        Sending a whole table as one blob makes the model collapse rows and
        columns; translating the pipe-joined text produced unreadable output.
        Cell-level translation keeps the structure intact, and pure-numeric
        cells are passed through untouched.
        """
        if not table_blocks:
            return

        tasks: List[Tuple[ContentBlock, int, int, str]] = []
        for block in table_blocks:
            rows = block.metadata.get("table_rows") or []
            for row_index, row in enumerate(rows):
                for column_index, cell in enumerate(row):
                    if cell and re.search(r'[A-Za-z\u0600-\u06FF]{2,}', cell):
                        tasks.append((block, row_index, column_index, cell))

        if not tasks:
            for block in table_blocks:
                block.metadata["translated_rows"] = block.metadata.get("table_rows")
            return

        report["total"] += len(tasks)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [
                (task, pool.submit(self._translate_cell, task[3])) for task in tasks
            ]

            for (block, row_index, column_index, source), future in futures:
                try:
                    translation = future.result()
                except Exception as exc:  # noqa: BLE001
                    report["failed"] += 1
                    report["failures"].append(
                        {"id": f"{block.id}:r{row_index}c{column_index}",
                         "page": block.page_number, "error": str(exc)[:300]}
                    )
                    translation = source
                else:
                    report["translated"] += 1

                rows = block.metadata.setdefault(
                    "translated_rows", [list(row) for row in block.metadata.get("table_rows") or []]
                )
                try:
                    rows[row_index][column_index] = translation
                except IndexError:
                    logger.debug("Table cell %s:%s,%s out of range", block.id, row_index, column_index)

        for block in table_blocks:
            if "translated_rows" not in block.metadata:
                block.metadata["translated_rows"] = block.metadata.get("table_rows")

    def _translate_cell(self, text: str) -> str:
        model = getattr(self.translator, "model_name", "") or "default"
        cache_key = TranslationCache.make_key(model, "cell", text)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.translator.stats["cache_hits"] += 1
            return cached

        translation = self.translator.translate_text(text)
        if not translation or not translation.strip():
            raise RuntimeError("model returned an empty translation")
        if is_non_translation_response(text, translation):
            raise RuntimeError("model replied conversationally instead of translating")
        self.cache.set(cache_key, translation, model=model, source_text=text[:200])
        return translation

    # -- figures -----------------------------------------------------------

    def _process_figures(self, figures, ordered_blocks, report: Dict) -> None:
        caption_jobs: List[Tuple[ContentBlock, str]] = []

        for fig in figures:
            caption = fig.metadata.get("caption", "")
            if caption and not fig.metadata.get("translated_caption"):
                caption_jobs.append((fig, "caption"))

        # Standalone caption blocks that were not attached to a figure.
        for block in ordered_blocks:
            if block.type == ElementType.FIGURE_CAPTION and not block.translated_content:
                caption_jobs.append((block, "caption"))

        if caption_jobs:
            caption_report: Dict = {
                "total": len(caption_jobs),
                "translated": 0,
                "failed": 0,
                "skipped": 0,
                "failures": [],
                "numeric_warnings": [],
                "circuit_breaker_tripped": False,
            }
            self._run_jobs(caption_jobs, caption_report)
            report["total"] += caption_report.get("total", 0)
            report["failed"] += caption_report.get("failed", 0)
            report["skipped"] += caption_report.get("skipped", 0)
            report["failures"].extend(caption_report.get("failures", []))
            if caption_report.get("circuit_breaker_tripped"):
                report["circuit_breaker_tripped"] = True
            report["translated"] += caption_report.get("translated", 0)

        # Optional vision analysis.
        for fig in figures:
            image_path = fig.metadata.get("image_path")
            if not image_path:
                continue
            try:
                vision_desc = self.translator.describe_image(image_path)
                if vision_desc:
                    fig.metadata["vision_description"] = vision_desc
            except Exception as exc:  # noqa: BLE001
                logger.warning("Vision analysis failed for %s: %s", image_path, exc)

    # -- reporting ---------------------------------------------------------

    def _report_summary(self, report: Dict, output_html: Path) -> None:
        total = report.get("total", 0)
        failed = report.get("failed", 0)
        if total:
            logger.info(
                "Translation summary: %d/%d blocks translated, %d failed, %d skipped.",
                report.get("translated", 0), total, failed, report.get("skipped", 0),
            )
        if failed or report.get("skipped"):
            logger.warning(
                "%d of %d text block(s) were NOT translated and remain in English. "
                "See the warning banner in %s.",
                failed + report.get("skipped", 0), total, output_html,
            )
        if report.get("numeric_warnings"):
            logger.warning(
                "%d block(s) show numeric drift between source and translation; "
                "they are flagged in the output for manual review.",
                len(report["numeric_warnings"]),
            )
