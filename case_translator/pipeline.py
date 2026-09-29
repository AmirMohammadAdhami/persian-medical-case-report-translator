"""
Translation pipeline orchestrating Downloader, PDFParser, Translators, and HTMLRenderer.
"""

from pathlib import Path
from typing import Callable, Optional

from .downloader import ArticleDownloader
from .glossary import GLOSSARY_NOTES, iter_known_terms, normalise_key, notes_index
from .html_renderer import HTMLRenderer
from .models import Document, ElementType
from .pdf_parser import PDFParser
from .translators.base import Translator, is_non_translation_response


class CaseReportPipeline:
    """
    Coordinates the full end-to-end medical translation workflow:
    PDF Retrieval -> Structural Extraction -> Figure Processing -> Translation -> HTML Rendering
    """

    def __init__(
        self,
        translator: Translator,
        output_dir: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        self.translator = translator
        self.output_dir = Path(output_dir)
        self.assets_dir = self.output_dir / "assets"
        self.progress_callback = progress_callback or (lambda step, total, msg: None)

        self.downloader = ArticleDownloader()
        self.pdf_parser = PDFParser(assets_dir=self.assets_dir)
        self.html_renderer = HTMLRenderer(output_dir=self.output_dir)

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
        document = self.pdf_parser.parse(pdf_path)

        # Step 3: Extract figures (figures already extracted during parsing; report count)
        figures = document.get_figures()
        self.progress_callback(3, total_steps, f"Extracted {len(figures)} figure(s) to assets...")

        # Step 4: Translate text blocks
        self.progress_callback(4, total_steps, "Translating medical text into fluent Persian...")
        ordered_blocks = document.get_ordered_blocks()

        # Prime the glossary so a recurring term gets the same Persian rendering
        # everywhere, instead of drifting between blocks.
        try:
            self.translator.prepare_terms(iter_known_terms())
        except Exception:
            # Glossary priming is an optimisation; a failure must not stop a run.
            pass

        # Reference blocks that sit inside a bibliography run are kept in the
        # original English (journal names, DOIs and volume/issue formatting must
        # keep their academic integrity). Numbered citation markers parsed out of
        # prose are NOT part of a run and are translated normally.
        reference_run_ids = set()
        for page in document.pages:
            reference_run_ids.update(page.metadata.get("ref_run_ids", set()))

        text_blocks_to_translate = [
            b for b in ordered_blocks
            if b.type in (ElementType.TITLE, ElementType.HEADING, ElementType.PARAGRAPH, ElementType.REFERENCE)
            and b.content.strip()
        ]

        for block in text_blocks_to_translate:
            # Bibliography entries bypass translation entirely.
            if block.type == ElementType.REFERENCE and block.id in reference_run_ids:
                block.translated_content = block.content
                continue

            try:
                translated = self.translator.translate_text(block.content)
                # Guard: some chat models answer a bare heading with a
                # conversational "please provide the text" instead of a
                # translation. Reject that and fall back to the source.
                if is_non_translation_response(block.content, translated):
                    block.translated_content = block.content
                else:
                    block.translated_content = translated
            except Exception as e:
                # Fault tolerance: fallback to original text so document is not ruined
                block.translated_content = block.content

        # Step 5: Process figure captions & optional vision analysis
        self.progress_callback(5, total_steps, "Processing figure captions & visual analysis...")
        for fig in figures:
            caption = fig.metadata.get("caption", "")
            if caption:
                try:
                    translated_cap = self.translator.translate_caption(caption)
                    fig.metadata["translated_caption"] = translated_cap
                except Exception:
                    fig.metadata["translated_caption"] = caption

            # Optional vision analysis
            image_path = fig.metadata.get("image_path")
            if image_path:
                try:
                    vision_desc = self.translator.describe_image(image_path)
                    if vision_desc:
                        fig.metadata["vision_description"] = vision_desc
                except Exception:
                    pass

        # Also translate any standalone caption blocks
        for block in ordered_blocks:
            if block.type == ElementType.FIGURE_CAPTION and not block.translated_content:
                try:
                    block.translated_content = self.translator.translate_caption(block.content)
                except Exception:
                    block.translated_content = block.content

        # Step 6: Render HTML
        self.progress_callback(6, total_steps, "Building RTL Persian HTML document...")

        # Seed explanations for the interactive glossary. Authored notes win;
        # any term without one gets a short definition built from its Persian
        # rendering, so every clickable term answers with something useful.
        index = notes_index()
        notes = dict(index)
        for term in self.html_renderer.glossary_terms:
            key = normalise_key(term)
            if key in index:
                continue
            persian = self.translator._term_translations.get(key)
            if persian:
                notes[key] = f"{term} در فارسی «{persian}» ترجمه می‌شود."
        document.metadata["glossary_notes"] = notes

        output_html = self.html_renderer.render(document, filename="article.html")

        return output_html
