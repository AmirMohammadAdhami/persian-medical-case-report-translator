"""
PDF parser for medical case reports.
Extracts document structure, preserves multi-column reading order,
and positions figures and captions at their exact logical flow locations.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from .figure_processor import FigureProcessor
from .models import BoundingBox, ContentBlock, Document, ElementType, Page


class PDFParser:
    """
    Parses medical case report PDFs into a structured Document model.
    Preserves multi-column reading order and exact relative figure positions.
    """

    HEADING_PATTERNS = [
        re.compile(r'^\s*(?:\d+[\.\s]+)?(?:Abstract|Introduction|Case Report|Case Presentation|Case Description|Observation|Discussion|Conclusion|Conclusions|References|Bibliography|Conflict of Interest|Conflicts of Interest|Acknowledgements|Acknowledgments|Author Contributions)\s*[:.]?\s*$', re.IGNORECASE),
        re.compile(r'^\s*(?:\d+[\.\s]+)?(?:Abstract|Introduction|Case Report|Case Presentation|Case Description|Observation|Discussion|Conclusion|Conclusions|References|Conflict of Interest|Conflicts of Interest|Acknowledgements|Author Contributions)\b', re.IGNORECASE),
    ]

    REFERENCE_ITEM_REGEX = re.compile(r'^\s*(?:\[\d+\]|\d{1,2}\.)\s+[A-Z]')

    # A reference's own number marker, e.g. "[3]" or "2.".
    # The digit count is deliberately capped: an unbounded \d+ matches a
    # publication year ("2000."), which then gets split off as if it were a
    # numbered entry — tearing the year away from the previous reference.
    REFERENCE_MARKER_REGEX = re.compile(r'^\s*(?:\[(\d{1,3})\]|(\d{1,2})[\.\)])\s+')

    # Fraction of page height treated as the running header/footer band.
    HEADER_FOOTER_BAND_RATIO = 0.09

    # Signals that a short block near a page edge is furniture rather than prose.
    RUNNING_HEADER_PATTERNS = [
        re.compile(r'^\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*$', re.IGNORECASE),
        re.compile(r'^\s*page\s+\d+', re.IGNORECASE),
        re.compile(r'case reports? in\b', re.IGNORECASE),
        re.compile(r'\bhindawi\b', re.IGNORECASE),
        re.compile(r'\bwiley\b', re.IGNORECASE),
        re.compile(r'\belsevier\b', re.IGNORECASE),
        re.compile(r'\bspringer\b', re.IGNORECASE),
        re.compile(r'\bdoi\s*[:.]', re.IGNORECASE),
        re.compile(r'^\s*(?:volume|vol\.|issue|no\.)\s*\d', re.IGNORECASE),
        re.compile(r'\b(?:journal|international journal|journal of)\b', re.IGNORECASE),
        re.compile(r'\bpublishing (?:corporation|corp|group|ltd)\b', re.IGNORECASE),
        re.compile(r'\b(?:received|accepted|published)\s+(?:\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', re.IGNORECASE),
        re.compile(r'^\s*\d{4}\s+\w+\s*(?:\.|$)'),  # bare citation like "2011 Hindawi."
    ]

    @classmethod
    def _looks_like_running_header(cls, text: str) -> bool:
        """True when a short block matches known header/footer/journal furniture."""
        return any(pattern.search(text) for pattern in cls.RUNNING_HEADER_PATTERNS)

    def __init__(self, assets_dir: Path):
        self.assets_dir = Path(assets_dir)
        self.figure_processor = FigureProcessor(assets_dir=self.assets_dir)

    def parse(self, pdf_path: Path) -> Document:
        """
        Parses the PDF at pdf_path and returns a Document object.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        doc = pymupdf.open(str(pdf_path))
        if len(doc) == 0:
            raise ValueError(f"PDF file '{pdf_path}' has 0 pages.")

        # Pass 1: Compute document-wide typography statistics (e.g. body font size)
        body_font_size = self._detect_body_font_size(doc)

        document = Document(
            metadata={
                "source_file": str(pdf_path),
                "title": "",
                "total_pages": len(doc),
            }
        )

        in_references_section = False

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            # Extract raw text blocks and spans
            raw_text_blocks = self._extract_text_blocks(page, page_num, body_font_size)

            # Extract figures from this page
            figure_blocks = self.figure_processor.extract_figures_from_page(doc, page, page_num)

            # Link captions to figures
            self.figure_processor.link_captions_to_figures(figure_blocks, raw_text_blocks)

            # Combine text and figure blocks for column-aware ordering
            all_page_blocks = raw_text_blocks + figure_blocks

            # Reconstruct reading order for multi-column layout
            ordered_blocks = self._reorder_blocks_for_reading(all_page_blocks, page_width, page_height)

            # Classify blocks (Title, Author, Headings, References, Paragraphs)
            classified_blocks = []
            for block in ordered_blocks:
                if block.type == ElementType.FIGURE:
                    classified_blocks.append(block)
                    continue

                if block.type == ElementType.FIGURE_CAPTION:
                    classified_blocks.append(block)
                    continue

                # Check for reference items
                if in_references_section:
                    # Check if block contains numbered citations like [1] ... [2] ...
                    ref_matches = list(re.finditer(r'(?:^|\n|\s)(?:\[(\d+)\]|(\d{1,2})\.)\s+', block.content))
                    if len(ref_matches) > 1:
                        # Split into separate reference blocks
                        split_indices = [m.start() for m in ref_matches]
                        split_indices.append(len(block.content))
                        for i in range(len(ref_matches)):
                            start_i = split_indices[i]
                            end_i = split_indices[i + 1]
                            ref_subtext = block.content[start_i:end_i].strip()
                            if ref_subtext:
                                ref_number, ref_body = self._extract_reference_number(ref_subtext)
                                sub_block = ContentBlock(
                                    id=f"{block.id}_ref{i+1}",
                                    type=ElementType.REFERENCE,
                                    content=ref_body or ref_subtext,
                                    bbox=block.bbox,
                                    page_number=page_num,
                                    metadata={
                                        "is_numbered_ref": True,
                                        "ref_number": ref_number,
                                    }
                                )
                                classified_blocks.append(sub_block)
                        continue
                    elif self.REFERENCE_ITEM_REGEX.search(block.content) or block.metadata.get("is_numbered_ref"):
                        block.type = ElementType.REFERENCE
                        ref_number, ref_body = self._extract_reference_number(block.content)
                        if ref_number is not None and ref_body:
                            block.content = ref_body
                            block.metadata["ref_number"] = ref_number
                    elif block.type == ElementType.HEADING:
                        # Another heading after references?
                        pass

                # Check if block triggers references section
                if block.type == ElementType.HEADING and "reference" in block.content.lower():
                    in_references_section = True

                # Determine if page 1 has Title
                if page_num == 1 and not document.metadata.get("title"):
                    if block.type == ElementType.TITLE:
                        document.metadata["title"] = block.content

                classified_blocks.append(block)

            # Post-process: clean and de-hyphenate text
            cleaned_blocks = self._post_process_blocks(classified_blocks)

            document.pages.append(
                Page(
                    page_number=page_num,
                    width=page_width,
                    height=page_height,
                    blocks=cleaned_blocks,
                )
            )

        # Cross-page pass: running headers/footers repeat verbatim on multiple
        # pages. This catches furniture the keyword heuristic misses, e.g. when
        # the journal banner carries no recognisable publisher name.
        self._drop_repeated_running_headers(document)

        # Reference lists are bibliography, not prose: they must stay in English
        # so journal names, DOIs and volume/issue formatting keep their academic
        # integrity. Done after the page loop so it sees the whole document.
        self.mark_reference_runs(document)

        doc.close()
        return document

    @staticmethod
    def mark_reference_runs(document: Document) -> None:
        """
        Flags each page-interval of consecutive reference items.

        The renderer uses this to keep reference text in the original English
        instead of translating it. Without it, quoted bibliographies come back
        half-translated: author names partly Persianised and journal titles
        turned into literal translations, which is worse than leaving them be.

        A run must start at the FIRST reference block on a page. Numbered
        citation markers like "[2]" are still parsed as references and can
        appear mid-paragraph, so a lone marker must not be treated as the start
        of a bibliography interval.
        """
        for page in document.pages:
            page_ref_run_ids = set()
            run_ids: List[str] = []
            for block in page.blocks:
                if block.type == ElementType.REFERENCE:
                    run_ids.append(block.id)
                else:
                    # A single reference between prose blocks is a citation
                    # marker, not a bibliography entry.
                    if len(run_ids) >= 2:
                        page_ref_run_ids.update(run_ids)
                    run_ids = []
            if len(run_ids) >= 2:
                page_ref_run_ids.update(run_ids)
            page.metadata["ref_run_ids"] = page_ref_run_ids

    def _drop_repeated_running_headers(self, document: Document) -> None:
        """
        Marks repeated page-edge text as OTHER.

        A running header (journal name, article DOI, page number) appears
        identically on many pages. Real prose never does, so repetition in the
        header/footer band is a strong, publisher-agnostic signal.
        """
        if len(document.pages) < 2:
            return

        counts: Dict[str, int] = {}
        for page in document.pages:
            band = page.height * self.HEADER_FOOTER_BAND_RATIO
            for block in page.blocks:
                if block.type == ElementType.OTHER:
                    continue
                in_band = block.bbox.y1 <= band or block.bbox.y0 >= (page.height - band)
                if not in_band:
                    continue
                text = block.content.strip()
                if not text or len(text) > 120:
                    continue
                key = re.sub(r'\s+', ' ', text).lower()
                counts[key] = counts.get(key, 0) + 1

        repeated = {key for key, count in counts.items() if count >= 2}

        for page in document.pages:
            for block in page.blocks:
                if block.type == ElementType.OTHER:
                    continue
                key = re.sub(r'\s+', ' ', block.content.strip()).lower()
                if key in repeated:
                    block.type = ElementType.OTHER

    def _detect_body_font_size(self, doc: pymupdf.Document) -> float:
        """Calculates the predominant (body) font size across the document."""
        size_counts: Dict[float, int] = {}
        for page_idx in range(min(5, len(doc))):
            page = doc[page_idx]
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # text
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            size = round(span.get("size", 10.0), 1)
                            text_len = len(span.get("text", "").strip())
                            if text_len > 3:
                                size_counts[size] = size_counts.get(size, 0) + text_len

        if not size_counts:
            return 10.0
        # Most frequent font size is the body text size
        body_size = max(size_counts.items(), key=lambda item: item[1])[0]
        return body_size

    def _extract_text_blocks(
        self,
        page: pymupdf.Page,
        page_num: int,
        body_font_size: float
    ) -> List[ContentBlock]:
        """Extracts text blocks with font sizes, weights, and bboxes."""
        page_dict = page.get_text("dict")
        blocks: List[ContentBlock] = []
        block_counter = 0

        for raw_b in page_dict.get("blocks", []):
            if raw_b.get("type") != 0:  # Not text
                continue

            bbox_tuple = raw_b.get("bbox", (0, 0, 0, 0))
            bbox = BoundingBox(bbox_tuple[0], bbox_tuple[1], bbox_tuple[2], bbox_tuple[3])

            # Gather lines and text spans
            line_strings = []
            max_font_size = 0.0
            is_bold = False
            total_chars = 0

            for line in raw_b.get("lines", []):
                line_text_parts = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if span_text:
                        line_text_parts.append(span_text)
                        size = span.get("size", 0.0)
                        if size > max_font_size:
                            max_font_size = size
                        flags = span.get("flags", 0)
                        font_name = span.get("font", "").lower()
                        if (flags & 2 != 0) or ("bold" in font_name) or ("black" in font_name):
                            is_bold = True
                        total_chars += len(span_text.strip())

                line_str = "".join(line_text_parts).strip()
                if line_str:
                    line_strings.append(line_str)

            if not line_strings:
                continue

            # Merge lines with de-hyphenation
            full_text = self._join_lines(line_strings)
            if not full_text.strip():
                continue

            block_counter += 1
            block_id = f"p{page_num}_b{block_counter}"

            # Initial classification
            element_type = self._classify_text_block(
                text=full_text,
                max_font_size=max_font_size,
                body_font_size=body_font_size,
                is_bold=is_bold,
                page_num=page_num,
                bbox=bbox,
                page_height=page.rect.height
            )

            block = ContentBlock(
                id=block_id,
                type=element_type,
                content=full_text,
                bbox=bbox,
                page_number=page_num,
                metadata={
                    "font_size": max_font_size,
                    "is_bold": is_bold,
                    "char_count": total_chars,
                }
            )
            blocks.append(block)

        return blocks

    def _join_lines(self, lines: List[str]) -> str:
        """Joins line fragments with proper space and de-hyphenation."""
        if not lines:
            return ""

        result = lines[0]
        for line in lines[1:]:
            # If previous line ends with a hyphen attached to a word character
            if result.endswith("-") and len(result) > 1 and result[-2].isalpha() and line and line[0].isalpha():
                result = result[:-1] + line
            else:
                result = result + " " + line
        return result

    def _classify_text_block(
        self,
        text: str,
        max_font_size: float,
        body_font_size: float,
        is_bold: bool,
        page_num: int,
        bbox: BoundingBox,
        page_height: float
    ) -> ElementType:
        """Classifies a text block into TITLE, HEADING, PARAGRAPH, etc."""
        text_clean = text.strip()

        # Check for header/footer artifacts (running heads, journal banners,
        # page numbers). The band is a fraction of page height rather than a
        # fixed point value, because a hardcoded threshold silently missed
        # banners that sit slightly lower on the page.
        top_band = page_height * self.HEADER_FOOTER_BAND_RATIO
        bottom_band = page_height - top_band
        if bbox.y1 <= top_band or bbox.y0 >= bottom_band:
            if len(text_clean) < 100 and self._looks_like_running_header(text_clean):
                return ElementType.OTHER

        # Check for Title on first page
        if page_num == 1 and max_font_size >= (body_font_size * 1.35) and len(text_clean) < 300 and bbox.y0 < 300:
            return ElementType.TITLE

        # Check for Figure Caption
        if FigureProcessor.CAPTION_REGEX.search(text_clean):
            return ElementType.FIGURE_CAPTION

        # Check for Section Headings
        for pattern in self.HEADING_PATTERNS:
            if pattern.search(text_clean) and len(text_clean) < 100:
                return ElementType.HEADING

        # Short bold text with larger font is likely a heading
        if (is_bold or max_font_size >= (body_font_size * 1.15)) and len(text_clean) < 120 and not text_clean.endswith("."):
            return ElementType.HEADING

        # Reference list item
        if self.REFERENCE_ITEM_REGEX.search(text_clean):
            return ElementType.REFERENCE

        return ElementType.PARAGRAPH

    @classmethod
    def _extract_reference_number(cls, text: str):
        """
        Splits a reference's own number marker off the front of its text.

        Returns (number_or_None, text_without_marker).

        The number is removed here, before translation, so the model can neither
        drop it nor shuffle it into the middle of the sentence (both of which
        happen when the marker is left in the translated payload). The renderer
        re-adds it deterministically from ref_number / its own counter.
        """
        match = cls.REFERENCE_MARKER_REGEX.match(text)
        if not match:
            return None, text
        number = int(match.group(1) or match.group(2))
        return number, text[match.end():]

    def _reorder_blocks_for_reading(
        self,
        blocks: List[ContentBlock],
        page_width: float,
        page_height: float
    ) -> List[ContentBlock]:
        """
        Orders blocks according to human reading flow.
        Correctly handles multi-column layouts by detecting column bands,
        sorting column 1 before column 2, while preserving full-width elements.
        """
        if not blocks:
            return []

        # Filter out negligible OTHER blocks (like running headers/footers)
        significant_blocks = [b for b in blocks if b.type != ElementType.OTHER]
        if not significant_blocks:
            return blocks

        # Threshold to consider an element "full-width" (spanning both columns)
        # Typically in a 2-column page, each column is ~45% width. Full-width is > 65%.
        full_width_threshold = page_width * 0.60
        mid_page_x = page_width / 2.0

        # Partition the page vertically into horizontal bands whenever a full-width block appears
        # Find all full-width blocks
        full_width_blocks = []
        for b in significant_blocks:
            if b.bbox.width >= full_width_threshold:
                full_width_blocks.append(b)

        # Sort full-width blocks by top position (y0)
        full_width_blocks.sort(key=lambda b: b.bbox.y0)

        # Build vertical slices/bands between full-width barriers
        bands: List[Tuple[float, float, Optional[ContentBlock]]] = []
        last_y = 0.0

        for fwb in full_width_blocks:
            # Region before this full-width block
            if fwb.bbox.y0 > last_y + 5:
                bands.append((last_y, fwb.bbox.y0, None))
            # Region containing this full-width block
            bands.append((fwb.bbox.y0, fwb.bbox.y1, fwb))
            last_y = fwb.bbox.y1

        # Region after the last full-width block
        if last_y < page_height:
            bands.append((last_y, page_height, None))

        ordered_result: List[ContentBlock] = []

        for band_top, band_bottom, barrier_block in bands:
            if barrier_block:
                ordered_result.append(barrier_block)
                continue

            # Get blocks falling in this vertical range
            band_blocks = [
                b for b in significant_blocks
                if b not in full_width_blocks
                and (b.bbox.center_y >= band_top - 5 and b.bbox.center_y <= band_bottom + 5)
            ]

            if not band_blocks:
                continue

            # Determine if this band has multiple columns
            # Check if there are blocks on both the left and right sides
            left_col = [b for b in band_blocks if b.bbox.center_x < mid_page_x]
            right_col = [b for b in band_blocks if b.bbox.center_x >= mid_page_x]

            if left_col and right_col:
                # Multi-column: read left column top-to-bottom, then right column top-to-bottom
                left_col.sort(key=lambda b: b.bbox.y0)
                right_col.sort(key=lambda b: b.bbox.y0)
                ordered_result.extend(left_col)
                ordered_result.extend(right_col)
            else:
                # Single column or all on one side: sort purely by y0
                band_blocks.sort(key=lambda b: b.bbox.y0)
                ordered_result.extend(band_blocks)

        # Ensure any blocks that might have missed banding (rare edge cases) are included
        included_ids = {b.id for b in ordered_result}
        for b in significant_blocks:
            if b.id not in included_ids:
                # Insert at approximate y position
                inserted = False
                for i, existing in enumerate(ordered_result):
                    if b.bbox.y0 < existing.bbox.y0:
                        ordered_result.insert(i, b)
                        inserted = True
                        break
                if not inserted:
                    ordered_result.append(b)
                included_ids.add(b.id)

        return ordered_result

    def _post_process_blocks(self, blocks: List[ContentBlock]) -> List[ContentBlock]:
        """Performs cleanup, merges consecutive short paragraphs when appropriate."""
        cleaned: List[ContentBlock] = []
        for block in blocks:
            # Strip excessive whitespace
            block.content = re.sub(r'\s+', ' ', block.content).strip()
            cleaned.append(block)
        return cleaned
