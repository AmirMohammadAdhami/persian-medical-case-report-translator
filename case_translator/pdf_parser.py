"""
PDF parser for medical case reports.
Extracts document structure, preserves multi-column reading order,
and positions figures and captions at their exact logical flow locations.
"""

import io
import logging
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from .figure_processor import FigureProcessor
from .models import BoundingBox, ContentBlock, Document, ElementType, Page

logger = logging.getLogger(__name__)


# PyMuPDF span flag bits (see TextPage.extractDICT docs).
_FLAG_ITALIC = 2
_FLAG_BOLD = 16

# Invisible characters that survive extraction and confuse both the structure
# heuristics and the translation model.
_SOFT_HYPHEN = "\u00ad"
# Private-use marker for a hyphenation break at the end of an extracted line.
# It is distinct from a real "-" so the line joiner can tell "pop\xad|ulation"
# (join into "population") from "three-\xad|dimensional" (join into
# "three-dimensional", keeping the hyphen that is part of the word).
_HYPHEN_BREAK = "\ue000"
_ZERO_WIDTH = "\u200b\ufeff\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
_EXOTIC_SPACES = "\u00a0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"


class PDFParser:
    """
    Parses medical case report PDFs into a structured Document model.
    Preserves multi-column reading order and exact relative figure positions.
    """

    SECTION_NAMES = (
        "Abstract|Introduction|Background|Case Report|Case Presentation|Case Description|"
        "Case Series|Clinical Findings|Observation|Observations|Materials and Methods|"
        "Patients and Methods|Methods|Methodology|Results|Discussion|Conclusion|Conclusions|"
        "References|Bibliography|Literature Cited|Acknowledgements|Acknowledgments|"
        "Conflict of Interest|Conflicts of Interest|Competing Interests|Declaration of Competing Interest|"
        "Author Contributions|Authors' Contributions|Funding|Funding Statement|Ethics Statement|"
        "Ethical Approval|Consent|Patient Consent|Data Availability Statement|Data Availability|"
        "Supporting Information|Abbreviations|Limitations|Appendix|Supplementary Material|ORCID"
    )

    HEADING_PATTERNS = [
        # A bare section label on its own line, optionally numbered.
        re.compile(
            rf'^\s*(?:\d+\s*[\.\)\|:]?\s*)?(?:{SECTION_NAMES})\s*[:.]?\s*$',
            re.IGNORECASE,
        ),
        # A numbered pipe heading, e.g. Wiley's "3 | Differential Diagnosis, ...".
        re.compile(r'^\s*\d+\s*\|\s*[A-Z]'),
        # A leading section label followed by more text on the same line.
        re.compile(rf'^\s*(?:\d+\s*[\.\)\|:]\s*)?(?:{SECTION_NAMES})\b', re.IGNORECASE),
    ]

    REFERENCE_ITEM_REGEX = re.compile(r'^\s*(?:\[\d+\]|\d{1,2}\.)\s+[A-Z]')

    # A reference's own number marker, e.g. "[3]" or "2.".
    # The digit count is deliberately capped: an unbounded \d+ matches a
    # publication year ("2000."), which then gets split off as if it were a
    # numbered entry — tearing the year away from the previous reference.
    REFERENCE_MARKER_REGEX = re.compile(r'^\s*(?:\[(\d{1,3})\]|(\d{1,2})[\.\)])\s+')

    # Markers *inside* a run of bibliography text. The lookahead requires a
    # letter (not a digit) so a DOI such as "10.1002/ccr3.73400" is not mistaken
    # for entry "10." followed by text.
    REFERENCE_SPLIT_REGEX = re.compile(
        r'(?:^|(?<=\s))(?P<marker>\[(?P<bracket>\d{1,3})\]|(?P<plain>\d{1,2})[\.\)])'
        r'[ \t\u202f]+(?=[A-Z“"\'\(]|[\u0600-\u06FF])'
    )

    REFERENCE_SECTION_REGEX = re.compile(
        r'^\s*(?:\d+\s*[\.\)\|:]\s*)?(references|bibliography|literature cited|'
        r'منابع|مراجع|منابع و مراجع)\s*[:.]?\s*$',
        re.IGNORECASE,
    )

    NON_REFERENCE_SECTION_REGEX = re.compile(
        r"^\s*(?:\d+\s*[\.\)\|:]\s*)?(acknowledg\w*|conflict\w*|competing\w*|"
        r"declaration of competing\w*|author contributions|authors' contributions|"
        r"funding\w*|ethics\w*|ethical approval|consent\w*|"
        r"data availability|supporting information|abbreviations|appendix|"
        r"supplementary\w*|orcid|conclusion\w*|discussion|abstract)\b",
        re.IGNORECASE,
    )

    # Fraction of page height treated as the running header/footer band.
    HEADER_FOOTER_BAND_RATIO = 0.09

    # Signals that a short block near a page edge is furniture rather than prose.
    RUNNING_HEADER_PATTERNS = [
        re.compile(r'^\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*$', re.IGNORECASE),
        re.compile(r'^\s*page\s+\d+', re.IGNORECASE),
        re.compile(r'\d+\s+of\s+\d+', re.IGNORECASE),          # "3 of 6"
        re.compile(r'case reports? in\b', re.IGNORECASE),
        re.compile(r'\bhindawi\b', re.IGNORECASE),
        re.compile(r'\bwiley\b', re.IGNORECASE),
        re.compile(r'\belsevier\b', re.IGNORECASE),
        re.compile(r'\bspringer\b', re.IGNORECASE),
        re.compile(r'\bmDPI\b'),
        re.compile(r'\bdoi\s*[:.]', re.IGNORECASE),
        re.compile(r'\bhttps?://', re.IGNORECASE),
        re.compile(r'\bwww\.', re.IGNORECASE),
        re.compile(r'\bonlinelibrary\b', re.IGNORECASE),
        re.compile(r'\bdownloaded from\b', re.IGNORECASE),
        re.compile(r'\bterms and conditions\b', re.IGNORECASE),
        re.compile(r'\bcreative commons\b', re.IGNORECASE),
        re.compile(r'\ball rights reserved\b', re.IGNORECASE),
        re.compile(r'\bcopyright\b', re.IGNORECASE),
        re.compile(r'\blicensed under\b', re.IGNORECASE),
        re.compile(r'^\s*(?:volume|vol\.|issue|no\.)\s*\d', re.IGNORECASE),
        re.compile(r'\b(?:journal|international journal|journal of)\b', re.IGNORECASE),
        re.compile(r'\bpublishing (?:corporation|corp|group|ltd)\b', re.IGNORECASE),
        re.compile(r'\b(?:received|revised|accepted|published|available online)\b\s*[:.]', re.IGNORECASE),
        re.compile(r'\borcid\b', re.IGNORECASE),
        re.compile(r'^\s*keywords?\b\s*[:.]?', re.IGNORECASE),
        re.compile(r'^\s*key\s+words\b\s*[:.]?', re.IGNORECASE),
        re.compile(r'^\s*correspondence\b', re.IGNORECASE),
        re.compile(r'^\s*(?:article id|article number|editor|reviewed by|academic editor)\b', re.IGNORECASE),
        re.compile(r'^\s*open access\b', re.IGNORECASE),
        re.compile(r'^\s*\d{4}\s+\w+\s*(?:\.|$)'),  # bare citation like "2011 Hindawi."
    ]

    # Institutional affiliations. These sit right under the author list, look
    # like prose to every other heuristic, and must never be translated.
    AFFILIATION_PATTERN = re.compile(
        r'^\s*(?:\d+\s*)?(?:department|division|school|faculty|college|university|hospital|'
        r'institute|institution|clinic|center|centre|graduate school|oral and maxillofacial|'
        r'دپارتمان|دانشکده|دانشگاه|بیمارستان)\b',
        re.IGNORECASE,
    )

    # Front-matter metadata. Unlike running heads these sit in the middle of
    # page 1, so the header-band test never sees them; they are short and often
    # bold, which means the heading rule used to translate them.
    FRONTMATTER_PATTERNS = [
        re.compile(r'^\s*correspondence\b', re.IGNORECASE),
        re.compile(r'^\s*(?:received|revised|accepted|published)\b\s*[:.]', re.IGNORECASE),
        re.compile(r'^\s*(?:keywords?|key\s+words)\b\s*[:.]', re.IGNORECASE),
        re.compile(r'^\s*(?:article id|article number|academic editor|reviewed by)\b', re.IGNORECASE),
        re.compile(r'^\s*orcid\b', re.IGNORECASE),
        re.compile(r'\bthis is an open access article\b', re.IGNORECASE),
        re.compile(r'\bcreative commons attribution\b', re.IGNORECASE),
        re.compile(r'\blicensed under\b', re.IGNORECASE),
        re.compile(r'\bterms and conditions\b', re.IGNORECASE),
        re.compile(r'\bdownloaded from\b', re.IGNORECASE),
        re.compile(r'\bwiley online library\b', re.IGNORECASE),
    ]

    # Capitalised name tokens, including hyphenated and apostrophe surnames.
    # A trailing digit is allowed because affiliation markers are attached
    # directly to the surname in most journal layouts ("Hideki Suito1").
    _NAME_TOKEN = re.compile(r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\d*")
    _NAMEY_TOKEN = re.compile(r"(?:[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\d*|[A-Z]{1,4}\.?\d*|\d{1,2})")
    _AUTHOR_PARTICLES = frozenset({
        "de", "van", "von", "der", "den", "del", "della", "la", "le", "el", "bin", "binti",
        "ibn", "and", "or", "et", "al", "jr", "sr", "ii", "iii", "md", "dds", "phd", "msc",
        "bds", "mds", "of", "the", "in", "on", "for", "university",
    })

    @classmethod
    def _looks_like_running_header(cls, text: str) -> bool:
        """True when a short block matches known header/footer/journal furniture."""
        return any(pattern.search(text) for pattern in cls.RUNNING_HEADER_PATTERNS)

    @classmethod
    def _looks_like_affiliation(cls, text: str) -> bool:
        return bool(cls.AFFILIATION_PATTERN.match(text))

    @classmethod
    def _looks_like_frontmatter(cls, text: str) -> bool:
        """True for publisher/administrative metadata that must not be translated."""
        candidate = (text or "").strip()
        if not candidate or len(candidate) > 600:
            return False
        return any(pattern.search(candidate) for pattern in cls.FRONTMATTER_PATTERNS)

    @classmethod
    def _looks_like_author_line(cls, text: str) -> bool:
        """
        True for a byline such as
        "Hideki Suito 1 | Yuuri Oku 1 | Tomoyuki Kondo 1 | Kumiko Kamada 2".

        A byline is a run of personal names and superscripts joined by pipes or
        commas. The discriminator against a title (which is also a run of
        capitalised words) is the separator: titles do not contain "|" and
        rarely contain two commas.
        """
        candidate = (text or "").strip()
        if not candidate or len(candidate) > 400:
            return False

        # A real sentence is never a byline. The lookbehind keeps author
        # initials ("John A. Smith") from being read as a sentence break.
        if re.search(r'(?<![A-Z])[.!?]\s+[A-Z]', candidate):
            return False

        has_separator = ("|" in candidate) or ("•" in candidate) or (candidate.count(",") >= 2)
        if not has_separator:
            return False

        words = [w for w in re.split(r'[\s|,;•&]+', candidate) if w]
        if len(words) < 3 or len(words) > 60:
            return False

        namey = sum(
            1 for w in words
            if cls._NAMEY_TOKEN.fullmatch(w) or w.lower() in cls._AUTHOR_PARTICLES
        )
        if namey / len(words) < 0.6:
            return False

        return len(cls._NAME_TOKEN.findall(candidate)) >= 3

    def __init__(self, assets_dir: Path):
        self.assets_dir = Path(assets_dir)
        self.figure_processor = FigureProcessor(assets_dir=self.assets_dir)

    # -- public API --------------------------------------------------------

    def reset_assets(self) -> int:
        """Removes figures extracted by a previous run (see FigureProcessor)."""
        return self.figure_processor.reset_assets()

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

            # Link captions to figures (and absorb multi-panel sub-images)
            self.figure_processor.link_captions_to_figures(
                figure_blocks, raw_text_blocks, page_height=page_height
            )

            # Tables: detected before ordering so their text can be excluded
            # from the prose stream instead of being translated twice.
            table_blocks = self._extract_tables(page, page_num)
            if table_blocks:
                raw_text_blocks = self._drop_blocks_inside_tables(raw_text_blocks, table_blocks)

            # Drop figure blocks whose panels were absorbed into another figure.
            figure_blocks = [f for f in figure_blocks if not f.metadata.get("merged_into")]

            # Combine text and figure blocks for column-aware ordering
            all_page_blocks = raw_text_blocks + figure_blocks + table_blocks

            # Reconstruct reading order for multi-column layout
            ordered_blocks = self._reorder_blocks_for_reading(all_page_blocks, page_width, page_height)

            # Reflow: several publishers (and every "print to PDF" from a phone
            # or reader app) emit one text block per visual line. Left alone,
            # every line becomes its own translation request and the model sees
            # sentence fragments with no context.
            ordered_blocks = self._merge_line_fragments(ordered_blocks, page_height)

            # Post-process: clean and de-hyphenate text
            cleaned_blocks = self._post_process_blocks(ordered_blocks)

            document.pages.append(
                Page(
                    page_number=page_num,
                    width=page_width,
                    height=page_height,
                    blocks=cleaned_blocks,
                )
            )

        # A paragraph split by a page break must be put back together: the
        # model would otherwise receive the first half and the second half as
        # two unrelated fragments.
        self._merge_across_pages(document)

        # Title comes from the largest-font block on page 1 (if any).
        self._record_title(document)

        # Panels of one printed figure arrive as several image XObjects; group
        # them before numbering so "Figure 1" is one entry, not four.
        absorbed = self.figure_processor.group_adjacent_figures(document)
        if absorbed:
            logger.info("Grouped %d figure panel(s) into their parent figures.", absorbed)
        self._renumber_figures(document)

        # Publisher mastheads sit above the article title: journal name, banner
        # ads, "OPEN ACCESS" ribbons. Nothing meaningful is ever there.
        self._suppress_page1_masthead(document)

        # Cross-page pass: running headers/footers repeat verbatim on multiple
        # pages. This catches furniture the keyword heuristic misses, e.g. when
        # the journal banner carries no recognisable publisher name.
        self._drop_repeated_running_headers(document)

        # Reference lists are bibliography, not prose: they must stay in English
        # so journal names, DOIs and volume/issue formatting keep their academic
        # integrity. Done after the page loop so it sees the whole document and
        # can follow a bibliography across a page break.
        self._classify_references(document)
        self.mark_reference_runs(document)
        self._mark_reference_section_runs(document)

        doc.close()
        return document

    # -- text extraction ---------------------------------------------------

    @staticmethod
    def _normalise_text(text: str) -> str:
        """
        Cleans extraction artefacts that break both heuristics and translation.

        Soft hyphens are removed together with any following whitespace: a
        soft hyphen marks a line-break hyphenation point, so "pop\\xad ulation"
        is really "population", not "pop ulation".
        """
        if not text:
            return ""
        # Zero-width and bidi control characters must go first: a soft hyphen
        # is often followed by a zero-width space and then a real space, and
        # removing the hyphen before those would leave "pop ulation" behind.
        for ch in _ZERO_WIDTH:
            text = text.replace(ch, "")
        text = text.translate({ord(ch): " " for ch in _EXOTIC_SPACES})
        # A soft hyphen at the very end of a line marks a hyphenation break
        # across the line boundary ("pop\xad" + "ulation"). It becomes a break
        # marker so the line joiner can put the word back together; every other
        # soft hyphen is simply dropped.
        text = re.sub(rf'{_SOFT_HYPHEN}(?=\s*$)', _HYPHEN_BREAK, text)
        text = re.sub(rf'{_SOFT_HYPHEN}\s*', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

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
            rotated = False

            for line in raw_b.get("lines", []):
                direction = line.get("dir", (1.0, 0.0))
                if direction and abs(direction[1]) > 0.3:
                    # Sideways margin text (publisher watermarks, "downloaded
                    # from" ribbons). It is never part of the article.
                    rotated = True

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
                        # NOTE: bit 2 is ITALIC in PyMuPDF, not bold. Treating
                        # italic as bold turned every italic reference entry into
                        # a "heading" and sent author lists to the model.
                        if (flags & _FLAG_BOLD) or any(
                            marker in font_name
                            for marker in ("bold", "black", "heavy", "semibold", "demibold")
                        ):
                            is_bold = True
                        total_chars += len(span_text.strip())

                line_str = self._normalise_text("".join(line_text_parts))
                if line_str:
                    line_strings.append(line_str)

            if not line_strings:
                continue

            # Merge lines with de-hyphenation
            full_text = self._join_lines(line_strings)
            if not full_text.strip():
                continue

            # Discard extraction noise: single soft hyphens, stray superscript
            # digits and whitespace-only blocks carry no content.
            if len(re.sub(r'[^A-Za-z0-9\u0600-\u06FF]', '', full_text)) < 2:
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
                page_height=page.rect.height,
                rotated=rotated,
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
                    "rotated": rotated,
                }
            )
            blocks.append(block)

        return blocks

    @staticmethod
    def _join_lines(lines: List[str]) -> str:
        """Joins line fragments with proper space and de-hyphenation."""
        if not lines:
            return ""

        result = lines[0]
        for line in lines[1:]:
            if result.endswith(_HYPHEN_BREAK):
                # A soft-hyphen break: the word continues on the next line. If
                # a real hyphen preceded the break ("three-\xad"), that hyphen
                # belongs to the word and is kept.
                result = result[:-1] + line
            # A bare trailing hyphen is a line-break hyphenation artefact
            # ("maxillofacial re-" + "gion" -> "maxillofacial region").
            elif result.endswith("-") and len(result) > 1 and result[-2].isalpha() and line and line[0].isalpha():
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
        page_height: float,
        rotated: bool = False,
    ) -> ElementType:
        """Classifies a text block into TITLE, AUTHOR, HEADING, PARAGRAPH, etc."""
        text_clean = text.strip()

        if rotated:
            return ElementType.OTHER

        # Publisher/administrative metadata anywhere on the page (correspondence
        # block, received/accepted dates, keywords, licence boilerplate).
        if self._looks_like_frontmatter(text_clean):
            return ElementType.OTHER

        # Check for header/footer artifacts (running heads, journal banners,
        # page numbers). The band is a fraction of page height rather than a
        # fixed point value, because a hardcoded threshold silently missed
        # banners that sit slightly lower on the page.
        top_band = page_height * self.HEADER_FOOTER_BAND_RATIO
        bottom_band = page_height - top_band
        if bbox.y1 <= top_band or bbox.y0 >= bottom_band:
            # The length cap is generous: a publisher banner can be a whole
            # sentence ("Hindawi Publishing Corporation, Case Reports in
            # Dentistry, Volume 2011, Article ID 401678, 5 pages") and the old
            # 100-character cap let those through to translation.
            if len(text_clean) < 300 and self._looks_like_running_header(text_clean):
                return ElementType.OTHER

        # Check for Title on first page
        if page_num == 1 and max_font_size >= (body_font_size * 1.35) and len(text_clean) < 300 and bbox.y0 < 300:
            return ElementType.TITLE

        # Check for Figure Caption
        if FigureProcessor.CAPTION_REGEX.search(text_clean):
            return ElementType.FIGURE_CAPTION

        # Institutional affiliations sit under the byline and are never prose.
        if self._looks_like_affiliation(text_clean) and len(text_clean) < 500:
            return ElementType.OTHER

        # Check for Section Headings
        for pattern in self.HEADING_PATTERNS:
            if pattern.search(text_clean) and len(text_clean) < 100:
                return ElementType.HEADING

        # Author byline: must be checked before the bold/short "heading" rule,
        # which used to swallow the author list and send names to the model.
        if page_num == 1 and self._looks_like_author_line(text_clean):
            return ElementType.AUTHOR

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

    # -- tables ------------------------------------------------------------

    def _extract_tables(self, page: pymupdf.Page, page_num: int) -> List[ContentBlock]:
        """
        Detects ruled tables and returns them as TABLE blocks.

        Previously ElementType.TABLE was never produced, so laboratory tables
        were flattened into paragraphs and machine-translated cell by cell with
        no structure left. PyMuPDF's table finder gives us the grid back.
        """
        try:
            # find_tables() prints a package recommendation to stdout on first
            # use; keep the CLI output clean.
            with redirect_stdout(io.StringIO()):
                finder = page.find_tables()
        except Exception as exc:  # noqa: BLE001 - table detection is best-effort
            logger.debug("Table detection failed on page %s: %s", page_num, exc)
            return []

        tables: List[ContentBlock] = []
        page_area = abs(page.rect.get_area()) or 1.0

        for index, table in enumerate(getattr(finder, "tables", []) or []):
            try:
                rows = table.extract()
            except Exception:  # noqa: BLE001
                continue

            rows = [
                [(self._normalise_text(cell) if cell else "") for cell in row]
                for row in rows or []
                if row
            ]
            rows = [row for row in rows if any(cell for cell in row)]
            if len(rows) < 2:
                continue
            if max(len(row) for row in rows) < 2:
                continue

            bbox = BoundingBox(*table.bbox)
            # A "table" covering the whole page is a false positive from the
            # layout analyser (it happens on two-column pages).
            if bbox.width * bbox.height > 0.8 * page_area:
                continue

            content = "\n".join(" | ".join(row) for row in rows)
            tables.append(
                ContentBlock(
                    id=f"p{page_num}_t{index + 1}",
                    type=ElementType.TABLE,
                    content=content,
                    bbox=bbox,
                    page_number=page_num,
                    metadata={"table_rows": rows, "row_count": len(rows)},
                )
            )

        return tables

    @staticmethod
    def _drop_blocks_inside_tables(
        text_blocks: List[ContentBlock],
        table_blocks: List[ContentBlock],
    ) -> List[ContentBlock]:
        """Removes prose blocks whose text is already represented by a table."""
        kept: List[ContentBlock] = []
        for block in text_blocks:
            inside = False
            for table in table_blocks:
                if (
                    block.bbox.center_x >= table.bbox.x0
                    and block.bbox.center_x <= table.bbox.x1
                    and block.bbox.center_y >= table.bbox.y0
                    and block.bbox.center_y <= table.bbox.y1
                ):
                    inside = True
                    break
            if not inside:
                kept.append(block)
        return kept

    # -- reading order -----------------------------------------------------

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

    # -- line reflow -------------------------------------------------------

    def _merge_line_fragments(
        self,
        blocks: List[ContentBlock],
        page_height: float,
    ) -> List[ContentBlock]:
        """
        Merges consecutive paragraph blocks that are really one paragraph.

        Some PDFs — notably anything exported by a mobile "print to PDF" or a
        reader app — place every visual line in its own text block. Feeding
        those to a translator as-is produces fragments ("role in the resultant
        properties of the final") and multiplies the number of API calls by the
        number of lines. This pass stitches them back together.

        The rules are deliberately conservative: same font size, same column
        band, vertically adjacent, and the previous fragment must not end a
        sentence. Anything else is left alone.
        """
        if not blocks:
            return []

        merged: List[ContentBlock] = []

        for block in blocks:
            if (
                merged
                and block.type == ElementType.PARAGRAPH
                and merged[-1].type == ElementType.PARAGRAPH
                and self._is_line_continuation(merged[-1], block, page_height)
            ):
                previous = merged[-1]
                previous.content = self._join_lines([previous.content, block.content])
                previous.metadata["merged_fragments"] = previous.metadata.get("merged_fragments", 1) + 1
                previous.bbox = BoundingBox(
                    min(previous.bbox.x0, block.bbox.x0),
                    previous.bbox.y0,
                    max(previous.bbox.x1, block.bbox.x1),
                    max(previous.bbox.y1, block.bbox.y1),
                )
                continue
            merged.append(block)

        return merged

    @classmethod
    def _is_line_continuation(
        cls,
        previous: ContentBlock,
        current: ContentBlock,
        page_height: float,
    ) -> bool:
        """True when `current` is the next line of `previous`'s paragraph."""
        # Same typography.
        prev_size = previous.metadata.get("font_size") or 0.0
        curr_size = current.metadata.get("font_size") or 0.0
        if prev_size and curr_size and abs(prev_size - curr_size) > 0.6:
            return False

        # Same column: the horizontal spans must line up.
        overlap = min(previous.bbox.x1, current.bbox.x1) - max(previous.bbox.x0, current.bbox.x0)
        min_width = min(previous.bbox.width, current.bbox.width) or 1.0
        if overlap / min_width < 0.55:
            return False

        # Vertically adjacent (one line apart, never overlapping or far away).
        line_height = (prev_size or curr_size or 10.0) * 1.9
        gap = current.bbox.y0 - previous.bbox.y1
        if gap < -2.0 or gap > line_height:
            return False

        # A fragment that ends a sentence starts a new paragraph after it.
        tail = previous.content.rstrip()
        if not tail or tail[-1] in ".!?:;،؛»\"":
            return False

        return True

    def _merge_across_pages(self, document: Document) -> None:
        """
        Joins a paragraph that continues from the bottom of one page to the top
        of the next.

        The conditions are deliberately tight — the earlier block must not end a
        sentence and the later one must start mid-word (lowercase), in the same
        column band and at the same font size. In a two-column journal the last
        block of a page and the first block of the next one are exactly the two
        halves of one paragraph, so this is also the correct reading order.
        """
        for index in range(len(document.pages) - 1):
            current = document.pages[index]
            following = document.pages[index + 1]
            if not current.blocks or not following.blocks:
                continue

            # A figure placed at the bottom of the page must not hide the
            # paragraph that actually continues on the next page.
            last = next(
                (b for b in reversed(current.blocks) if b.type == ElementType.PARAGRAPH),
                None,
            )
            first = next(
                (
                    b for b in following.blocks
                    if b.type not in (
                        ElementType.FIGURE,
                        ElementType.FIGURE_CAPTION,
                        ElementType.TABLE,
                    )
                ),
                None,
            )
            if last is None or first is None:
                continue
            if last.type != ElementType.PARAGRAPH or first.type != ElementType.PARAGRAPH:
                continue

            prev_size = last.metadata.get("font_size") or 0.0
            curr_size = first.metadata.get("font_size") or 0.0
            if prev_size and curr_size and abs(prev_size - curr_size) > 0.6:
                continue

            overlap = min(last.bbox.x1, first.bbox.x1) - max(last.bbox.x0, first.bbox.x0)
            min_width = min(last.bbox.width, first.bbox.width) or 1.0
            same_column = (overlap / min_width) >= 0.55
            if not same_column:
                # In a two-column journal the paragraph continues from the
                # bottom of the right column to the top of the left column on
                # the next page, so a column change is expected there. Require
                # the previous half to sit in the lower part of its page; the
                # next half must also be the first prose block of the next page
                # (it already is, by construction above).
                at_page_bottom = last.bbox.y1 >= current.height * 0.6
                if not at_page_bottom:
                    continue

            tail = last.content.rstrip()
            if not tail or tail[-1] in ".!?:;،؛»\"":
                continue
            if not first.content[:1].islower():
                continue

            last.content = self._join_lines([last.content, first.content])
            last.bbox = BoundingBox(
                min(last.bbox.x0, first.bbox.x0),
                last.bbox.y0,
                max(last.bbox.x1, first.bbox.x1),
                first.bbox.y1,
            )
            last.metadata["merged_across_page"] = index + 2
            following.blocks.remove(first)

    # -- document-level cleanups ------------------------------------------

    @staticmethod
    def _record_title(document: Document) -> None:
        """Stores the document title (largest-font block on page 1) in metadata."""
        for block in document.get_ordered_blocks():
            if block.type == ElementType.TITLE and block.content.strip():
                document.metadata["title"] = block.content.strip()
                return

    @staticmethod
    def _renumber_figures(document: Document) -> None:
        """
        Numbers figures in reading order, preferring an explicit caption number.

        The counter used to be a parse-time global that only *happened* to line
        up with the caption numbering; a multi-panel figure or an unnumbered
        image shifted every later figure by one.
        """
        counter = 0
        for page in document.pages:
            for block in page.blocks:
                if block.type != ElementType.FIGURE:
                    continue
                caption_num = block.metadata.get("caption_num")
                if caption_num:
                    counter = int(caption_num)
                else:
                    counter += 1
                block.metadata["figure_num"] = counter

    def _suppress_page1_masthead(self, document: Document) -> None:
        """
        Drops anything sitting above the article title on page 1.

        Journal names, "OPEN ACCESS" ribbons and banner blocks live there and
        carry no article content, but they are short and bold, so the heading
        heuristics happily translated them.
        """
        if not document.pages:
            return

        page = document.pages[0]
        title_block = next((b for b in page.blocks if b.type == ElementType.TITLE), None)
        if title_block is None:
            candidates = [
                b for b in page.blocks
                if b.type not in (ElementType.FIGURE, ElementType.FIGURE_CAPTION, ElementType.TABLE)
            ]
            if not candidates:
                return
            title_block = max(candidates, key=lambda b: b.metadata.get("font_size") or 0.0)
            if (title_block.metadata.get("font_size") or 0.0) <= 0:
                return

        kept: List[ContentBlock] = []
        # Only trust the reference point when it sits in the upper half of the
        # page; otherwise "everything above it" could swallow the whole body.
        if title_block.bbox.y0 >= page.height * 0.5:
            return
        for block in page.blocks:
            if block.type in (ElementType.TITLE, ElementType.FIGURE, ElementType.FIGURE_CAPTION, ElementType.TABLE):
                kept.append(block)
                continue
            if block.bbox.y1 <= title_block.bbox.y0 + 2 and len(block.content) < 300:
                continue
            kept.append(block)
        page.blocks = kept

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
                if not text or len(text) > 400:
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

    # -- references --------------------------------------------------------

    def _classify_references(self, document: Document) -> None:
        """
        Converts everything between the "References" heading and the next
        section heading into REFERENCE blocks, and splits numbered entries.

        The old per-block heuristic only recognised an entry when it started
        with "[n]" or "n." AND the block itself looked like a citation. In real
        journal PDFs entries start mid-block ("... 2011;12(3):245-50. 15. J. M.
        Bernaba, ...") or continue onto the next page, so most of the
        bibliography was left as PARAGRAPH and machine-translated — the exact
        opposite of what the README promises.
        """
        in_references = False
        last_reference: Optional[ContentBlock] = None

        for page in document.pages:
            new_blocks: List[ContentBlock] = []

            for block in page.blocks:
                if block.type == ElementType.HEADING:
                    if self.REFERENCE_SECTION_REGEX.match(block.content):
                        in_references = True
                        last_reference = None
                        new_blocks.append(block)
                        continue
                    if in_references and self.NON_REFERENCE_SECTION_REGEX.match(block.content):
                        in_references = False
                        last_reference = None
                        new_blocks.append(block)
                        continue

                if not in_references:
                    new_blocks.append(block)
                    continue

                if block.type in (ElementType.FIGURE, ElementType.FIGURE_CAPTION, ElementType.TABLE):
                    new_blocks.append(block)
                    last_reference = None
                    continue

                pieces = self._split_reference_text(block.content)

                if pieces is None:
                    # No marker at all. Either a continuation of the previous
                    # entry that spilled onto this page/column, or a marker-less
                    # entry (some publishers omit them).
                    if last_reference is not None and self._is_reference_continuation(
                        last_reference.content, block.content
                    ):
                        last_reference.content = self._join_lines(
                            [last_reference.content, block.content]
                        )
                        last_reference.bbox = BoundingBox(
                            min(last_reference.bbox.x0, block.bbox.x0),
                            last_reference.bbox.y0,
                            max(last_reference.bbox.x1, block.bbox.x1),
                            max(last_reference.bbox.y1, block.bbox.y1),
                        )
                        continue
                    sub = self._make_reference_block(block, block.content, None, 0)
                    new_blocks.append(sub)
                    last_reference = sub
                    continue

                for position, (number, text) in enumerate(pieces):
                    if not text.strip():
                        continue
                    sub = self._make_reference_block(block, text, number, position)
                    new_blocks.append(sub)
                    last_reference = sub

            page.blocks = new_blocks

    @staticmethod
    def _is_reference_continuation(previous_text: str, text: str) -> bool:
        """True when `text` looks like the tail of the previous entry."""
        stripped = (text or "").strip()
        if not stripped:
            return False
        # Starts mid-sentence -> continuation.
        if stripped[0].islower():
            return True
        # The previous entry was cut mid-sentence -> this completes it.
        tail = (previous_text or "").rstrip()
        return bool(tail) and tail[-1] not in ".!?»\""

    @classmethod
    def _split_reference_text(cls, content: str):
        """
        Splits bibliography text into (number, text) entries.

        Returns None when the text contains no entry marker, so the caller can
        treat it as a continuation instead of guessing.
        """
        matches = list(cls.REFERENCE_SPLIT_REGEX.finditer(content or ""))
        if not matches:
            return None

        pieces = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            body = content[start:end].strip()
            number_token = match.group("bracket") or match.group("plain")
            try:
                number = int(number_token)
            except (TypeError, ValueError):
                number = None
            pieces.append((number, body))

        # Text before the first marker belongs to the previous entry.
        prefix = content[:matches[0].start()].strip()
        if prefix:
            pieces.insert(0, (None, prefix))

        return pieces

    @staticmethod
    def _make_reference_block(
        template: ContentBlock,
        text: str,
        number: Optional[int],
        position: int,
    ) -> ContentBlock:
        suffix = f"_ref{position + 1}" if position else ""
        metadata: Dict[str, Any] = {"is_numbered_ref": number is not None}
        if number is not None:
            metadata["ref_number"] = number
        return ContentBlock(
            id=f"{template.id}{suffix}",
            type=ElementType.REFERENCE,
            content=text.strip(),
            bbox=template.bbox,
            page_number=template.page_number,
            metadata=metadata,
        )

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

    @staticmethod
    def _mark_reference_section_runs(document: Document) -> None:
        """
        Adds blocks classified inside a References section to the run set.

        A bibliography page can legitimately contain a single entry (a short
        page, or the first page of the list), which `mark_reference_runs`
        deliberately ignores when it stands alone among prose. Blocks that came
        from an explicit References section are unambiguous, so they are added
        on top of that heuristic.
        """
        in_references = False
        for page in document.pages:
            run_ids = set(page.metadata.get("ref_run_ids") or set())
            for block in page.blocks:
                if block.type == ElementType.HEADING:
                    if PDFParser.REFERENCE_SECTION_REGEX.match(block.content):
                        in_references = True
                        continue
                    if in_references:
                        in_references = False
                        continue
                if in_references and block.type == ElementType.REFERENCE:
                    run_ids.add(block.id)
            page.metadata["ref_run_ids"] = run_ids

    # -- post processing ---------------------------------------------------

    # Text exported from a screen capture / reflowed reader sometimes carries an
    # overlapping duplicate layer, so a run of words appears twice in a row.
    _DUPLICATED_RUN = re.compile(r'(\b.{20,}?)\s+\1\b')

    @classmethod
    def _collapse_duplicated_text(cls, text: str) -> str:
        """
        Collapses an immediately repeated phrase ("...graduates recent dental
        graduates recent dental graduates" -> "...recent dental graduates").

        Only runs of 20+ characters are collapsed, so ordinary emphatic
        repetition is left alone.
        """
        previous = None
        while previous != text:
            previous = text
            text = cls._DUPLICATED_RUN.sub(r'\1', text)
        return text

    def _post_process_blocks(self, blocks: List[ContentBlock]) -> List[ContentBlock]:
        """Performs cleanup, merges consecutive short paragraphs when appropriate."""
        cleaned: List[ContentBlock] = []
        for block in blocks:
            if block.type == ElementType.TABLE:
                # Table rows are newline-separated; collapsing all whitespace
                # would flatten the grid back into one line.
                block.content = re.sub(r'[^\S\n]+', ' ', block.content).strip()
                rows = block.metadata.get("table_rows")
                if rows:
                    block.metadata["table_rows"] = [
                        [re.sub(r'\s+', ' ', cell).strip() for cell in row] for row in rows
                    ]
            else:
                # Strip excessive whitespace
                block.content = re.sub(r'\s+', ' ', block.content).strip()
                # A hyphenation break that never got joined (the word continues
                # in a block we deliberately did not merge) must not leak the
                # private-use marker into the translated output.
                block.content = block.content.replace(_HYPHEN_BREAK, "")
                if block.type in (ElementType.PARAGRAPH, ElementType.HEADING, ElementType.TITLE):
                    block.content = self._collapse_duplicated_text(block.content)
            if not block.content and block.type != ElementType.FIGURE:
                continue
            cleaned.append(block)
        return cleaned
