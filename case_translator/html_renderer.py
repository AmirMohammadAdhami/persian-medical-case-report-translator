"""
HTML renderer for generating clean, responsive, RTL Persian medical case report documents.
Preserves original reading order, relative figure placement, and mixed Persian/English typography.
"""

import html
import json
import re
from pathlib import Path
from typing import List, Optional

from .glossary import is_anatomical_term
from .models import ContentBlock, Document, ElementType


# Document-structure labels that are not clinical terms.
_NON_TERM_LABELS = frozenset({
    "case report", "case reports", "case presentation", "case description",
    "abstract", "introduction", "discussion", "conclusion", "conclusions",
    "references", "bibliography", "acknowledgements", "acknowledgments",
    "conflict of interest", "conflicts of interest", "author contributions",
})

# A glossary term is a clinical concept. Institutional affiliations look like
# terms to the pair regex ("گروه دندان‌پزشکی ترمیمی و اندودنتیکس (Conservative
# Dentistry and Endodontics)") but are not, so they are excluded by keyword.
_NON_TERM_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\b(?:department|school|faculty|college|university|hospital|institute|"
    r"clinic|center|centre|division|unit|academy|society|association)\b",
    r"\b(?:dental school|medical center|health sciences)\b",
    # Discipline/department names, which are organisational even when they
    # contain clinical words ("Conservative Dentistry and Endodontics").
    r"\b(?:dentistry|endodontics|periodontics|orthodontics|prosthodontics|"
    r"restorative dentistry|oral medicine|oral surgery)\b",
))


def _is_non_term_label(english: str) -> bool:
    """True when the parenthetical is a label or affiliation, not a term."""
    text = (english or "").strip()
    if text.lower() in _NON_TERM_LABELS:
        return True
    return any(pattern.search(text) for pattern in _NON_TERM_PATTERNS)


class HTMLRenderer:
    """
    Renders a Document model into an elegant, self-contained Persian RTL HTML document.
    """

    EN_PAREN_REGEX = re.compile(r'(\([A-Za-z0-9\s,;:\-\./\+%]+\))')

    # A "[Persian] ([English])" pair, as produced by the translation prompt.
    # Anchored on the parenthetical, because that half has a reliable, checkable
    # shape. The Persian side is then taken by scanning back from the opening
    # bracket (see _persian_prefix); it cannot be matched directly, as its word
    # count is unknowable and any quantifier either swallows the preceding
    # clause or cuts into the term itself.
    GLOSS_PAIR_REGEX = re.compile(
        r'\(\s*(?P<en>[A-Za-z][A-Za-z0-9\s,;:\-\./\+%\'’]*?)\s*\)'
    )

    # Characters that end a candidate Persian term. Persian uses ZWNJ (U+200C)
    # inside words, so it is kept; the Arabic comma/semicolon and Latin
    # punctuation are boundaries.
    _TERM_BOUNDARY = "،؛:.,;!؟?)"
    # Sentence terminators. A term never contains one, and prose after a
    # sentence end is never part of the term, so the scan stops immediately.
    _SENTENCE_END = ".!؟?"

    # Upper bound on a plausible term, by word count and by character length.
    # Exceeding either aborts the scan rather than truncating, so prose is never
    # silently clipped into a term.
    _MAX_TERM_WORDS = 5
    _MAX_TERM_CHARS = 60

    # Standalone function words that introduce or join terms and are not part of
    # them. Without this, "پست فایبر (Fiber Post) و رزین کامپوزیت (Composite
    # Resin)" makes the conjunction "و" part of the second clickable term.
    _TERM_STOPWORDS = frozenset({
        # conjunctions / prepositions
        "و", "یا", "با", "از", "به", "در", "بر", "که", "تا", "را", "برای",
        "این", "آن", "هم", "نیز", "اما", "ولی", "روی", "بین", "پس", "مورد",
        "طور", "صورت", "دلیل", "وسیله", "عنوان", "جهت", "بدون", "داخل",
        # light verbs and copulas that introduce a clause, not a term
        "شد", "شده", "شود", "کرد", "کرده", "کند", "است", "بود", "گردید",
        "می", "نمی", "دارد", "دارای", "داد", "یافت", "گشت", "انجام",
    })

    @classmethod
    def _persian_prefix(cls, text: str):
        """
        Returns (term, consumed_length) for the Persian term before a bracket.

        Walks back from the bracket to the nearest real boundary, preferring a
        sentence end. A term never straddles a sentence boundary, so prose after
        the previous full stop is never part of it — a length cap alone would
        stop mid-word and capture a run of unrelated prose instead.

        The scan gives up (rather than truncating) once the run exceeds
        _MAX_TERM_CHARS, so an over-long candidate yields no term at all and the
        parenthetical is rendered as a plain English term.
        """
        end = len(text)
        i = end
        while i > 0:
            ch = text[i - 1]
            # A sentence end means the preceding text belongs to another
            # sentence, not to this term.
            if ch in cls._SENTENCE_END:
                break
            if ch in "()[]«»\"'":
                return "", 0
            if ch in cls._TERM_BOUNDARY:
                break
            if end - i > cls._MAX_TERM_CHARS:
                # Too long to be a term and no boundary was found: give up
                # instead of returning a truncated fragment of prose.
                return "", 0
            i -= 1

        raw = text[i:end]
        term = raw.strip()
        if not term:
            return "", 0

        words = term.split()
        # A term is the trailing noun phrase of the run. Strip leading words that
        # are standalone function words or verbs, since they belong to the
        # surrounding prose ("استفاده شد و پست فایبر" -> "پست فایبر"). Trimming
        # stops at the first content word, so a genuine multi-word term is kept.
        while len(words) > 1 and words[0] in cls._TERM_STOPWORDS:
            words.pop(0)
        term = " ".join(words)

        # A clinical noun phrase is short. Anything longer is prose that happened
        # to sit next to the bracket, so refuse it rather than linking it.
        if not term or len(words) > cls._MAX_TERM_WORDS:
            return "", 0

        consumed = len(raw) - len(raw.rstrip()) + len(term)
        return term, consumed

    @staticmethod
    def _normalise_term(text: str) -> str:
        """Collapses whitespace so a term matches regardless of source wrapping."""
        return re.sub(r'\s+', ' ', text or '').strip()

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # English terms seen while rendering, in document order.
        self.glossary_terms: List[str] = []

    def format_bidi_text(self, text: str) -> str:
        """
        Escapes HTML, wraps English parenthetical terms in LTR spans, and turns
        "[Persian] ([English])" pairs into clickable glossary terms.

        Order matters: the glossary scan MUST run first, on the raw text. Every
        clickable pair *contains* a parenthetical, so if en-term wrapping ran
        first it would consume the pair's English half and no pair would ever
        match. The glossary pass emits placeholders that the en-term pass then
        cannot reach inside.
        """
        if not text:
            return ""

        escaped = html.escape(text)

        def _to_en_span(match: re.Match) -> str:
            return f'<span class="en-term" dir="ltr">{match.group(1)}</span>'

        # Generated markup is parked here and swapped in at the end, so the
        # en-term pass can never rewrite anything inside it.
        pending: List[str] = []

        def _to_en_span(match: re.Match) -> str:
            return f'<span class="en-term" dir="ltr">{match.group(1)}</span>'

        # 1. Glossary pairs first, replaced by placeholders. The placeholder
        #    consumes the Persian term as well as the bracket, so no double
        #    space is left behind.
        result_parts: List[str] = []
        cursor = 0
        for match in self.GLOSS_PAIR_REGEX.finditer(escaped):
            prefix, consumed = self._persian_prefix(escaped[cursor:match.start()])
            if not prefix:
                # No Persian term in front of the bracket: leave it for the
                # en-term pass rather than inventing a clickable span.
                continue
            english = self._normalise_term(match.group("en"))
            # Anatomical names are deliberately kept in English and have no
            # Persian term to explain, so they are not clickable.
            if is_anatomical_term(english):
                continue
            # Section labels and institutional affiliations are structural, not
            # clinical terms; making them clickable just adds noise.
            if _is_non_term_label(english):
                continue
            if english.lower() not in {t.lower() for t in self.glossary_terms}:
                self.glossary_terms.append(english)
            pending.append(
                f'<span class="glossary-term" data-term="{html.escape(english, quote=True)}" '
                f'tabindex="0" role="button">{prefix}'
                f'<span class="term-en" dir="ltr"> ({html.escape(english)})</span></span>'
            )
            result_parts.append(escaped[cursor:match.start() - consumed])
            result_parts.append(f"\x00{len(pending) - 1}\x00")
            cursor = match.end()
        result_parts.append(escaped[cursor:])
        marked = "".join(result_parts)

        # 2. Remaining bare parentheticals get LTR isolation. Placeholders hold
        #    no parentheses, so they are untouched by this pass.
        marked = self.EN_PAREN_REGEX.sub(_to_en_span, marked)
        # 3. Restore the generated glossary markup.
        for index, markup in enumerate(pending):
            marked = marked.replace(f"\x00{index}\x00", markup)
        return marked

    def render(self, document: Document, filename: str = "article.html") -> Path:
        """
        Renders the document into an HTML file at output_dir / filename.
        Returns the path to the written HTML file.
        """
        output_file = self.output_dir / filename
        blocks = document.get_ordered_blocks()

        # The HTML template embeds an explanation for every clickable term, and
        # that list is finalised as the body is rendered. Reset it here so a
        # second render never carries terms in from an earlier document.
        self.glossary_terms = []

        title_text = document.metadata.get("title", "گزارش مورد بالینی")
        translated_title = title_text
        for b in blocks:
            if b.type == ElementType.TITLE and b.translated_content:
                translated_title = b.translated_content
                break

        body_content = self._render_body(blocks)

        html_content = self._generate_html_template(
            title=translated_title,
            original_title=title_text if title_text != translated_title else "",
            body_content=body_content,
            metadata=document.metadata
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_file

    def _render_body(self, blocks) -> str:
        """
        Builds the body HTML for the ordered blocks.

        Split out of render() so reference numbering can be unit-tested without
        writing files or building a full Document.
        """
        body_html_parts: List[str] = []
        in_references = False
        # Running fallback for references whose number the parser could not read.
        ref_counter = 1

        for block in blocks:
            content = block.translated_content or block.content
            if not content and block.type != ElementType.FIGURE:
                continue

            if block.type == ElementType.TITLE:
                # Rendered in header
                continue

            elif block.type == ElementType.HEADING:
                if in_references:
                    body_html_parts.append('</ol></section>')
                    in_references = False

                formatted = self.format_bidi_text(content)
                body_html_parts.append(
                    f'<h2 class="section-heading" id="{block.id}">{formatted}</h2>'
                )

                if "مرجع" in content or "منابع" in content or "reference" in block.content.lower():
                    in_references = True
                    body_html_parts.append('<section class="references-section"><ol class="references-list">')

            elif block.type == ElementType.PARAGRAPH:
                if in_references:
                    body_html_parts.append('</ol></section>')
                    in_references = False

                formatted = self.format_bidi_text(content)
                body_html_parts.append(f'<p class="article-paragraph" id="{block.id}">{formatted}</p>')

            elif block.type == ElementType.FIGURE:
                if in_references:
                    body_html_parts.append('</ol></section>')
                    in_references = False

                # Extract figure details
                rel_path = block.metadata.get("relative_path", "")
                fig_num = block.metadata.get("figure_num", 1)
                caption = block.metadata.get("translated_caption") or block.metadata.get("caption", "")
                vision_desc = block.metadata.get("vision_description")

                formatted_caption = self.format_bidi_text(caption) if caption else ""
                vision_card_html = ""
                if vision_desc:
                    formatted_vision = self.format_bidi_text(vision_desc)
                    vision_card_html = (
                        f'<div class="vision-card">'
                        f'  <span class="vision-badge">تحلیل هوشمند تصویر (AI Vision):</span>'
                        f'  <p class="vision-text">{formatted_vision}</p>'
                        f'</div>'
                    )

                body_html_parts.append(
                    f'<figure class="article-figure" id="{block.id}">'
                    f'  <div class="figure-image-wrapper">'
                    f'    <img src="{rel_path}" alt="شکل {fig_num}" class="figure-img" loading="lazy" />'
                    f'  </div>'
                    f'  {f"<figcaption class=\"figure-caption\">{formatted_caption}</figcaption>" if formatted_caption else ""}'
                    f'  {vision_card_html}'
                    f'</figure>'
                )

            elif block.type == ElementType.FIGURE_CAPTION:
                # If caption was not already embedded with the figure, render it
                if not block.metadata.get("figure_id"):
                    formatted = self.format_bidi_text(content)
                    body_html_parts.append(f'<p class="figure-caption standalone-caption" id="{block.id}">{formatted}</p>')

            elif block.type == ElementType.REFERENCE:
                formatted = self.format_bidi_text(content)
                ref_number = block.metadata.get("ref_number") or ref_counter
                # The number is emitted here, never taken from the translated
                # text, so it cannot be dropped or displaced by bidi reordering.
                # Trailing space so copy/paste yields "[1] Andreasen ...", not
                # "[1]Andreasen ...".
                number_html = f'<span class="ref-number" dir="ltr">[{ref_number}]</span> '
                ref_counter = ref_number + 1
                if in_references:
                    body_html_parts.append(
                        f'<li class="reference-item" id="{block.id}">{number_html}{formatted}</li>'
                    )
                else:
                    body_html_parts.append(
                        f'<section class="references-section"><ol class="references-list">'
                        f'<li class="reference-item" id="{block.id}">{number_html}{formatted}</li>'
                    )
                    in_references = True

            elif block.type == ElementType.TABLE:
                formatted = self.format_bidi_text(content)
                body_html_parts.append(
                    f'<div class="table-container" id="{block.id}">'
                    f'  <pre class="table-content">{formatted}</pre>'
                    f'</div>'
                )

            elif block.type == ElementType.OTHER:
                # Skip minor metadata or render cleanly
                pass

        if in_references:
            body_html_parts.append('</ol></section>')

        return "\n".join(body_html_parts)

    def _generate_html_template(
        self,
        title: str,
        original_title: str,
        body_content: str,
        metadata: dict
    ) -> str:
        """Constructs the full HTML document with embedded CSS."""
        escaped_title = html.escape(title)
        escaped_orig = html.escape(original_title) if original_title else ""

        total_pages = metadata.get("total_pages", "")

        # Serialise glossary notes defensively: JSON is not a subset of JS
        # string literals, so "</script>" and U+2028/U+2029 would break out of
        # the inline script block and must be escaped.
        notes = {
            term.lower(): note
            for term, note in (metadata.get("glossary_notes") or {}).items()
        }
        term_notes_json = (
            json.dumps(notes, ensure_ascii=False)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )

        return f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escaped_title}</title>
  <!-- Google Fonts: Vazirmatn -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --primary-color: #1e3a8a;
      --primary-light: #eff6ff;
      --secondary-color: #0d9488;
      --text-color: #1f2937;
      --text-muted: #4b5563;
      --bg-color: #f8fafc;
      --card-bg: #ffffff;
      --border-color: #e2e8f0;
      --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
      --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.08), 0 2px 4px -1px rgba(0,0,0,0.04);
      --radius: 8px;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background-color: var(--bg-color);
      color: var(--text-color);
      line-height: 1.85;
      font-size: 16px;
      direction: rtl;
      text-align: right;
      padding: 2rem 1rem;
    }}

    .article-container {{
      max-width: 900px;
      margin: 0 auto;
      background-color: var(--card-bg);
      border-radius: var(--radius);
      box-shadow: var(--shadow-md);
      padding: 2.5rem 3rem;
      border: 1px solid var(--border-color);
    }}

    /* Header & Title */
    .article-header {{
      border-bottom: 2px solid var(--primary-light);
      padding-bottom: 1.75rem;
      margin-bottom: 2rem;
    }}

    .article-badge {{
      display: inline-block;
      background-color: var(--primary-light);
      color: var(--primary-color);
      font-size: 0.85rem;
      font-weight: 600;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      margin-bottom: 1rem;
    }}

    .article-title {{
      font-size: 1.85rem;
      font-weight: 800;
      color: var(--primary-color);
      line-height: 1.45;
      margin-bottom: 0.75rem;
    }}

    .article-original-title {{
      font-size: 1.05rem;
      font-weight: 400;
      color: var(--text-muted);
      direction: ltr;
      text-align: left;
      margin-top: 0.5rem;
      font-style: italic;
    }}

    /* Typography & Headings */
    .section-heading {{
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--primary-color);
      margin-top: 2.25rem;
      margin-bottom: 1rem;
      padding-bottom: 0.4rem;
      border-bottom: 2px solid var(--border-color);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .article-paragraph {{
      margin-bottom: 1.25rem;
      text-align: justify;
      color: var(--text-color);
      font-size: 1.05rem;
      word-spacing: 0.05em;
    }}

    /* English Parenthetical Terms */
    .en-term {{
      direction: ltr;
      display: inline-block;
      font-family: inherit;
      color: #334155;
      font-weight: 500;
      padding: 0 0.15rem;
    }}

    /* Clickable glossary terms */
    .glossary-term {{
      cursor: pointer;
      border-bottom: 1px dotted #0d9488;
      color: inherit;
      transition: background-color 0.15s ease;
    }}
    .glossary-term:hover,
    .glossary-term:focus {{
      background-color: #ccfbf1;
      outline: none;
    }}
    .glossary-term.active {{
      background-color: #99f6e4;
      border-bottom-color: #0f766e;
    }}
    .glossary-term .term-en {{
      color: #0f766e;
      font-weight: 500;
    }}

    /* Term explanation panel */
    .term-panel {{
      position: fixed;
      z-index: 60;
      max-width: 22rem;
      background-color: #ffffff;
      border: 1px solid var(--border-color);
      border-radius: var(--radius);
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
      padding: 0.9rem 1.05rem;
      display: none;
      direction: rtl;
      text-align: right;
    }}
    .term-panel.visible {{ display: block; }}
    .term-panel .term-panel-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.6rem;
      margin-bottom: 0.45rem;
    }}
    .term-panel .term-panel-title {{
      font-weight: 700;
      color: var(--primary-color);
      direction: ltr;
      text-align: left;
    }}
    .term-panel .term-panel-close {{
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-size: 1.1rem;
      line-height: 1;
      cursor: pointer;
      padding: 0 0.25rem;
    }}
    .term-panel .term-panel-body {{
      font-size: 0.92rem;
      line-height: 1.75;
      color: var(--text-color);
    }}
    .term-panel .term-panel-loading {{ color: var(--text-muted); }}

    /* Figures & Images */
    .article-figure {{
      margin: 2rem auto;
      text-align: center;
      background-color: #ffffff;
      border: 1px solid var(--border-color);
      border-radius: var(--radius);
      padding: 1.25rem;
      box-shadow: var(--shadow-sm);
      max-width: 100%;
    }}

    .figure-image-wrapper {{
      display: flex;
      justify-content: center;
      align-items: center;
      background-color: #fafafa;
      border-radius: calc(var(--radius) - 2px);
      padding: 0.5rem;
      overflow: hidden;
    }}

    .figure-img {{
      max-width: 100%;
      height: auto;
      border-radius: 4px;
      box-shadow: 0 2px 5px rgba(0,0,0,0.05);
      object-fit: contain;
    }}

    .figure-caption {{
      margin-top: 1rem;
      font-size: 0.95rem;
      font-weight: 500;
      color: var(--text-muted);
      line-height: 1.6;
      text-align: justify;
      padding: 0 0.5rem;
    }}

    .standalone-caption {{
      background: var(--primary-light);
      padding: 0.75rem 1rem;
      border-radius: var(--radius);
      border-right: 4px solid var(--primary-color);
      margin: 1.5rem 0;
    }}

    /* AI Vision Analysis Card */
    .vision-card {{
      margin-top: 1rem;
      background-color: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-radius: calc(var(--radius) - 2px);
      padding: 0.85rem 1rem;
      text-align: right;
    }}

    .vision-badge {{
      display: inline-block;
      font-size: 0.8rem;
      font-weight: 700;
      color: #166534;
      margin-bottom: 0.35rem;
    }}

    .vision-text {{
      font-size: 0.9rem;
      color: #14532d;
      line-height: 1.6;
    }}

    /* References */
    .references-section {{
      margin-top: 2rem;
      background-color: #f8fafc;
      border-radius: var(--radius);
      padding: 1.25rem 1.5rem;
      border: 1px solid var(--border-color);
    }}

    /* Numbering is rendered explicitly (.ref-number), so the list marker is
       suppressed to avoid showing "1. [1] ...". The list follows the document's
       RTL base direction — the entries are Persian, and an LTR base would make
       neutral punctuation at run boundaries jump to the wrong side. */
    .references-list {{
      list-style: none;
      margin: 0;
      padding: 0;
      direction: rtl;
      text-align: right;
    }}

    .ref-number {{
      display: inline-block;
      margin-left: 0.45rem;
      color: var(--primary-color);
      font-weight: 600;
      unicode-bidi: isolate;
    }}

    .reference-item {{
      font-size: 0.88rem;
      color: var(--text-muted);
      margin-bottom: 0.65rem;
      line-height: 1.5;
    }}

    /* Tables */
    .table-container {{
      overflow-x: auto;
      margin: 1.5rem 0;
      border: 1px solid var(--border-color);
      border-radius: var(--radius);
      background-color: #ffffff;
      padding: 1rem;
    }}

    .table-content {{
      font-family: monospace;
      font-size: 0.9rem;
      white-space: pre-wrap;
    }}

    /* Responsive */
    @media (max-width: 768px) {{
      body {{
        padding: 0.75rem 0.5rem;
        font-size: 15px;
      }}
      .article-container {{
        padding: 1.5rem 1.25rem;
      }}
      .article-title {{
        font-size: 1.45rem;
      }}
      .section-heading {{
        font-size: 1.2rem;
      }}
    }}

    /* Print Stylesheet */
    @media print {{
      body {{
        background: none;
        color: #000;
        padding: 0;
      }}
      .article-container {{
        box-shadow: none;
        border: none;
        padding: 0;
      }}
      .article-figure {{
        page-break-inside: avoid;
        box-shadow: none;
      }}
      .section-heading {{
        page-break-after: avoid;
      }}
    }}
  </style>
</head>
<body>
  <div class="article-container">
    <header class="article-header">
      <div class="article-badge">گزارش مورد بالینی (Medical Case Report)</div>
      <h1 class="article-title">{escaped_title}</h1>
      {f'<p class="article-original-title">{escaped_orig}</p>' if escaped_orig else ""}
    </header>

    <main class="article-body">
{body_content}
    </main>
  </div>

  <div class="term-panel" id="term-panel" role="dialog" aria-live="polite">
    <div class="term-panel-head">
      <span class="term-panel-title" id="term-panel-title"></span>
      <button type="button" class="term-panel-close" id="term-panel-close" aria-label="بستن">×</button>
    </div>
    <div class="term-panel-body" id="term-panel-body"></div>
  </div>

  <script>
  (function () {{
    "use strict";

    var panel = document.getElementById("term-panel");
    var panelTitle = document.getElementById("term-panel-title");
    var panelBody = document.getElementById("term-panel-body");
    var panelClose = document.getElementById("term-panel-close");
    var activeTerm = null;

    // Explanations are authored offline and injected here. Keys are lowercase
    // English terms; a missing key falls back to a generic definition instead
    // of showing nothing.
    var TERM_NOTES = {term_notes_json};

    var FALLBACK_NOTE = "توضیح این اصطلاح در واژه‌نامهٔ این سند ثبت نشده است. " +
      "برای تعریف دقیق، به منابع تخصصی دندان‌پزشکی مراجعه کنید.";

    // Mirror of glossary.normalise_key: lower-case, collapse whitespace, drop a
    // trailing plural "s". Keeps lookups matching when the model writes
    // "crown-root fractures" but the note is authored as "crown-root fracture".
    function noteKey(term) {{
      var k = (term || "").trim().toLowerCase().replace(/\\s+/g, " ");
      if (k.length > 3 && k.charAt(k.length - 1) === "s" &&
          !/(ss|is|us)$/.test(k)) {{ k = k.slice(0, -1); }}
      return k;
    }}

    function lookupNote(term) {{
      var key = noteKey(term);
      if (Object.prototype.hasOwnProperty.call(TERM_NOTES, key)) {{
        return TERM_NOTES[key];
      }}
      return null;
    }}

    function positionPanel(anchor) {{
      var rect = anchor.getBoundingClientRect();
      var panelRect = panel.getBoundingClientRect();
      var top = rect.bottom + window.scrollY + 8;
      var left = rect.left + window.scrollX;

      // Keep the panel inside the viewport horizontally.
      var maxLeft = window.scrollX + document.documentElement.clientWidth - panelRect.width - 12;
      if (left > maxLeft) {{ left = Math.max(window.scrollX + 12, maxLeft); }}
      if (left < window.scrollX + 12) {{ left = window.scrollX + 12; }}

      panel.style.top = top + "px";
      panel.style.left = left + "px";
    }}

    function hidePanel() {{
      panel.classList.remove("visible");
      if (activeTerm) {{ activeTerm.classList.remove("active"); }}
      activeTerm = null;
    }}

    function showTerm(termEl) {{
      var term = termEl.getAttribute("data-term") || "";
      var note = lookupNote(term) || FALLBACK_NOTE;

      panelTitle.textContent = term;
      panelBody.textContent = note;
      panel.classList.add("visible");

      if (activeTerm && activeTerm !== termEl) {{ activeTerm.classList.remove("active"); }}
      termEl.classList.add("active");
      activeTerm = termEl;

      positionPanel(termEl);
    }}

    document.addEventListener("click", function (event) {{
      var termEl = event.target.closest(".glossary-term");
      if (termEl) {{
        event.preventDefault();
        if (activeTerm === termEl) {{ hidePanel(); }} else {{ showTerm(termEl); }}
        return;
      }}
      if (!panel.contains(event.target)) {{ hidePanel(); }}
    }});

    document.addEventListener("keydown", function (event) {{
      if (event.key === "Escape") {{ hidePanel(); return; }}
      var termEl = event.target.closest && event.target.closest(".glossary-term");
      if (termEl && (event.key === "Enter" || event.key === " ")) {{
        event.preventDefault();
        showTerm(termEl);
      }}
    }});

    panelClose.addEventListener("click", hidePanel);
    window.addEventListener("resize", function () {{ if (activeTerm) {{ positionPanel(activeTerm); }} }});
  }})();
  </script>
</body>
</html>"""
