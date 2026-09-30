#!/usr/bin/env python3
"""
PDF structure inspector.

Answers the question "will this PDF translate well?" before spending any API
calls. It reports what the parser can actually see: whether there is a text
layer, the body font size, how many columns each page uses, and how the blocks
get classified.

Usage:
    python inspect_pdf.py article.pdf [more.pdf ...]
"""

import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from case_translator.models import ElementType
from case_translator.pdf_parser import PDFParser


def _columns(page: pymupdf.Page) -> str:
    """Rough column count: how the text blocks cluster horizontally."""
    blocks = [b for b in page.get_text("dict").get("blocks", []) if b.get("type") == 0]
    if not blocks:
        return "no text"
    width = page.rect.width
    left = sum(1 for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 < width / 2)
    right = len(blocks) - left
    if left and right:
        return f"2-column ({left} left / {right} right)"
    return "1-column"


def inspect(path: Path) -> None:
    print("=" * 78)
    print(f"FILE: {path.name}")
    print("=" * 78)

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:  # noqa: BLE001
        print(f"  Cannot open: {exc}\n")
        return

    if len(doc) == 0:
        print("  Empty document.\n")
        return

    text_pages = sum(1 for page in doc if page.get_text().strip())
    print(f"  pages: {len(doc)}   pages with text: {text_pages}")
    if text_pages == 0:
        print("  !! No text layer. This is a scan — run OCR first, e.g.:")
        print("     ocrmypdf --language eng input.pdf output.pdf")
        doc.close()
        print()
        return

    sizes: Counter = Counter()
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if len(span.get("text", "").strip()) > 3:
                        sizes[round(span.get("size", 0), 1)] += len(span["text"].strip())

    print(f"  body font size (most common): {sizes.most_common(1)[0][0] if sizes else 'unknown'}")
    print(f"  font sizes seen: {sorted(sizes)[:8]}")

    for index, page in enumerate(doc):
        images = len(page.get_images(full=True))
        print(f"  page {index + 1}: {_columns(page)}   images: {images}")

    doc.close()

    # Now run the real parser and report the classification.
    with TemporaryDirectory() as tmp:
        parser = PDFParser(assets_dir=Path(tmp) / "assets")
        document = parser.parse(path)

    counts = Counter(b.type.value for page in document.pages for b in page.blocks)
    figures = document.get_figures()
    references = [b for page in document.pages for b in page.blocks
                  if b.type == ElementType.REFERENCE]
    in_runs = sum(len(page.metadata.get("ref_run_ids") or set()) for page in document.pages)

    print(f"\n  title: {document.metadata.get('title', '')[:80]!r}")
    print(f"  block types: {dict(counts)}")
    print(f"  figures: {len(figures)}  references: {len(references)} "
          f"(kept in English: {in_runs})")

    if counts.get("reference", 0) and in_runs == 0:
        print("  !! A reference list was detected but not marked as a bibliography run;")
        print("     check that the 'References' heading is on its own line.")
    if counts.get("reference", 0) == 0:
        print("  note: no reference list detected — the bibliography, if any, will be translated.")

    fragments = sum(
        1 for page in document.pages for b in page.blocks
        if (b.metadata.get("merged_fragments") or 1) > 1
    )
    print(f"  reflowed paragraphs (line-per-block input): {fragments}")
    print()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    for argument in sys.argv[1:]:
        path = Path(argument)
        if not path.exists():
            print(f"Not found: {path}\n")
            continue
        inspect(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
