"""
Figure processor for extracting, preserving, and linking images and captions from PDFs.
Preserves original images without modification or alteration.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf

from .models import BoundingBox, ContentBlock, ElementType

logger = logging.getLogger(__name__)


# Formats a browser can display directly. Anything else (jpx/jp2, jbig2, cmyk
# tiffs, ...) is converted to PNG, because an <img> pointing at an unreadable
# format renders as a broken icon and silently loses the figure.
WEB_SAFE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"})


class FigureProcessor:
    """
    Extracts embedded images from PDF pages, saves original image files to an assets folder,
    and identifies / links figure captions.
    """

    CAPTION_REGEX = re.compile(
        r'^\s*(?:Figure|Fig\.|Fig|Image|شکل)\s*(\d+)[\.:\s\-]',
        re.IGNORECASE
    )

    # A figure caption can also be written as "FIGURE 1" in small caps, or
    # carry a panel letter ("Figure 1A").
    CAPTION_NUMBER_REGEX = re.compile(
        r'\b(?:Figure|Fig\.?|Image|شکل)\s*(\d+)', re.IGNORECASE
    )

    def __init__(self, assets_dir: Path, min_dimension: int = 40):
        """
        :param assets_dir: Directory where extracted images will be stored (e.g. output/assets).
        :param min_dimension: Minimum width and height in points to filter out icons / divider lines.
        """
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.min_dimension = min_dimension
        self._figure_counter = 0

    # -- housekeeping ------------------------------------------------------

    def reset_assets(self) -> int:
        """
        Deletes previously extracted figures so a run never mixes images from
        an earlier document into the new output.

        Only files matching the extractor's own naming scheme are touched, so a
        user's unrelated files in the same folder are left alone.
        """
        removed = 0
        if not self.assets_dir.exists():
            return 0
        for path in self.assets_dir.glob("figure-*"):
            if not path.is_file():
                continue
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Could not remove stale asset %s: %s", path, exc)
        self._figure_counter = 0
        return removed

    # -- extraction --------------------------------------------------------

    def extract_figures_from_page(
        self,
        doc: pymupdf.Document,
        page: pymupdf.Page,
        page_num: int
    ) -> List[ContentBlock]:
        """
        Extracts image blocks from a PDF page and saves original image files to assets_dir.
        Returns a list of ContentBlock items of type FIGURE with their exact bboxes.
        """
        figure_blocks: List[ContentBlock] = []

        try:
            image_infos = page.get_image_info(xrefs=True)
        except Exception:
            image_infos = []

        seen_rects = []

        for info in image_infos:
            bbox_coords = info.get("bbox")
            if not bbox_coords:
                continue

            x0, y0, x1, y1 = bbox_coords
            width = x1 - x0
            height = y1 - y0

            # Filter tiny decorative images, icons, or horizontal/vertical line artifacts
            if width < self.min_dimension or height < self.min_dimension:
                continue

            # Check if this rect overlaps heavily with an already processed image on this page
            rect = pymupdf.Rect(x0, y0, x1, y1)
            is_duplicate = False
            for prev_rect in seen_rects:
                intersection = rect & prev_rect
                if not intersection.is_empty and intersection.get_area() > 0.8 * min(rect.get_area(), prev_rect.get_area()):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            xref = info.get("xref", 0)
            if xref == 0:
                continue

            seen_rects.append(rect)
            self._figure_counter += 1
            figure_id = f"figure_{self._figure_counter}"

            try:
                saved = self._save_image(doc, xref, self._figure_counter)
            except Exception as exc:  # noqa: BLE001 - one bad stream must not stop parsing
                logger.warning("Could not extract image xref %s on page %s: %s", xref, page_num, exc)
                self._figure_counter -= 1
                continue

            if not saved:
                self._figure_counter -= 1
                continue

            image_path, rel_path = saved

            block = ContentBlock(
                id=figure_id,
                type=ElementType.FIGURE,
                content=f"[Figure {self._figure_counter}]",
                bbox=BoundingBox(x0, y0, x1, y1),
                page_number=page_num,
                metadata={
                    "figure_id": figure_id,
                    "figure_num": self._figure_counter,
                    "image_path": str(image_path),
                    "relative_path": rel_path,
                    "width": width,
                    "height": height,
                    "xref": xref,
                    "caption": "",
                    "caption_id": None,
                }
            )
            figure_blocks.append(block)

        return figure_blocks

    def _save_image(self, doc: pymupdf.Document, xref: int, number: int):
        """
        Saves an image stream to disk in a browser-readable format.

        PyMuPDF's extract_image() hands back the raw stream, which may be a
        format no browser can render (JPEG2000, JBIG2) or a CMYK pixmap that
        decodes to inverted colours. Those are re-encoded to RGB PNG.
        """
        base_img = doc.extract_image(xref)
        if not base_img:
            return None

        image_bytes = base_img.get("image")
        image_ext = (base_img.get("ext") or "png").lower()
        if not image_bytes:
            return None

        colorspace = base_img.get("colorspace")
        needs_conversion = image_ext not in WEB_SAFE_EXTENSIONS
        # colorspace 4 is CMYK in MuPDF's numbering; browsers misrender it.
        if colorspace == 4:
            needs_conversion = True
        if base_img.get("smask") and image_ext not in ("png",):
            needs_conversion = True

        if needs_conversion:
            try:
                pixmap = pymupdf.Pixmap(doc, xref)
                if pixmap.colorspace is None or pixmap.colorspace.n > 3:
                    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
                image_bytes = pixmap.tobytes("png")
                image_ext = "png"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falling back to the raw stream for xref %s (%s).", xref, exc)

        filename = f"figure-{number}.{image_ext}"
        image_path = self.assets_dir / filename
        with open(image_path, "wb") as img_file:
            img_file.write(image_bytes)

        return image_path, f"assets/{filename}"

    # -- composite figures -------------------------------------------------

    def group_adjacent_figures(self, document, tolerance: float = 6.0) -> int:
        """
        Merges image blocks that physically touch into one composite figure.

        A single printed figure is usually exported as several image XObjects
        (one per panel). Extracted naively they become independent FIGURE blocks
        with independent numbering, and only whichever one happens to sit next
        to the caption ever shows one. Panels of the same figure always touch or
        overlap, so geometric adjacency is a reliable grouping signal.

        Returns the number of blocks that were absorbed.
        """
        absorbed = 0

        for page in document.pages:
            figures = [b for b in page.blocks if b.type == ElementType.FIGURE]
            if len(figures) < 2:
                continue

            parent = {f.id: f.id for f in figures}

            def find(node: str) -> str:
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node

            def union(left: str, right: str) -> None:
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parent[root_right] = root_left

            for i in range(len(figures)):
                for j in range(i + 1, len(figures)):
                    if self._rects_touch(figures[i].bbox, figures[j].bbox, tolerance):
                        union(figures[i].id, figures[j].id)

            groups: Dict[str, List[ContentBlock]] = {}
            for figure in figures:
                groups.setdefault(find(figure.id), []).append(figure)

            for members in groups.values():
                if len(members) < 2:
                    continue
                members.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
                primary = members[0]
                sub_images = primary.metadata.setdefault("sub_images", [])
                x0, y0, x1, y1 = primary.bbox.x0, primary.bbox.y0, primary.bbox.x1, primary.bbox.y1

                for extra in members[1:]:
                    rel = extra.metadata.get("relative_path")
                    if rel and rel not in sub_images:
                        sub_images.append(rel)
                    extra.metadata["merged_into"] = primary.id
                    absorbed += 1
                    x0 = min(x0, extra.bbox.x0)
                    y0 = min(y0, extra.bbox.y0)
                    x1 = max(x1, extra.bbox.x1)
                    y1 = max(y1, extra.bbox.y1)

                primary.bbox = BoundingBox(x0, y0, x1, y1)

        for page in document.pages:
            page.blocks = [
                b for b in page.blocks
                if not (b.type == ElementType.FIGURE and b.metadata.get("merged_into"))
            ]

        return absorbed

    @staticmethod
    def _rects_touch(a: BoundingBox, b: BoundingBox, tolerance: float) -> bool:
        """True when two boxes touch or overlap within `tolerance` points."""
        gap_x = max(a.x0 - b.x1, b.x0 - a.x1)
        gap_y = max(a.y0 - b.y1, b.y0 - a.y1)
        return gap_x <= tolerance and gap_y <= tolerance

    # -- caption linking ---------------------------------------------------

    def link_captions_to_figures(
        self,
        figure_blocks: List[ContentBlock],
        text_blocks: List[ContentBlock],
        page_height: float = 800.0,
    ) -> None:
        """
        Detects caption blocks and associates them with their corresponding figures.
        Updates the metadata of both blocks.
        """
        unmatched_figures = list(figure_blocks)

        for text_block in text_blocks:
            match = self.CAPTION_REGEX.search(text_block.content)
            if not match:
                continue

            text_block.type = ElementType.FIGURE_CAPTION
            detected_num = int(match.group(1))
            text_block.metadata["caption_num"] = detected_num

            matched_fig = None
            # 1. Explicit number match on the same page.
            for fig in unmatched_figures:
                if fig.page_number == text_block.page_number and fig.metadata.get("figure_num") == detected_num:
                    matched_fig = fig
                    break

            # 2. Spatial proximity on the same page.
            if not matched_fig:
                same_page_figs = [f for f in unmatched_figures if f.page_number == text_block.page_number]
                if same_page_figs:
                    same_page_figs.sort(
                        key=lambda f: abs(f.bbox.center_y - text_block.bbox.center_y)
                    )
                    matched_fig = same_page_figs[0]

            if not matched_fig:
                continue

            matched_fig.metadata["caption"] = text_block.content
            matched_fig.metadata["caption_id"] = text_block.id
            matched_fig.metadata["caption_num"] = detected_num
            text_block.metadata["figure_id"] = matched_fig.id
            if matched_fig in unmatched_figures:
                unmatched_figures.remove(matched_fig)

            # 3. Absorb adjacent uncaptioned panels into this figure. A
            #    multi-panel figure (A/B/C) arrives as several separate image
            #    blocks; without this they become independent FIGURE blocks and
            #    only the first one ever shows a caption.
            self._absorb_panels(matched_fig, unmatched_figures, page_height)

    def _absorb_panels(
        self,
        primary: ContentBlock,
        unmatched: List[ContentBlock],
        page_height: float,
    ) -> None:
        """Merges image blocks that are clearly panels of the same figure."""
        sub_images = primary.metadata.setdefault("sub_images", [])
        vertical_window = max(120.0, page_height * 0.35)

        # The merged figure must span every panel, otherwise reading-order
        # placement can drop the group between the panels it contains.
        x0, y0, x1, y1 = primary.bbox.x0, primary.bbox.y0, primary.bbox.x1, primary.bbox.y1

        for candidate in list(unmatched):
            if candidate.page_number != primary.page_number:
                continue

            # Horizontal overlap relative to the narrower of the two.
            overlap = min(primary.bbox.x1, candidate.bbox.x1) - max(primary.bbox.x0, candidate.bbox.x0)
            min_width = min(primary.bbox.width, candidate.bbox.width) or 1.0
            if overlap / min_width < 0.4:
                continue

            # Vertically close to the figure, in either direction.
            gap = min(
                abs(candidate.bbox.y0 - primary.bbox.y1),
                abs(primary.bbox.y0 - candidate.bbox.y1),
            )
            if gap > vertical_window:
                continue

            rel = candidate.metadata.get("relative_path")
            if rel:
                sub_images.append(rel)
            candidate.metadata["merged_into"] = primary.id
            unmatched.remove(candidate)

            x0 = min(x0, candidate.bbox.x0)
            y0 = min(y0, candidate.bbox.y0)
            x1 = max(x1, candidate.bbox.x1)
            y1 = max(y1, candidate.bbox.y1)

        if sub_images:
            primary.bbox = BoundingBox(x0, y0, x1, y1)
            primary.metadata["sub_images"] = sub_images
