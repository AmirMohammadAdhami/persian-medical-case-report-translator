"""
Figure processor for extracting, preserving, and linking images and captions from PDFs.
Preserves original images without modification or alteration.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pymupdf
from PIL import Image

from .models import BoundingBox, ContentBlock, ElementType


class FigureProcessor:
    """
    Extracts embedded images from PDF pages, saves original image files to an assets folder,
    and identifies / links figure captions.
    """

    CAPTION_REGEX = re.compile(
        r'^\s*(?:Figure|Fig\.|Fig|Figure\s*\d+|Image|شکل)\s*(\d+)[\.:\s\-]',
        re.IGNORECASE
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

        seen_xrefs = set()
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

            # Extract the raw image bytes preserving original encoding
            try:
                base_img = doc.extract_image(xref)
                image_bytes = base_img.get("image")
                image_ext = base_img.get("ext", "png")

                if not image_bytes:
                    continue

                filename = f"figure-{self._figure_counter}.{image_ext}"
                image_path = self.assets_dir / filename

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                # Store relative path for HTML reference (assets/filename)
                rel_path = f"assets/{filename}"

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

            except Exception as e:
                # Corrupted or unextractable image stream; handle gracefully
                continue

        return figure_blocks

    def link_captions_to_figures(
        self,
        figure_blocks: List[ContentBlock],
        text_blocks: List[ContentBlock]
    ) -> None:
        """
        Detects caption blocks and associates them with their corresponding figures.
        Updates the metadata of both blocks.
        """
        # First check for explicit numbering match (e.g. Figure 1 matches figure_num 1)
        unmatched_figures = list(figure_blocks)

        for text_block in text_blocks:
            match = self.CAPTION_REGEX.search(text_block.content)
            if match:
                text_block.type = ElementType.FIGURE_CAPTION
                detected_num = int(match.group(1))
                text_block.metadata["caption_num"] = detected_num

                # Try to find figure with matching number or closest on same page
                matched_fig = None
                for fig in unmatched_figures:
                    if fig.page_number == text_block.page_number and fig.metadata.get("figure_num") == detected_num:
                        matched_fig = fig
                        break

                if not matched_fig:
                    # Match by spatial proximity on the same page
                    same_page_figs = [f for f in unmatched_figures if f.page_number == text_block.page_number]
                    if same_page_figs:
                        # Find closest figure (usually caption is below or right above)
                        same_page_figs.sort(
                            key=lambda f: abs(f.bbox.center_y - text_block.bbox.center_y)
                        )
                        matched_fig = same_page_figs[0]

                if matched_fig:
                    matched_fig.metadata["caption"] = text_block.content
                    matched_fig.metadata["caption_id"] = text_block.id
                    text_block.metadata["figure_id"] = matched_fig.id
                    if matched_fig in unmatched_figures:
                        unmatched_figures.remove(matched_fig)
