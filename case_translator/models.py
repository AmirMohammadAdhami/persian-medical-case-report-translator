"""
Document model data structures representing parsed medical case reports.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ElementType(str, Enum):
    TITLE = "title"
    AUTHOR = "author"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    REFERENCE = "reference"
    OTHER = "other"


@dataclass
class BoundingBox:
    """Bounding box coordinates (x0, y0, x1, y1) in PDF points."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> Dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_tuple(cls, coords: Tuple[float, float, float, float]) -> "BoundingBox":
        return cls(x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])


@dataclass
class ContentBlock:
    """A single logical element extracted from the PDF."""
    id: str
    type: ElementType
    content: str
    bbox: BoundingBox
    page_number: int  # 1-indexed
    metadata: Dict[str, Any] = field(default_factory=dict)
    translated_content: Optional[str] = None

    @property
    def is_translatable(self) -> bool:
        """Determines if this block contains text that should be translated."""
        if self.type in (ElementType.TITLE, ElementType.HEADING, ElementType.PARAGRAPH,
                         ElementType.FIGURE_CAPTION, ElementType.REFERENCE):
            return bool(self.content and self.content.strip())
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, ElementType) else str(self.type),
            "content": self.content,
            "translated_content": self.translated_content,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }


@dataclass
class Page:
    """A page in the document containing an ordered sequence of blocks."""
    page_number: int
    width: float
    height: float
    blocks: List[ContentBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "blocks": [b.to_dict() for b in self.blocks],
        }


@dataclass
class Document:
    """Internal document representation preserving logical reading order."""
    pages: List[Page] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_ordered_blocks(self) -> List[ContentBlock]:
        """Returns all blocks across all pages in strictly preserved reading order."""
        ordered = []
        for page in self.pages:
            ordered.extend(page.blocks)
        return ordered

    def get_figures(self) -> List[ContentBlock]:
        """Returns all figure blocks in the document."""
        return [b for b in self.get_ordered_blocks() if b.type == ElementType.FIGURE]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "pages": [p.to_dict() for p in self.pages],
        }
