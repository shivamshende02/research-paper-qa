
"""
Coordinate-based PDF text extraction for research papers.

Uses PyMuPDF block coordinates to reconstruct two-column reading order.
Produces Markdown sections mapped to a document UUID.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import fitz


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COLUMN_SEARCH_MIN = 0.30
COLUMN_SEARCH_MAX = 0.70
MIN_COLUMN_BLOCKS = 2
FULL_WIDTH_THRESHOLD = 0.75

HEADING_PATTERNS = [
    r"^\d+(\.\d+)*\.?\s+[A-Z]",
    r"^(ABSTRACT|INTRODUCTION|CONCLUSION|REFERENCES|"
    r"RELATED WORK|METHODOLOGY|EXPERIMENTS|RESULTS)$",
]


# --------------------------------------------------
# Block extraction
# --------------------------------------------------

def extract_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    """
    Extract text blocks and their coordinates from a PDF page.

    Each block contains:
        x0, y0, x1, y1, text, block_number
    """
    extracted = []

    for block in page.get_text("blocks", sort=False):
        x0, y0, x1, y1, text, block_number, block_type = block[:7]

        if block_type != 0 or not text.strip():
            continue

        extracted.append({
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "text": text.strip(),
            "block_number": block_number,
        })

    return extracted


# --------------------------------------------------
# Column detection
# --------------------------------------------------

def detect_column_boundary(
    blocks: list[dict[str, Any]],
    page_width: float,
) -> float | None:
    """
    Detect the vertical gutter separating two text columns.

    Candidate boundaries are generated from block edges.
    Blocks spanning most of the page are excluded from detection.

    Returns:
        The estimated horizontal column boundary, or None
        when the page does not appear to contain two columns.
    """
    min_x = page_width * COLUMN_SEARCH_MIN
    max_x = page_width * COLUMN_SEARCH_MAX

    column_blocks = [
        block
        for block in blocks
        if block["x1"] - block["x0"]
        < page_width * FULL_WIDTH_THRESHOLD
    ]

    if len(column_blocks) < MIN_COLUMN_BLOCKS * 2:
        return None

    edges = sorted({
        coordinate
        for block in column_blocks
        for coordinate in (block["x0"], block["x1"])
        if min_x <= coordinate <= max_x
    })

    if len(edges) < 2:
        return None

    best_boundary = None
    best_score = float("-inf")

    for left_edge, right_edge in zip(edges, edges[1:]):
        if right_edge <= left_edge:
            continue

        boundary = (left_edge + right_edge) / 2

        left_blocks = [
            block for block in column_blocks
            if block["x1"] <= boundary
        ]

        right_blocks = [
            block for block in column_blocks
            if block["x0"] >= boundary
        ]

        crossing_blocks = [
            block for block in column_blocks
            if block["x0"] < boundary < block["x1"]
        ]

        left_count = len(left_blocks)
        right_count = len(right_blocks)

        if (
            left_count < MIN_COLUMN_BLOCKS
            or right_count < MIN_COLUMN_BLOCKS
        ):
            continue

        # Reward balanced columns and wider gutters.
        balance_score = min(left_count, right_count)
        gap_score = (right_edge - left_edge) / page_width
        crossing_penalty = len(crossing_blocks)

        score = (
            balance_score * 2
            + gap_score * 10
            - crossing_penalty * 3
        )

        if score > best_score:
            best_score = score
            best_boundary = boundary

    return best_boundary


# --------------------------------------------------
# Reading-order reconstruction
# --------------------------------------------------

def sort_two_column_blocks(
    blocks: list[dict[str, Any]],
    boundary: float,
    page_width: float,
) -> list[dict[str, Any]]:
    """
    Sort blocks in natural two-column reading order.

    Full-width blocks divide the page into horizontal reading bands.
    Within each band, the left column is read top-to-bottom first,
    followed by the right column top-to-bottom.
    """
    spanning_blocks = []
    column_blocks = []

    for block in blocks:
        width = block["x1"] - block["x0"]

        crosses_boundary = (
            block["x0"] < boundary < block["x1"]
        )

        is_full_width = width >= (
            page_width * FULL_WIDTH_THRESHOLD
        )

        if crosses_boundary or is_full_width:
            spanning_blocks.append(block)
        else:
            column_blocks.append(block)

    spanning_blocks.sort(
        key=lambda block: (block["y0"], block["x0"])
    )

    ordered = []
    previous_y = float("-inf")

    # Each spanning block separates column-reading regions.
    for spanning in spanning_blocks:
        band_blocks = [
            block for block in column_blocks
            if previous_y <= block["y0"] < spanning["y0"]
        ]

        left = [
            block for block in band_blocks
            if (block["x0"] + block["x1"]) / 2 < boundary
        ]

        right = [
            block for block in band_blocks
            if (block["x0"] + block["x1"]) / 2 >= boundary
        ]

        left.sort(key=lambda block: (block["y0"], block["x0"]))
        right.sort(key=lambda block: (block["y0"], block["x0"]))

        ordered.extend(left)
        ordered.extend(right)
        ordered.append(spanning)

        previous_y = spanning["y1"]

    # Process remaining blocks below the last spanning block.
    remaining = [
        block for block in column_blocks
        if block["y0"] >= previous_y
    ]

    left = [
        block for block in remaining
        if (block["x0"] + block["x1"]) / 2 < boundary
    ]

    right = [
        block for block in remaining
        if (block["x0"] + block["x1"]) / 2 >= boundary
    ]

    left.sort(key=lambda block: (block["y0"], block["x0"]))
    right.sort(key=lambda block: (block["y0"], block["x0"]))

    ordered.extend(left)
    ordered.extend(right)

    return ordered


def sort_single_column_blocks(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort ordinary single-column text blocks top-to-bottom."""
    return sorted(
        blocks,
        key=lambda block: (block["y0"], block["x0"]),
    )


def reconstruct_reading_order(
    page: fitz.Page,
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Detect page layout and return blocks in reading order.

    Falls back to ordinary coordinate sorting for single-column pages.
    """
    page_width = page.rect.width

    boundary = detect_column_boundary(blocks, page_width)

    if boundary is None:
        return sort_single_column_blocks(blocks)

    return sort_two_column_blocks(
        blocks,
        boundary,
        page_width,
    )


# --------------------------------------------------
# Markdown generation
# --------------------------------------------------

def is_heading(text: str) -> bool:
    """Identify likely research-paper section headings."""
    cleaned = text.strip()

    if not cleaned or len(cleaned) > 120:
        return False

    return any(
        re.match(pattern, cleaned, re.IGNORECASE)
        for pattern in HEADING_PATTERNS
    )


def clean_text(text: str) -> str:
    """Normalize whitespace without removing mathematical symbols."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def build_markdown_sections(
    ordered_blocks: list[dict[str, Any]],
    page_number: int,
) -> list[dict[str, Any]]:
    """
    Convert ordered blocks into Markdown sections.

    Heading-like blocks start new sections. Other blocks are
    accumulated as section content.
    """
    sections = []

    current_header = f"Page {page_number}"
    current_content = []

    def flush() -> None:
        content = "\n\n".join(current_content).strip()

        if content:
            sections.append({
                "header": current_header,
                "content": content,
                "metadata": {
                    "type": "section",
                    "page": page_number,
                },
            })

    for block in ordered_blocks:
        text = clean_text(block["text"])

        if not text:
            continue

        if is_heading(text):
            flush()
            current_header = text
            current_content = []
        else:
            current_content.append(text)

    flush()

    return sections


# --------------------------------------------------
# Public extraction function
# --------------------------------------------------

def extract_research_paper(
    pdf_path: str | Path,
    document_id: str | None = None,
) -> dict[str, Any]:
    """
    Extract a research paper and reconstruct two-column reading order.

    Args:
        pdf_path: Path to the PDF document.
        document_id: Existing document UUID. Generated if omitted.

    Returns:
        Dictionary containing document UUID and Markdown sections.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document_id = document_id or str(uuid.uuid4())

    try:
        uuid.UUID(document_id)
    except ValueError as exc:
        raise ValueError("Invalid document UUID.") from exc

    sections = []

    with fitz.open(pdf_path) as document:
        if document.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        for page_index, page in enumerate(document):
            page_number = page_index + 1

            blocks = extract_blocks(page)

            ordered_blocks = reconstruct_reading_order(
                page,
                blocks,
            )

            page_sections = build_markdown_sections(
                ordered_blocks,
                page_number,
            )

            sections.extend(page_sections)

    return {
        "uuid": document_id,
        "status": "success",
        "sections": sections,
    }
