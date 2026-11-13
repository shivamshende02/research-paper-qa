
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MAX_FILE_SIZE = 10 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024

PDF_SIGNATURE = b"%PDF-"

logger = logging.getLogger(__name__)

# Marker model initialization is expensive.
# Initialize once and serialize conversions through one lock.
marker_converter: PdfConverter | None = None
marker_lock = asyncio.Lock()


# --------------------------------------------------
# Response schemas
# --------------------------------------------------

class SectionMetadata(BaseModel):
    """Metadata associated with an extracted section."""

    type: str
    page: int


class SectionNode(BaseModel):
    """A logical section extracted from a PDF."""

    header: str
    content: str
    metadata: SectionMetadata


class UploadResponse(BaseModel):
    """Successful document processing response."""

    uuid: str
    status: str
    sections: list[SectionNode] = Field(default_factory=list)


# --------------------------------------------------
# Marker initialization
# --------------------------------------------------

def initialize_marker() -> PdfConverter:
    """
    Initialize Marker with JSON output and default layout processors.

    Marker handles layout detection, reading order, mathematical
    expressions, and table reconstruction through its processors.

    Header and footer removal is enabled through Marker defaults.
    """
    config = {
        "output_format": "json",
        "keep_pageheader_in_output": False,
        "keep_pagefooter_in_output": False,
    }

    config_parser = ConfigParser(config)

    return PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the storage directory and initialize Marker at startup."""
    global marker_converter

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Model loading is blocking, so run it outside the event loop.
        marker_converter = await asyncio.to_thread(initialize_marker)

        logger.info("Marker initialized successfully.")

    except OSError:
        logger.exception("Unable to initialize document storage.")
        raise

    except Exception:
        logger.exception("Unable to initialize Marker.")
        raise

    yield

    marker_converter = None


app = FastAPI(
    title="Vaani Document Processing API",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# File handling
# --------------------------------------------------

async def save_pdf(file: UploadFile, destination: Path) -> int:
    """
    Save an uploaded PDF asynchronously with a strict size limit.

    Returns:
        int: Number of bytes written.

    Raises:
        HTTPException: If the upload exceeds the size limit.
    """
    total_size = 0
    first_chunk = True

    async with aiofiles.open(destination, "xb") as output_file:
        while chunk := await file.read(CHUNK_SIZE):
            if first_chunk:
                first_chunk = False

                if not chunk.startswith(PDF_SIGNATURE):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid PDF file signature.",
                    )

            total_size += len(chunk)

            if total_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail="PDF exceeds the 10 MB upload limit.",
                )

            await output_file.write(chunk)

    if total_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty.",
        )

    return total_size


# --------------------------------------------------
# Marker output normalization
# --------------------------------------------------

def to_dict(value: Any) -> Any:
    """
    Convert Marker/Pydantic objects into ordinary Python structures.

    Supports dictionaries, lists, Pydantic models, and nested objects.
    """
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_dict(item) for item in value]

    if hasattr(value, "model_dump"):
        return to_dict(value.model_dump())

    if hasattr(value, "__dict__"):
        return to_dict(vars(value))

    if hasattr(value, "value"):
        return value.value

    return value


def get_block_type(block: dict[str, Any]) -> str:
    """Return a normalized Marker block type."""
    block_type = block.get("block_type", "Text")

    if isinstance(block_type, dict):
        block_type = block_type.get("value", "Text")

    return str(block_type).split(".")[-1]


def get_block_text(block: dict[str, Any]) -> str:
    """
    Extract readable content from a Marker block.

    HTML is retained when available, which helps preserve table structure.
    """
    for key in ("text", "markdown", "html"):
        value = block.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def get_page_number(
    block: dict[str, Any],
    inherited_page: int,
) -> int:
    """Extract a one-based page number from Marker metadata."""
    metadata = block.get("metadata") or {}

    if not isinstance(metadata, dict):
        return inherited_page

    page_id = metadata.get("page_id", block.get("page_id"))

    if isinstance(page_id, int):
        return page_id + 1

    return inherited_page


# --------------------------------------------------
# Section extraction
# --------------------------------------------------

def extract_section_nodes(rendered_document: Any) -> list[SectionNode]:
    """
    Convert Marker JSON output into logical section nodes.

    SectionHeader blocks start new sections. Content blocks are
    accumulated until the next section header.

    Page numbers are retained from Marker block metadata.
    """
    document = to_dict(rendered_document)

    if not isinstance(document, dict):
        raise ValueError("Unexpected Marker JSON output.")

    sections: list[SectionNode] = []

    current_header = "Unsectioned"
    current_content: list[str] = []
    current_page = 1
    current_type = "Text"

    def flush_section() -> None:
        """Store the current section when it contains content."""
        nonlocal current_content

        content = "\n\n".join(
            part for part in current_content if part.strip()
        )

        if content.strip():
            sections.append(
                SectionNode(
                    header=current_header,
                    content=content,
                    metadata=SectionMetadata(
                        type=current_type,
                        page=current_page,
                    ),
                )
            )

        current_content = []

    def walk(
        node: Any,
        inherited_page: int = 1,
    ) -> None:
        """Recursively traverse Marker document blocks."""
        nonlocal current_header
        nonlocal current_page
        nonlocal current_type

        if isinstance(node, list):
            for item in node:
                walk(item, inherited_page)
            return

        if not isinstance(node, dict):
            return

        block_type = get_block_type(node)
        page_number = get_page_number(node, inherited_page)

        # Skip recurring artifacts if they appear in the output.
        if block_type in {"PageHeader", "PageFooter"}:
            return

        if block_type == "SectionHeader":
            flush_section()

            current_header = get_block_text(node) or "Untitled Section"
            current_page = page_number
            current_type = block_type

        elif block_type not in {"Document", "Page"}:
            text = get_block_text(node)

            if text:
                if not current_content:
                    current_page = page_number
                    current_type = block_type

                current_content.append(text)

        # Process child blocks recursively.
        children = node.get("children", [])

        if children:
            walk(children, page_number)

    walk(document)
    flush_section()

    return sections


# --------------------------------------------------
# Parsing service
# --------------------------------------------------

def parse_pdf_with_marker(pdf_path: Path) -> list[SectionNode]:
    """
    Run Marker synchronously and convert its output into section nodes.

    This function is called through asyncio.to_thread so model inference
    does not block FastAPI's event loop.
    """
    if marker_converter is None:
        raise RuntimeError("Marker converter is not initialized.")

    rendered = marker_converter(str(pdf_path))

    return extract_section_nodes(rendered)


async def process_pdf(pdf_path: Path) -> list[SectionNode]:
    """
    Process a PDF while keeping Marker conversions serialized.

    A single converter instance is shared. The lock prevents concurrent
    requests from invoking it simultaneously.
    """
    async with marker_lock:
        return await asyncio.to_thread(
            parse_pdf_with_marker,
            pdf_path,
        )


# --------------------------------------------------
# API endpoint
# --------------------------------------------------

@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=200,
)
async def upload_pdf(
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload and process a PDF using Marker.

    Returns:
        UploadResponse: UUID and extracted section nodes.
    """
    if file.content_type != "application/pdf":
        await file.close()

        raise HTTPException(
            status_code=400,
            detail="Only application/pdf files are accepted.",
        )

    document_id = str(uuid.uuid4())
    pdf_path = DATA_DIR / f"{document_id}.pdf"

    try:
        await save_pdf(file, pdf_path)

        sections = await process_pdf(pdf_path)

        if not sections:
            logger.warning(
                "No sections extracted for document %s",
                document_id,
            )

        return UploadResponse(
            uuid=document_id,
            status="success",
            sections=sections,
        )

    except HTTPException:
        raise

    except OSError:
        logger.exception(
            "File storage failure for document %s",
            document_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to store or access the uploaded PDF.",
        )

    except Exception:
        logger.exception(
            "Marker parsing failed for document %s",
            document_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process the uploaded PDF.",
        )

    finally:
        await file.close()

        # Remove raw PDF after extraction.
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError:
            logger.exception(
                "Failed to clean up temporary PDF: %s",
                document_id,
            )
