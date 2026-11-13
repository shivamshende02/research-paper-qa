
"""Document upload API."""

import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.core.config import PROJECT_ROOT
from src.db.session import get_db
from src.models.document import Document
from src.workers.tasks import process_document

import json

from fastapi import Body

from src.schemas.summary import SectionSummaryRequest
from src.services.manual_summary_ingestion import (
    get_manual_summary_service,
)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
MAX_FILE_SIZE = 50 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Validate, store, and register a research paper PDF."""

    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed",
        )

    # Verify the actual PDF signature.
    header = await file.read(5)

    if header != b"%PDF-":
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file",
        )

    await file.seek(0)

    document_id = uuid.uuid4()
    original_filename = Path(file.filename).name[:255]

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    destination = UPLOAD_DIR / f"{document_id}.pdf"
    temporary_path = UPLOAD_DIR / f"{document_id}.part"

    file_size = 0
    db_committed = False

    try:
        # Save PDF using chunked asynchronous writes.
        async with aiofiles.open(temporary_path, "wb") as output:
            while chunk := await file.read(CHUNK_SIZE):
                file_size += len(chunk)

                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 50 MB upload limit",
                    )

                await output.write(chunk)

        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty PDF file",
            )

        # Atomically move the completed upload into permanent storage.
        temporary_path.replace(destination)

        document = Document(
            id=document_id,
            filename=original_filename,
            storage_path=str(destination),
            file_size=file_size,
            status="queued",
        )

        # Persist document metadata.
        db.add(document)
        db.commit()
        db_committed = True

        db.refresh(document)

        # Submit asynchronous processing task.
        try:
            process_document.delay(str(document.id))

        except Exception as exc:
            document.status = "failed"
            document.error_message = (
                f"Task submission failed: {str(exc)[:1000]}"
            )
            db.commit()

            raise HTTPException(
                status_code=503,
                detail=(
                    "Document saved, but processing could not be queued"
                ),
            ) from exc

        return {
            "document_id": str(document.id),
            "filename": document.filename,
            "file_size": document.file_size,
            "status": document.status,
            "message": "Document uploaded successfully",
        }

    except HTTPException:
        db.rollback()

        if not db_committed:
            temporary_path.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)

        raise

    except Exception:
        db.rollback()

        if not db_committed:
            temporary_path.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)

        raise

    finally:
        await file.close()


@router.get("/{document_id}/status")
def get_document_status(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Return the current processing status of a document."""

    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "status": document.status,
        "error_message": document.error_message,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }
@router.post("/{document_id}/sections/{section_id}/summary")
def submit_section_summary(
    document_id: uuid.UUID,
    section_id: str,
    request: SectionSummaryRequest,
    db: Session = Depends(get_db),
):
    """Store a manually generated section summary in Qdrant."""

    # 1. Verify document exists.
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    # 2. Locate extracted JSON.
    processed_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / f"{document_id}.json"
    )

    if not processed_path.is_file():
        raise HTTPException(
            status_code=409,
            detail="Document extraction is not available yet",
        )

    # 3. Read extracted sections.
    try:
        with processed_path.open("r", encoding="utf-8") as file:
            extracted_document = json.load(file)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail="Could not read extracted document data",
        )

    sections = extracted_document.get("sections")

    if not isinstance(sections, list):
        raise HTTPException(
            status_code=500,
            detail="Invalid extracted document structure",
        )

    # 4. Match the section using the existing ingestion ID scheme.
    matching_section = None

    for index, section in enumerate(sections):
        if str(index) == section_id:
            matching_section = section
            break

    if matching_section is None:
        raise HTTPException(
            status_code=404,
            detail="Section not found",
        )

    header = str(
        matching_section.get("header") or "Untitled section"
    )

    metadata = matching_section.get("metadata") or {}
    page = metadata.get("page")

    # 5. Store summary in Qdrant.
    try:
        service = get_manual_summary_service()

        result = service.store_summary(
            document_id=str(document_id),
            section_id=section_id,
            header=header,
            page=page,
            summary=request.summary.strip(),
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Summary ingestion failed: {str(exc)}",
        ) from exc
    
@router.post("/{document_id}/sections/{section_id}/content")
def submit_section_content(
        document_id: uuid.UUID,
        section_id: str,
        db: Session = Depends(get_db),
    ):
        document = db.get(Document, document_id)

        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        processed_path = (
            PROJECT_ROOT / "data" / "processed" / f"{document_id}.json"
        )

        if not processed_path.is_file():
            raise HTTPException(
                status_code=409,
                detail="Processed document content not found",
            )

        try:
            with processed_path.open("r", encoding="utf-8") as file:
                extracted_document = json.load(file)
        except (OSError, json.JSONDecodeError):
            raise HTTPException(
                status_code=500,
                detail="Could not read processed document",
            )

        sections = extracted_document.get("sections")

        if not isinstance(sections, list):
            raise HTTPException(
                status_code=500,
                detail="Invalid extracted document structure",
            )

        matching_section = None

        for index, section in enumerate(sections):
            if str(index) == section_id:
                matching_section = section
                break

        if matching_section is None:
            raise HTTPException(
                status_code=404,
                detail="Section not found",
            )

        header = str(matching_section.get("header") or "Untitled section")
        content = str(matching_section.get("content") or "").strip()

        metadata = matching_section.get("metadata") or {}
        page = metadata.get("page")

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Section has no content to ingest",
            )

        try:
            service = get_manual_summary_service()

            result = service.store_content(
                document_id=str(document_id),
                section_id=section_id,
                header=header,
                page=page,
                content=content,
            )

            return result

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Content ingestion failed: {str(exc)}",
            )