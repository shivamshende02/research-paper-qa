
"""Background document processing tasks."""

import json
import uuid
from pathlib import Path

from src.db.session import SessionLocal
from src.models.document import Document
from src.pdf_extractor import extract_research_paper
from src.workers.celery_app import celery_app
from src.services.vector_ingestion import VectorIngestion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@celery_app.task(name="vaani.health_check")
def health_check():
    """Verify Celery worker execution."""
    return {
        "status": "success",
        "message": "Celery worker is processing tasks",
        "service": "vaani",
    }


@celery_app.task(
    name="vaani.process_document",
    bind=True,
)
def process_document(self, document_id: str):
    """Extract PDF content and update document processing status."""

    db = SessionLocal()

    try:
        document_uuid = uuid.UUID(document_id)

        document = db.get(Document, document_uuid)

        if document is None:
            raise ValueError(f"Document not found: {document_id}")

        if document.status == "completed":
            return {
                "status": "skipped",
                "reason": "Document already processed",
                "document_id": document_id,
            }

        document.status = "processing"
        document.error_message = None
        db.commit()

        result = extract_research_paper(
            pdf_path=document.storage_path,
            document_id=document_id,
        )

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        output_path = PROCESSED_DIR / f"{document_id}.json"
        temporary_path = PROCESSED_DIR / f"{document_id}.tmp"

        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)

        temporary_path.replace(output_path)

# Run vector ingestion after PDF extraction.
        vector_ingestion = VectorIngestion(
            chunk_size=500,
            overlap=50,
        )

        vector_result = vector_ingestion.ingest(result)

# Mark completed only after vector ingestion succeeds.
        document.status = "completed"
        db.commit()


        pages = {
            section["metadata"]["page"]
            for section in result["sections"]
        }

        return {
            "status": "completed",
            "document_id": document_id,
            "section_count": len(result["sections"]),
            "page_count": len(pages),
            "output_path": str(output_path),
            "vector_ingestion": vector_result,
        }

    except Exception as exc:
        db.rollback()

        try:
            document = db.get(Document, uuid.UUID(document_id))

            if document is not None:
                document.status = "failed"
                document.error_message = str(exc)[:4000]
                db.commit()
        except Exception:
            db.rollback()

        raise

    finally:
        db.close()
