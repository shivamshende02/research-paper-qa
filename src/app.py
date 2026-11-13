
import logging
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import BinaryIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 1024 * 1024  # 1 MB

ALLOWED_CONTENT_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        ".docx",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Upload API",
    description="Secure PDF and DOCX upload service",
    version="1.0.0",
)


# --------------------------------------------------
# Startup
# --------------------------------------------------

@app.on_event("startup")
async def create_upload_directory() -> None:
    """Create the upload directory if it does not exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Response Schema
# --------------------------------------------------

class UploadResponse(BaseModel):
    """Response returned after a successful document upload."""

    document_id: str
    filename: str
    status: str
    message: str


# --------------------------------------------------
# Validation and Sanitization
# --------------------------------------------------

def sanitize_filename(filename: str | None) -> str:
    """
    Remove directory prefixes and unsafe path components.

    The original filename is used only for validation.
    It is never used as the stored filename.
    """
    if not filename:
        return ""

    # Handle both Windows and Unix path separators.
    safe_name = filename.replace("\\", "/").split("/")[-1]

    # Reject empty names and special path components.
    if safe_name in {"", ".", ".."}:
        return ""

    return safe_name


def validate_file_type(file: UploadFile, extension: str) -> None:
    """
    Validate the uploaded file's declared MIME type and extension.

    Raises:
        HTTPException: If the file type is not supported.
    """
    expected_extension = ALLOWED_CONTENT_TYPES.get(file.content_type)

    if expected_extension is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF and DOCX are allowed.",
        )

    if extension.lower() != expected_extension:
        raise HTTPException(
            status_code=400,
            detail="File extension does not match the declared content type.",
        )


def validate_file_signature(file_path: Path, extension: str) -> None:
    """
    Validate basic file signatures to reject obvious format mismatches.

    PDF files must begin with a PDF signature.
    DOCX files must be valid ZIP archives containing required entries.
    """
    if extension == ".pdf":
        with file_path.open("rb") as uploaded_file:
            signature = uploaded_file.read(5)

        if signature != b"%PDF-":
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file content.",
            )

    elif extension == ".docx":
        try:
            with zipfile.ZipFile(file_path, "r") as docx_file:
                required_files = {
                    "[Content_Types].xml",
                    "word/document.xml",
                }

                if not required_files.issubset(set(docx_file.namelist())):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid DOCX file structure.",
                    )

        except zipfile.BadZipFile:
            raise HTTPException(
                status_code=400,
                detail="Invalid DOCX file content.",
            )


# --------------------------------------------------
# Storage
# --------------------------------------------------

def save_uploaded_file(
    source: BinaryIO,
    destination: Path,
) -> None:
    """
    Stream an uploaded file to disk using shutil.copyfileobj.

    The caller is responsible for enforcing the upload size limit.
    """
    with destination.open("xb") as destination_file:
        shutil.copyfileobj(
            source,
            destination_file,
            length=CHUNK_SIZE,
        )


# --------------------------------------------------
# Upload Endpoint
# --------------------------------------------------

@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=201,
)
async def upload_document(
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload a PDF or DOCX document with a maximum size of 10 MB.

    Returns:
        UploadResponse: Document identifier and stored filename.

    Raises:
        HTTPException: For invalid files, size violations, or storage errors.
    """

    # 1. Sanitize the original filename.
    original_filename = sanitize_filename(file.filename)

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    extension = Path(original_filename).suffix.lower()

    # 2. Validate MIME type and extension.
    validate_file_type(file, extension)

    # 3. Generate an anonymous, collision-resistant identifier.
    document_id = str(uuid.uuid4())
    stored_filename = f"{document_id}{extension}"
    destination = UPLOAD_DIR / stored_filename

    try:
        # 4. Stream to a temporary file while enforcing the size limit.
        temp_path = UPLOAD_DIR / f".{document_id}.uploading"
        total_size = 0

        try:
            with temp_path.open("xb") as temp_file:
                while chunk := await file.read(CHUNK_SIZE):
                    total_size += len(chunk)

                    if total_size > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="File exceeds the maximum allowed size of 10 MB.",
                        )

                    temp_file.write(chunk)

            # 5. Validate the actual file content.
            validate_file_signature(temp_path, extension)

            # 6. Atomically move the completed file to its final name.
            temp_path.replace(destination)

        finally:
            # Remove incomplete uploads after errors.
            if temp_path.exists():
                temp_path.unlink()

        logger.info(
            "Document uploaded successfully: %s",
            document_id,
        )

        return UploadResponse(
            document_id=document_id,
            filename=stored_filename,
            status="success",
            message="Document uploaded successfully.",
        )

    except HTTPException:
        raise

    except OSError:
        logger.exception("Storage failure while uploading document.")

        raise HTTPException(
            status_code=500,
            detail="Unable to save the uploaded document.",
        )

    finally:
        await file.close()
