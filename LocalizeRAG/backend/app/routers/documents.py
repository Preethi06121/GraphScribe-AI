import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.deps import get_graph_builder, get_ingestion_pipeline
from app.graph.graph_builder import GraphBuilder
from app.rag.ingestion_pipeline import IngestionPipeline
from app.schemas.document import DocumentUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}
ALLOWED_EXTENSION = ".pdf"


def _validate_pdf_file(file: UploadFile) -> str:
    filename = file.filename or ""
    if not filename.lower().endswith(ALLOWED_EXTENSION):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    return filename


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
    graph_builder: GraphBuilder = Depends(get_graph_builder),
) -> DocumentUploadResponse:
    document_name = _validate_pdf_file(file)
    document_id = str(uuid.uuid4())
    settings = get_settings()

    upload_dir = Path(settings.temp_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ALLOWED_EXTENSION,
            dir=upload_dir,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        logger.info(
            "Processing uploaded document: %s (id=%s)",
            document_name,
            document_id,
        )
        result = await pipeline.ingest(temp_path, document_id, document_name)
        await graph_builder.build_from_file(temp_path, document_id, document_name)

        return DocumentUploadResponse(
            status="success",
            document_id=result.document_id,
            document=result.document_name,
            pages=result.pages,
            chunks=result.chunks,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Document ingestion failed for %s: %s", document_name, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during document upload: %s", document_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the document.",
        ) from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
