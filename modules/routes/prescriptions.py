"""
Prescription routes — /api/prescriptions endpoints.
"""

import re
from pathlib import Path

from fastapi             import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses   import FileResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from modules.schemas.prescription import GeneratePrescriptionRequest, VALID_INSURERS
from modules.services.prescription_service import (
    generate_prescription,
    get_prescription_or_404,
    generate_prescription_pdf_bytes,
    list_prescriptions,
    upload_prescription,
)

router = APIRouter(prefix="/api", tags=["Prescriptions"])


@router.post("/generate-prescription", summary="Generate a synthetic patient prescription")
async def api_generate_prescription(
    body:    GeneratePrescriptionRequest,
    request: Request,
    db:      AsyncSession = Depends(get_db),
):
    """
    Generate a synthetic prescription that mirrors every field in the
    `synthetic_pa_requests.json` schema, persist it, and return the record.

    Accepted insurers: Bupa Arabia | Tawuniya | Al-Rajhi Takaful | Medgulf | Solidarity Saudi Takaful

    The `download_url` in the response links directly to the PDF download endpoint.
    """
    if body.insurer and body.insurer not in VALID_INSURERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown insurer '{body.insurer}'. "
                f"Valid options: {', '.join(VALID_INSURERS)}."
            ),
        )

    prescription = await generate_prescription(
        db=db,
        insurer=body.insurer,
        likelihood=body.approval_likelihood or "random",
        base_url=str(request.base_url),
    )
    return {
        "success":      True,
        "prescription": prescription,
        "message":      f"Prescription {prescription['prescription_id']} generated and saved.",
        "download_url": prescription["download_url"],
    }


@router.post("/upload-prescription", summary="Upload, extract and store a prescription PDF")
async def api_upload_prescription(
    request: Request,
    file:    UploadFile = File(...),
    insurer: str        = Form(default="Bupa Arabia"),
    db:      AsyncSession = Depends(get_db),
):
    """
    Upload a prescription document (PDF / DOCX / TXT), extract clinical fields,
    and persist the record.

    The `download_url` in the response links to a generated PDF for the stored record.
    """
    if insurer not in VALID_INSURERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown insurer '{insurer}'. "
                f"Valid options: {', '.join(VALID_INSURERS)}."
            ),
        )

    prescription = await upload_prescription(
        db=db,
        file=file,
        insurer=insurer,
        base_url=str(request.base_url),
    )
    return {
        "success":      True,
        "prescription": prescription,
        "message":      f"Processed '{prescription['filename']}' and stored prescription {prescription['prescription_id']}.",
        "download_url": prescription["download_url"],
    }


@router.get("/prescriptions", summary="List all stored prescriptions")
async def api_list_prescriptions(
    request: Request,
    db:      AsyncSession = Depends(get_db),
):
    """Return all prescriptions, newest first, each with a `download_url`."""
    prescriptions = await list_prescriptions(db=db, base_url=str(request.base_url))
    return {"success": True, "prescriptions": prescriptions}


@router.get(
    "/prescriptions/{prescription_db_id}",
    summary="Get a single prescription by ID",
)
async def api_get_prescription(
    prescription_db_id: int,
    request: Request,
    db:      AsyncSession = Depends(get_db),
):
    """Fetch a single prescription record by its database ID."""
    record = await get_prescription_or_404(db, prescription_db_id)
    from modules.services.prescription_service import build_prescription_dict
    return {"success": True, "prescription": build_prescription_dict(record, str(request.base_url))}


@router.get(
    "/download/prescription/{prescription_db_id}",
    summary="Download a prescription as a PDF",
)
async def download_prescription_file(
    prescription_db_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Download the prescription record as a formatted PDF.

    - **Generated prescriptions** → PDF built on demand from stored fields.
    - **Uploaded prescriptions**  → serves the original uploaded file (if it still
      exists on disk); otherwise falls back to a generated PDF from stored data.
    """
    record = await get_prescription_or_404(db, prescription_db_id)

    # For uploaded records, serve original file if available
    if record.source == "uploaded" and record.file_path:
        file_path = Path(record.file_path)
        if file_path.exists():
            ext      = file_path.suffix.lower()
            mime_map = {
                ".pdf":  "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".txt":  "text/plain",
            }
            media_type = mime_map.get(ext, "application/octet-stream")
            filename   = record.filename or file_path.name
            return FileResponse(
                path=str(file_path),
                media_type=media_type,
                filename=filename,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

    # Generated or fallback — build PDF from stored fields
    try:
        pdf_bytes = generate_prescription_pdf_bytes(record)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="PDF generation requires the 'reportlab' package. Install it and restart.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", record.prescription_id)
    filename  = f"{safe_name}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
