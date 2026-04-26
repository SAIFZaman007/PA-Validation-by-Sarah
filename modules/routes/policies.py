"""
Policy routes — /api/policies endpoints.
"""

import re
from pathlib import Path

from fastapi             import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses   import FileResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from modules.schemas.policy import GeneratePolicyRequest
from modules.schemas.prescription import VALID_INSURERS
from modules.services.policy_service import (
    generate_policy,
    get_policy_or_404,
    list_policies,
    upload_policy,
)
from modules.services.submission_service import generate_policy_pdf_bytes

router = APIRouter(prefix="/api", tags=["Policies"])


@router.post("/generate-policy", summary="Generate a synthetic insurance policy")
async def api_generate_policy(
    body:    GeneratePolicyRequest,
    request: Request,
    db:      AsyncSession = Depends(get_db),
):
    """
    Generate a synthetic insurance policy for one of the supported insurers,
    persist it to the database, and return the record.

    Supported insurers: Bupa Arabia | Tawuniya | Al-Rajhi Takaful | Medgulf | Solidarity Saudi Takaful

    The `download_url` in the response links directly to a PDF download.
    Call `GET /api/download/policy/{id}` to retrieve the formatted PDF.
    """
    if body.insurer and body.insurer not in VALID_INSURERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown insurer '{body.insurer}'. "
                f"Valid options: {', '.join(VALID_INSURERS)}."
            ),
        )

    policy = await generate_policy(
        db=db,
        insurer=body.insurer,
        base_url=str(request.base_url),
    )
    return {
        "success":      True,
        "policy":       policy,
        "message":      f"Policy {policy['policy_id']} generated and saved.",
        "download_url": policy["download_url"],
    }


@router.post("/upload-policy", summary="Upload, extract and store a policy document")
async def api_upload_policy(
    request: Request,
    file:    UploadFile = File(...),
    insurer: str        = Form(default="Bupa Arabia"),
    db:      AsyncSession = Depends(get_db),
):
    """
    Upload a policy document (PDF / DOCX / TXT), extract coverage rules, and persist the record.

    The insurer must be one of the supported names. The `download_url` in the response
    links to the original uploaded file.
    """
    if insurer not in VALID_INSURERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown insurer '{insurer}'. "
                f"Valid options: {', '.join(VALID_INSURERS)}."
            ),
        )

    policy = await upload_policy(
        db=db,
        file=file,
        insurer=insurer,
        base_url=str(request.base_url),
    )
    return {
        "success":      True,
        "policy":       policy,
        "message":      f"Processed '{policy['filename']}' — {policy['num_rules']} rule(s) extracted.",
        "download_url": policy["download_url"],
    }


@router.get("/policies", summary="List all stored policies")
async def api_list_policies(
    request: Request,
    db:      AsyncSession = Depends(get_db),
):
    """
    Return all policies, newest first.

    Each policy includes a `download_url` linking to a downloadable PDF:
    - Generated policies → PDF of all coverage rules, built on demand.
    - Uploaded policies  → the original uploaded file.
    """
    policies = await list_policies(db=db, base_url=str(request.base_url))
    return {"success": True, "policies": policies, "count": len(policies)}


@router.get("/policies/{policy_db_id}", summary="Get a single policy by ID")
async def api_get_policy(
    policy_db_id: int,
    request:      Request,
    db:           AsyncSession = Depends(get_db),
):
    """Fetch a single policy record by its database ID."""
    record = await get_policy_or_404(db, policy_db_id)
    from modules.services.policy_service import build_policy_dict
    return {"success": True, "policy": build_policy_dict(record, str(request.base_url))}


@router.get(
    "/download/policy/{policy_db_id}",
    summary="Download a policy as a PDF file",
)
async def download_policy_file(
    policy_db_id: int,
    db:           AsyncSession = Depends(get_db),
):
    """
    Download the file associated with a policy record as a PDF.

    - **Generated policies** → builds a formatted PDF on demand from coverage_rules in the DB.
    - **Uploaded policies**  → serves the original uploaded file (PDF / DOCX / TXT).

    Returns HTTP 404 if the record does not exist.
    """
    record = await get_policy_or_404(db, policy_db_id)

    # Generated policy — build PDF on demand
    if record.source == "generated":
        try:
            pdf_bytes = generate_policy_pdf_bytes(record)
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="PDF generation requires 'reportlab'. Install it and restart the server.",
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", record.policy_id)
        filename  = f"{safe_name}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # Uploaded policy — serve original file
    if record.source == "uploaded" and record.file_path:
        file_path = Path(record.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Uploaded file no longer exists on the server.",
            )
        ext      = file_path.suffix.lower()
        mime_map = {
            ".pdf":  "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc":  "application/msword",
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

    # Fallback: generate PDF from coverage_rules for any edge case
    if record.coverage_rules:
        try:
            pdf_bytes = generate_policy_pdf_bytes(record)
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="PDF generation requires 'reportlab'. Install it and restart the server.",
            )
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", record.policy_id)
        filename  = f"{safe_name}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    raise HTTPException(
        status_code=404,
        detail="No downloadable file is available for this policy record.",
    )
