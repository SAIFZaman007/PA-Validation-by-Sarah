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
from modules.services.policy_service import (
    generate_policy,
    get_policy_or_404,
    list_policies,
    upload_policy,
)
from modules.services.submission_service import generate_policy_pdf_bytes

router = APIRouter(prefix="/api", tags=["Policies"])


@router.post("/generate-policy", summary="Auto-generate a synthetic policy")
async def api_generate_policy(
    body:    GeneratePolicyRequest,
    request: Request,
    db:      AsyncSession = Depends(get_db),
):
    """
    Generate a synthetic insurance policy, persist it to the database,
    and return the record.

    The `download_url` in the response is a direct PDF download link.
    Open it in a browser or call `GET /api/download/policy/{id}` to
    download the policy as a formatted PDF file.
    """
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


@router.post("/upload-policy", summary="Upload a PDF/DOCX policy document")
async def api_upload_policy(
    request: Request,
    file:    UploadFile = File(...),
    insurer: str        = Form(default="Unknown Insurer"),
    db:      AsyncSession = Depends(get_db),
):
    """
    Upload a policy document, extract coverage rules, and persist the record.

    The `download_url` in the response links to the original uploaded file.
    Call `GET /api/download/policy/{id}` to download it.
    """
    policy = await upload_policy(
        db=db,
        file=file,
        insurer=insurer,
        base_url=str(request.base_url),
    )
    return {
        "success":      True,
        "policy":       policy,
        "message":      f"Processed {policy['filename']} — {policy['num_rules']} rule(s) extracted.",
        "download_url": policy["download_url"],
    }


@router.get("/policies", summary="List all stored policies")
async def api_list_policies(
    request: Request,
    db:      AsyncSession = Depends(get_db),
):
    """
    Return all policies, newest first.

    Each policy includes a `download_url` that links to a downloadable PDF:
      - Generated policies → PDF of all coverage rules, built on demand.
      - Uploaded policies  → the original uploaded file (PDF/DOCX/TXT).

    Call `GET /api/download/policy/{id}` to trigger the download.
    """
    policies = await list_policies(db=db, base_url=str(request.base_url))
    return {"success": True, "policies": policies}


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

    - **Generated policies** → builds a formatted PDF from the coverage rules
      stored in the database and streams it directly. The PDF is created on
      demand — no file needs to exist on disk.

    - **Uploaded policies** → serves the original uploaded file (PDF/DOCX/TXT).

    The `Content-Disposition: attachment` header forces the browser to save
    the file rather than display it inline.

    If `reportlab` is not installed, returns HTTP 503 with clear install instructions.
    """
    record = await get_policy_or_404(db, policy_db_id)

    # ── Generated policy — build PDF from DB coverage_rules ───────────────
    if record.source == "generated":
        try:
            pdf_bytes = generate_policy_pdf_bytes(record)
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail=(
                    "PDF generation requires the 'reportlab' package. "
                    "Install it by running: pip install reportlab  "
                    "then restart the server."
                ),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"PDF generation failed: {exc}",
            )

        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", record.policy_id)
        filename  = f"{safe_name}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ── Uploaded policy — serve the original file ──────────────────────────
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

    # ── Fallback: old record with no file — generate PDF from coverage_rules ──
    # This handles records created before the current policy_service was deployed
    # where file_path may have pointed to a now-deleted .txt file.
    if record.coverage_rules:
        try:
            pdf_bytes = generate_policy_pdf_bytes(record)
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail=(
                    "PDF generation requires the 'reportlab' package. "
                    "Install it by running: pip install reportlab  "
                    "then restart the server."
                ),
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