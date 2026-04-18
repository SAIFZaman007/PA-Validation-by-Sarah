"""
Submission routes — PA request validation and history endpoints.
"""

import re
from pathlib import Path

from fastapi              import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses    import Response
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from modules.services.submission_service import (
    generate_validation_report_bytes,
    get_result_or_404,
    list_submissions,
    live_stats,
    reset_all_results,
    submit_autogenerate,
    submit_file_upload,
)

router = APIRouter(prefix="/api", tags=["Submissions"])

@router.post("/submit-request", summary="Submit a PA request for ML validation")
async def api_submit_request(
    request:    Request,
    db:         AsyncSession = Depends(get_db),
    file:       UploadFile   = File(default=None),
    likelihood: str          = Form(default="random"),
):
    """
    Validate a PA request and return the routing decision.

    The endpoint accepts **two modes**:

    **Mode 1 — Auto-generate** (no file attached, likelihood from dropdown):
      - Dropdown options: `random` | `high` | `medium` | `low`
      - A synthetic PA request is generated from the latest loaded policy.
      - The result is stored permanently in the database.

    **Mode 2 — File upload** (PDF, DOCX, or TXT attached):
      - The clinical document is saved to `uploads/requests/`.
      - Text is extracted from the document (PDF via pdfplumber).
      - CPT and ICD-10 codes are parsed from the extracted text.
      - **If the file is unreadable or contains no clinical codes, the system
        returns HTTP 422 with a clear error — it will NOT process garbage or
        irrelevant files.**

    **Key response fields (both modes):**
      - `result_id`     — **DB primary key for this submission** (e.g. 7, 8, 9).
                          Use this value with `GET /api/download/request/{result_id}`
                          to download the validation result PDF.
                          ⚠ Do NOT confuse with `request_id` (PA-XXXXXX string).
      - `download_url`  — Direct link to the validation result PDF.
                          Works for both auto-generated and file-upload submissions.
      - `reasoning`     — Plain-English explanation of the decision:
                          how confident the model is and what to do next.
    """
    base_url = str(request.base_url)

    if file is not None and file.filename:
        return await submit_file_upload(
            db=db,
            file=file,
            likelihood=likelihood,
            base_url=base_url,
        )

    return await submit_autogenerate(
        db=db,
        likelihood=likelihood,
        base_url=base_url,
    )

@router.get(
    "/download/request/{result_id}",
    summary="Download the validation result report as a PDF",
)
async def download_request_file(
    result_id: int,
    db:        AsyncSession = Depends(get_db),
):
    """
    Download a formatted PDF validation result report for any submission.

    **`result_id` is the `result_id` returned by `POST /api/submit-request`.**
    It is the database row number (e.g. 7) — NOT the `request_id` string
    (PA-XXXXXX) shown in the request details.

    Available for **both** submission modes:
      - Auto-generated submissions → full PDF report of the validation result.
      - File-upload submissions    → full PDF report (also references the
                                     uploaded document filename).

    The PDF report includes:
      - result_id prominently highlighted (so you always know the right ID)
      - Routing decision with colour-coded tier (approve / deny / review)
      - Confidence and probability scores
      - Plain-English reasoning and guidance
      - Patient, procedure, and diagnosis details

    The `Content-Disposition: attachment` header forces the browser to
    save the PDF to the Downloads folder.

    Returns HTTP 404 when `result_id` does not exist in the database.
    """
    record = await get_result_or_404(db, result_id)

    # Generate the PDF validation result report on demand — nothing is stored
    pdf_bytes = generate_validation_report_bytes(record)

    safe_req_id = re.sub(
        r"[^A-Za-z0-9_-]", "_",
        record.request_id or f"record_{result_id}"
    )
    filename = f"validation_result_{safe_req_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@router.get("/submissions", summary="List all stored validation results")
async def api_list_submissions(
    request: Request,
    db:      AsyncSession = Depends(get_db),
    limit:   int          = Query(default=200, ge=1, le=1000),
    tier:    str          = Query(
        default=None,
        description="Filter by routing tier: auto_approve | auto_deny | manual_review",
    ),
):
    """
    Return stored validation results, newest first.

    Each record includes a `download_url` — a direct link to the PDF
    validation result report for that submission.
    """
    rows = await list_submissions(
        db=db,
        base_url=str(request.base_url),
        limit=limit,
        tier=tier,
    )
    return {"success": True, "submissions": rows}

@router.get("/stats", summary="Live routing statistics")
async def api_stats(db: AsyncSession = Depends(get_db)):
    """Live routing statistics polled by the Home page and Analytics Dashboard."""
    stats     = await live_stats(db)
    total     = stats["total_requests"]
    automated = stats["auto_approved"] + stats["auto_denied"]
    rate      = round(automated / total * 100, 1) if total else 0.0

    return {
        "stats":           stats,
        "automation_rate": rate,
        "time_saved_min":  automated * 15,
    }

@router.post("/reset-stats", summary="Delete all validation results (demo only)")
async def api_reset_stats(db: AsyncSession = Depends(get_db)):
    """FOR DEMO AND TESTING ONLY. Add auth before exposing in production."""
    await reset_all_results(db)
    return {"success": True, "message": "All validation results cleared."}