"""
Submission routes — PA validation and history endpoints.
"""

import re
from pathlib import Path

from fastapi              import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses    import Response
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from modules.schemas.prescription import VALID_INSURERS
from modules.services.submission_service import (
    generate_validation_report_bytes,
    get_available_insurers,
    get_result_or_404,
    list_submissions,
    live_stats,
    reset_all_results,
    submit_prescription_upload,
)

router = APIRouter(prefix="/api", tags=["Submissions"])


@router.post(
    "/submit-request",
    summary="Submit a prescription PDF for ML PA validation",
)
async def api_submit_request(
    request: Request,
    db:      AsyncSession = Depends(get_db),
    file:    UploadFile   = File(..., description="Prescription PDF to validate"),
    insurer: str          = Form(
        ...,
        description=(
            "Insurer name — the policy for this insurer is automatically loaded. "
            f"Valid values: {', '.join(VALID_INSURERS)}"
        ),
    ),
):
    """
    **Upload a prescription PDF + select an insurer** to validate a PA request.

    **What happens under the hood:**
    1. The prescription PDF is uploaded and text is extracted.
    2. The system automatically reads the most recent policy for the selected insurer.
    3. The prescription is cross-checked against the policy's coverage rules.
    4. The ML pipeline (Logistic Regression) produces a routing decision.
    5. A detailed validation report is generated and stored.

    **Key response fields:**
    - `result_id`    — DB primary key. Use with `GET /api/download/request/{result_id}` for PDF.
    - `download_url` — Direct link to the PDF validation report.
    - `routing`      — Decision: `auto_approve` | `auto_deny` | `manual_review`.
    - `reasoning`    — Plain-English explanation of the clinical decision.
    - `policy_used`  — The policy that was matched and evaluated.

    **Note:** The same prescription + insurer combination always produces the same
    prediction (deterministic model). Download the PDF at any time for an identical report.
    """
    return await submit_prescription_upload(
        db=db,
        file=file,
        insurer=insurer,
        base_url=str(request.base_url),
    )


@router.get("/insurers-with-policies", summary="List insurers that have an active policy")
async def api_get_insurers_with_policies(db: AsyncSession = Depends(get_db)):
    """
    Return the list of insurers for which at least one policy exists in the database.

    Use this to populate the insurer dropdown on the Submissions page before
    calling POST /api/submit-request.
    """
    insurers = await get_available_insurers(db)
    return {
        "success":  True,
        "insurers": insurers,
        "message":  (
            f"{len(insurers)} insurer(s) with active policies. "
            "Generate or upload policies for additional insurers via the Policies section."
        ),
    }


@router.get(
    "/download/request/{result_id}",
    summary="Download the validation result report as a PDF",
)
async def download_request_file(
    result_id: int,
    db:        AsyncSession = Depends(get_db),
):
    """
    Download a formatted PDF validation report for any submission.

    `result_id` is the database row ID returned by `POST /api/submit-request`.
    The PDF includes the routing decision, probability scores, reasoning,
    patient/procedure/diagnosis data, and all policy coverage rules evaluated.
    """
    record = await get_result_or_404(db, result_id)
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

    Each record includes:
    - `download_url` — direct link to the PDF validation report.
    - `policy_id_used` — which policy was evaluated for this submission.
    - `routing_tier` — the ML decision tier.
    """
    rows = await list_submissions(
        db=db,
        base_url=str(request.base_url),
        limit=limit,
        tier=tier,
    )
    return {"success": True, "submissions": rows, "count": len(rows)}


@router.get(
    "/submissions/{result_id}",
    summary="Get a single validation result by ID",
)
async def api_get_submission(
    result_id: int,
    request:   Request,
    db:        AsyncSession = Depends(get_db),
):
    """Fetch a single validation result record by its database ID."""
    record = await get_result_or_404(db, result_id)
    d = record.to_dict()
    d["download_url"] = f"{str(request.base_url).rstrip('/')}/api/download/request/{record.id}"
    return {"success": True, "submission": d}


@router.get("/stats", summary="Live routing statistics")
async def api_stats(db: AsyncSession = Depends(get_db)):
    """
    Live routing statistics — polled by the dashboard and analytics pages.

    Returns total counts, automation rate, time saved, and per-insurer breakdown.
    """
    stats     = await live_stats(db)
    total     = stats["total_requests"]
    automated = stats["auto_approved"] + stats["auto_denied"]
    rate      = round(automated / total * 100, 1) if total else 0.0

    return {
        "stats":           stats,
        "automation_rate": rate,
        "time_saved_min":  automated * 15,
        "by_insurer":      stats.get("by_insurer", []),
    }


@router.post("/reset-stats", summary="Delete all validation results (demo only)")
async def api_reset_stats(db: AsyncSession = Depends(get_db)):
    """FOR DEMO AND TESTING ONLY. Add authentication before exposing in production."""
    await reset_all_results(db)
    return {"success": True, "message": "All validation results cleared."}
