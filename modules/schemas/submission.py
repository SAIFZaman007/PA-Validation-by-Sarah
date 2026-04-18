"""
Pydantic schemas for the /api/submit-request and /api/submissions endpoints.

Two submission modes share one endpoint:
  Mode 1 — Auto-generate  (Content-Type: multipart/form-data, no file)
  Mode 2 — File upload    (Content-Type: multipart/form-data, file attached)

The response shape is identical for both modes so the frontend
never needs to branch on how the request was submitted.
"""

from typing   import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Mode-1 request body ───────────────────────────────────────────────────────

class AutoGenerateRequest(BaseModel):
    """Body for the autogenerate mode (JSON)."""

    auto_generate: bool = True
    likelihood:    str  = Field(
        default="random",
        description="Expected approval likelihood hint fed to the data generator.",
        examples=["random", "high", "medium", "low"],
    )


# ── Routing decision (mirrors CaseRouter.route_case output) ──────────────────

class RoutingDecision(BaseModel):
    """The ML routing decision returned by CaseRouter.route_case()."""

    tier:                str    # "auto_approve" | "auto_deny" | "manual_review"
    action:              str    # "AUTO-APPROVE" | "AUTO-DENY" | "MANUAL REVIEW"
    color:               str    # "green" | "red" | "yellow"
    confidence:          float
    explanation:         str
    prediction:          str    # "Approve" | "Deny"
    approve_probability: float
    deny_probability:    float


# ── Live statistics ───────────────────────────────────────────────────────────

class StatsPayload(BaseModel):
    """The stats block embedded in every submit response."""

    total_requests: int
    auto_approved:  int
    auto_denied:    int
    manual_review:  int


# ── Submit response ───────────────────────────────────────────────────────────

class SubmitResponse(BaseModel):
    """
    Response envelope for POST /api/submit-request.

    Identical structure for both autogenerate and file-upload modes.

    Key fields for follow-up actions:
      result_id    — use with GET /api/download/request/{result_id} to download
                     the validation-result report.
      download_url — pre-built direct link to the validation-result report.
      reasoning    — plain-English reliability assessment of the ML decision.
    """

    success:      bool
    request:      Dict[str, Any]       # the raw PA request dict
    features:     Dict[str, Any]       # the 16-feature vector
    routing:      RoutingDecision
    stats:        StatsPayload

    # ── Identity / download ───────────────────────────────────────────────
    result_id:    int                  # DB primary key — use with /api/download/request/{result_id}
    record_id:    int                  # alias of result_id, kept for backward compatibility
    download_url: str                  # direct link to the validation-result report

    # ── Decision reasoning ────────────────────────────────────────────────
    reasoning:    str                  # plain-English confidence and reliability statement

    # ── File-upload quality metadata (None for auto-generated) ────────────
    extraction_status:  Optional[str]  # "success" | "partial" | "failed" | None
    extraction_warning: Optional[str]  # user-facing message when quality < success
    chars_extracted:    Optional[int]  # character count extracted from uploaded file
    cpt_codes_found:    Optional[List[str]]    # CPT codes parsed from uploaded file
    icd10_codes_found:  Optional[List[str]]    # ICD-10 codes parsed from uploaded file


# ── Submissions history ───────────────────────────────────────────────────────

class SubmissionRecord(BaseModel):
    """One row from the validation_results table."""

    id:                  int
    request_id:          str
    timestamp:           str
    insurer:             Optional[str]
    patient_name:        Optional[str]
    patient_age:         Optional[int]
    patient_gender:      Optional[str]
    cpt_code:            Optional[str]
    procedure_name:      Optional[str]
    procedure_category:  Optional[str]
    icd10_code:          Optional[str]
    diagnosis_desc:      Optional[str]
    routing_tier:        str
    routing_action:      str
    routing_color:       str
    confidence:          float
    approve_probability: float
    deny_probability:    float
    explanation:         Optional[str]
    source:              str
    uploaded_filename:   Optional[str]
    created_at:          str
    download_url:        str           # always present — points to validation-result report


class SubmissionsListResponse(BaseModel):
    """Response envelope for GET /api/submissions."""

    success:     bool
    submissions: List[SubmissionRecord]


# ── Stats endpoint ────────────────────────────────────────────────────────────

class StatsResponse(BaseModel):
    """Response envelope for GET /api/stats."""

    stats:           StatsPayload
    automation_rate: float
    time_saved_min:  int