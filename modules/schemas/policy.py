"""
Pydantic schemas for the /api/policies endpoints.

These are the data-transfer objects (DTOs) that FastAPI uses to:
  - Validate and parse incoming request bodies.
  - Serialise outgoing JSON responses with type safety.

Keeping schemas separate from ORM models avoids tight coupling between
the API contract and the database layer.
"""

from datetime import datetime
from typing   import List, Optional

from pydantic import BaseModel, Field


# ── Request bodies ────────────────────────────────────────────────────────────

class GeneratePolicyRequest(BaseModel):
    """Body for POST /api/generate-policy."""

    insurer: Optional[str] = Field(
        default=None,
        description="Insurance company name. Randomly selected when omitted.",
        examples=["Bupa Arabia"],
    )


# ── Response models ───────────────────────────────────────────────────────────

class CoverageRule(BaseModel):
    """One coverage rule inside a policy."""

    cpt_code:            str
    procedure_name:      str
    category:            str
    requires_prior_auth: bool
    prerequisites:       List[str]
    coverage_criteria:   str
    estimated_cost:      str


class PolicyResponse(BaseModel):
    """A single policy returned by the API."""

    id:             int
    policy_id:      str
    insurer:        str
    effective_date: str
    version:        str
    source:         str               # "generated" or "uploaded"
    filename:       Optional[str]     # only set for uploaded policies
    num_rules:      int
    coverage_rules: List[dict]        # kept as dict for flexibility
    created_at:     str
    download_url:   str               # always present — generated or real file


class PolicyListResponse(BaseModel):
    """Response envelope for GET /api/policies."""

    success:  bool
    policies: List[PolicyResponse]


class PolicyCreateResponse(BaseModel):
    """Response envelope for POST /api/generate-policy and /api/upload-policy."""

    success:  bool
    policy:   PolicyResponse
    message:  str