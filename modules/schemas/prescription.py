"""
Pydantic schemas for the /api/prescriptions endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

VALID_INSURERS = [
    "Bupa Arabia",
    "Tawuniya",
    "Al-Rajhi Takaful",
    "Medgulf",
    "Solidarity Saudi Takaful",
]

VALID_LIKELIHOODS = ["high", "medium", "low", "random"]


class GeneratePrescriptionRequest(BaseModel):
    """Body for POST /api/generate-prescription."""

    insurer: Optional[str] = Field(
        default=None,
        description=f"Insurer name. One of: {', '.join(VALID_INSURERS)}. Random if omitted.",
        examples=["Bupa Arabia"],
    )
    approval_likelihood: Optional[str] = Field(
        default="random",
        description="Controls how approvable the generated request is: high | medium | low | random.",
        examples=["high"],
    )
