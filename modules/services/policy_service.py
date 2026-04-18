"""
Policy service — all business logic for the /api/policies endpoints.

Keeps route handlers thin: they validate the request, call a function here,
and return the result.  No database queries or file I/O live in the routes.
"""

import io
import re
import uuid
from pathlib import Path
from typing  import Optional

from fastapi             import UploadFile, HTTPException
from sqlalchemy          import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config  import settings, CPT_PATTERN
from core.models  import Policy
from modules.services.ml_pipeline import extractor, extend_policy_rules, generator


# ── File helpers ──────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    """Return True if the file extension is on the allow-list."""
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext in settings.ALLOWED_EXTENSIONS


async def _save_upload(file: UploadFile, subfolder: str) -> tuple[str, str]:
    """
    Write an uploaded file to uploads/<subfolder>/ with a UUID prefix.
    Returns (original_filename, absolute_server_path).
    """
    original  = file.filename or "upload"
    safe      = re.sub(r"[^\w.\-]", "_", original)
    unique    = f"{uuid.uuid4().hex}_{safe}"
    dest_dir  = settings.upload_dir / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / unique

    contents = await file.read()
    dest_path.write_bytes(contents)
    file.file = io.BytesIO(contents)

    return original, str(dest_path)


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract plain text from a PDF. Returns empty string on failure."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as exc:
        print(f"[PDF] Extraction failed — {exc}")
        return ""


def _rules_from_text(text: str, insurer: str) -> list:
    """
    Build a minimal coverage_rules list from raw extracted policy text.

    CPT codes found in the text become individual rules.  When none are
    found a single placeholder rule is returned so the policy is still
    usable for PA submission testing.
    """
    cpt_codes = list(set(re.findall(CPT_PATTERN, text)))[:10]

    if not cpt_codes:
        return [{
            "cpt_code":            "99999",
            "procedure_name":      "Procedure (extracted from document)",
            "category":            "General",
            "requires_prior_auth": True,
            "prerequisites":       [],
            "coverage_criteria":   text[:500] if text else "No text could be extracted.",
            "estimated_cost":      "SAR — see document",
        }]

    return [
        {
            "cpt_code":            cpt,
            "procedure_name":      f"Procedure CPT-{cpt}",
            "category":            "General",
            "requires_prior_auth": True,
            "prerequisites":       [],
            "coverage_criteria":   f"Prior authorization is mandatory. (Source: {insurer})",
            "estimated_cost":      "SAR — see document",
        }
        for cpt in cpt_codes
    ]


def build_download_url(base_url: str, policy_id: int) -> str:
    """Assemble the PDF download URL for a policy record."""
    return f"{base_url.rstrip('/')}/api/download/policy/{policy_id}"


def build_policy_dict(record: Policy, base_url: str) -> dict:
    """
    Merge the ORM dict with a freshly assembled download_url.
    The download_url always points to the PDF download endpoint.
    """
    d = record.to_dict()
    d["download_url"] = build_download_url(base_url, record.id)
    return d


# ── Service functions ─────────────────────────────────────────────────────────

async def generate_policy(
    db: AsyncSession,
    insurer: Optional[str],
    base_url: str,
) -> dict:
    """
    Auto-generate a synthetic policy, persist it, update the rules cache.

    The policy is stored in the database. When the download endpoint is called,
    a PDF is generated on-the-fly from the coverage_rules stored in the DB —
    no separate file is written to disk for generated policies.

    Returns the full policy dict with download_url pointing to the PDF endpoint.
    """
    policy_doc = generator.generate_policy_document(
        insurer_name=insurer, num_procedures=5
    )
    rules = extractor.extract_policy_rules(policy_doc)

    # Generated policies do NOT need a file on disk — the PDF is built
    # on-demand by generate_policy_pdf_bytes() in submission_service.py.
    # filename and file_path are left None so the download router takes
    # the correct "generated" branch.
    record = Policy(
        policy_id      = policy_doc["policy_id"],
        insurer        = policy_doc["insurer"],
        effective_date = policy_doc.get("effective_date", "2024-01-01"),
        version        = policy_doc.get("version", "2024.1"),
        source         = "generated",
        filename       = None,
        file_path      = None,
        coverage_rules = policy_doc["coverage_rules"],
        num_rules      = len(policy_doc["coverage_rules"]),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    extend_policy_rules(rules)

    result = build_policy_dict(record, base_url)
    result["coverage_rules"] = policy_doc["coverage_rules"]
    return result


async def upload_policy(
    db: AsyncSession,
    file: UploadFile,
    insurer: str,
    base_url: str,
) -> dict:
    """
    Accept a policy file, extract text (PDF), persist the record.

    Returns the full policy dict with download_url.
    """
    if not _allowed_file(file.filename or ""):
        ext = Path(file.filename or "").suffix.lower()
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext!r} is not allowed. Upload a PDF or DOCX.",
        )

    ext        = Path(file.filename or "").suffix.lower().lstrip(".")
    file_bytes = await file.read()

    extracted_text = _extract_pdf_text(file_bytes) if ext == "pdf" else ""
    file.file      = io.BytesIO(file_bytes)

    original_name, saved_path = await _save_upload(file, subfolder="policies")

    coverage_rules = _rules_from_text(extracted_text, insurer)
    policy_id      = f"POL-UPLOAD-{uuid.uuid4().hex[:6].upper()}"

    record = Policy(
        policy_id      = policy_id,
        insurer        = insurer,
        effective_date = "2024-01-01",
        version        = "uploaded",
        source         = "uploaded",
        filename       = original_name,
        file_path      = saved_path,
        coverage_rules = coverage_rules,
        num_rules      = len(coverage_rules),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    rules = extractor.extract_policy_rules({"coverage_rules": coverage_rules})
    extend_policy_rules(rules)

    result = build_policy_dict(record, base_url)
    result["coverage_rules"] = coverage_rules
    return result


async def list_policies(db: AsyncSession, base_url: str) -> list:
    """Return all policies ordered by newest first, with download_url on each."""
    result  = await db.execute(select(Policy).order_by(Policy.created_at.desc()))
    records = result.scalars().all()
    return [build_policy_dict(r, base_url) for r in records]


async def get_policy_or_404(db: AsyncSession, policy_db_id: int) -> Policy:
    """Fetch a Policy by PK or raise HTTP 404."""
    result = await db.execute(select(Policy).where(Policy.id == policy_db_id))
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Policy not found.")
    return record