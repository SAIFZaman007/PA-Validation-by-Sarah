"""
Policy service — all business logic for the /api/policies endpoints.
"""

import io
import re
import uuid
from pathlib import Path
from typing  import Optional

from fastapi             import UploadFile, HTTPException
from sqlalchemy          import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config  import settings
from core.models  import Policy
from modules.services.ml_pipeline import extractor, extend_policy_rules, generator


# ── Known procedure catalogue ─────────────

_PROCEDURE_META = {
    "27447": {"name": "Total Knee Arthroplasty",           "category": "Orthopedic Surgery",
              "prerequisites": ["Failed conservative therapy for 6 months", "BMI < 40"],
              "cost": "SAR 15,000 - 25,000", "requires_pa": True},
    "99213": {"name": "Office Visit - Established Patient", "category": "Primary Care",
              "prerequisites": [], "cost": "SAR 100 - 200", "requires_pa": False},
    "70450": {"name": "CT Head without Contrast",           "category": "Radiology",
              "prerequisites": ["Clinical indication documented"],
              "cost": "SAR 500 - 1,500", "requires_pa": True},
    "29827": {"name": "Arthroscopy Shoulder",               "category": "Orthopedic Surgery",
              "prerequisites": ["Failed physical therapy for 12 weeks", "MRI documented tear"],
              "cost": "SAR 8,000 - 15,000", "requires_pa": True},
    "93000": {"name": "Electrocardiogram (ECG)",            "category": "Cardiology",
              "prerequisites": [], "cost": "SAR 50 - 150", "requires_pa": False},
    "45378": {"name": "Colonoscopy",                        "category": "Gastroenterology",
              "prerequisites": ["Age >= 45 OR family history OR symptoms"],
              "cost": "SAR 2,000 - 4,000", "requires_pa": True},
    "88305": {"name": "Tissue Examination",                 "category": "Pathology",
              "prerequisites": [], "cost": "SAR 200 - 500", "requires_pa": False},
    "73721": {"name": "MRI Lower Extremity",                "category": "Radiology",
              "prerequisites": ["Conservative therapy failed", "X-ray completed"],
              "cost": "SAR 1,500 - 3,000", "requires_pa": True},
    "51184": {"name": "Cystography",                        "category": "Urology",
              "prerequisites": ["Clinical indication documented"],
              "cost": "SAR 1,000 - 2,500", "requires_pa": True},
}

_PA_KEYWORDS    = {"prior authorization", "pre-authorization", "preauthorization",
                   "authorization required", "authorization mandatory", "approval required",
                   "must be approved", "mandatory before procedure"}
_NO_PA_KEYWORDS = {"no prior authorization", "not required", "no authorization required",
                   "authorization not required"}

#   "Rule 1: MRI Lower Extremity (CPT: 73721)"
_RULE_HEADER_RE = re.compile(
    r'^Rule\s+\d+:\s+(.+?)\s+\(CPT:\s*(\d{5})\)',
    re.IGNORECASE,
)

# Strip these before using the text as prerequisite values.
_CID_RE = re.compile(r'\(cid:\d+\)\s*')

# Matches a standalone CPT code embedded in a line like "(CPT: 73721)" or "CPT 73721"
_INLINE_CPT_RE = re.compile(r'\(CPT:\s*(\d{5})\)', re.IGNORECASE)


# ── File helpers ──────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext in settings.ALLOWED_EXTENSIONS


async def _save_upload(file: UploadFile, subfolder: str) -> tuple[str, str]:
    """Write an uploaded file to uploads/<subfolder>/ with a UUID prefix.
    Returns (original_filename, absolute_server_path)."""
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


def _extract_file_text(file_bytes: bytes, ext: str) -> str:
    """Extract plain text from PDF / TXT / DOCX. Returns empty string on failure."""
    if ext == "pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join((page.extract_text() or "") for page in pdf.pages)
        except Exception as exc:
            print(f"[PDF] Extraction failed — {exc}")
            return ""

    if ext == "txt":
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception:
            return ""

    if ext == "docx":
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as exc:
            print(f"[DOCX] Extraction failed — {exc}")
            return ""

    return ""


# ── Structured rule-block parser ──────────────────────────────────────────────

def _parse_rule_block(block_lines: list[str], cpt_code: str, proc_name: str) -> dict:
    """
    Parse a single rule block (the lines between two Rule headers) into a
    structured coverage rule dict.

    Field parsing is label-prefix based — the same technique that fixed the
    prescription extractor — so it is immune to (cid:127) bullet characters,
    random 5-digit numbers in non-CPT positions, and multi-line criteria.
    """
    block          = {}
    prereq_lines   = []
    collecting_pre = False

    for line in block_lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Category "):
            block["category"]  = line[len("Category "):].strip()
            collecting_pre     = False

        elif line.startswith("Requires PA "):
            val = line[len("Requires PA "):].strip().lower()
            block["requires_pa"] = val in ("yes", "true", "1")
            collecting_pre       = False

        elif line.startswith("Estimated Cost "):
            block["cost"]  = line[len("Estimated Cost "):].strip()
            collecting_pre = False

        elif line.startswith("Coverage Criteria "):
            block["criteria"] = line[len("Coverage Criteria "):].strip()
            collecting_pre    = False

        elif line.startswith("Prerequisites"):
            # The text immediately after "Prerequisites" on the same line is
            # either "None", a (cid:127)-prefixed item, or empty.
            after = line[len("Prerequisites"):].strip()
            collecting_pre = True
            if after and after.lower() not in ("none", "—", "-"):
                cleaned = _CID_RE.sub("", after).strip()
                if cleaned:
                    prereq_lines.append(cleaned)

        elif collecting_pre:
            # Continuation lines for prerequisites — strip (cid:NNN) bullets.
            # Stop collecting if we hit the document footer.
            if any(line.lower().startswith(fp) for fp in (
                "this document was", "legally binding", "pa validation system |"
            )):
                collecting_pre = False
                continue
            cleaned = _CID_RE.sub("", line).strip()
            if cleaned and cleaned.lower() not in ("none", "—"):
                prereq_lines.append(cleaned)

    prerequisites = [p for p in prereq_lines if p and len(p) > 2]
    requires_pa   = block.get("requires_pa", True)

    # Use the Coverage Criteria text parsed directly from the document — it is always more accurate than anything we can reconstruct from prerequisites.
    criteria = block.get("criteria", "")
    if not criteria:
        if requires_pa:
            criteria = "Prior authorization is mandatory before procedure."
            if prerequisites:
                criteria += " Patient must meet the following conditions: " + \
                            "  ".join(f"- {p}" for p in prerequisites)
        else:
            criteria = "No prior authorization required."

    return {
        "cpt_code":            cpt_code,
        "procedure_name":      proc_name,
        "category":            block.get("category", "General"),
        "requires_prior_auth": requires_pa,
        "prerequisites":       prerequisites,
        "coverage_criteria":   criteria,
        "estimated_cost":      block.get("cost", "SAR — see document"),
    }


def _parse_rules_structured(text: str) -> list[dict]:
    """
    Parse coverage rules from a policy document using the Rule-header structure.

    Looks for lines matching:  "Rule N: <Procedure Name> (CPT: XXXXX)"
    and treats everything between consecutive Rule headers as a field block.

    This is the primary strategy for any policy generated by this system and
    for any well-formatted third-party policy document.

    Returns an empty list when no Rule headers are found (triggers fallback).
    """
    lines        = text.splitlines()
    rule_starts  = []

    for i, line in enumerate(lines):
        m = _RULE_HEADER_RE.match(line.strip())
        if m:
            rule_starts.append((i, m.group(2), m.group(1).strip()))

    if not rule_starts:
        return []

    rules = []
    for idx, (start_line, cpt_code, proc_name) in enumerate(rule_starts):
        end_line    = rule_starts[idx + 1][0] if idx + 1 < len(rule_starts) else len(lines)
        block_lines = lines[start_line + 1 : end_line]
        rule        = _parse_rule_block(block_lines, cpt_code, proc_name)
        rules.append(rule)

    return rules


def _parse_rules_fallback(text: str, insurer: str) -> list[dict]:
    """
    Fallback for third-party / non-standard policy documents that don't use
    the "Rule N: Name (CPT: XXXXX)" header format.

    Scans for CPT codes embedded in explicit "(CPT: XXXXX)" patterns only —
    never using the bare \\b\\d{5}\\b regex — so policy IDs, license numbers,
    and other 5-digit numbers in the filename or header are never mistaken for
    CPT codes.

    Enriches each found CPT with catalogue metadata where available.
    """
    # Only match CPT codes from explicit "(CPT: XXXXX)" or "CPT Code XXXXX" patterns
    cpt_matches = _INLINE_CPT_RE.findall(text)
    # Preserve insertion order
    cpt_codes = list(dict.fromkeys(cpt_matches))[:10]

    if not cpt_codes:
        # Last resort: search for "CPT Code XXXXX" label-value pattern
        for m in re.finditer(r'\bCPT\s+(?:Code\s+)?(\d{5})\b', text, re.IGNORECASE):
            cpt = m.group(1)
            if cpt not in cpt_codes:
                cpt_codes.append(cpt)

    if not cpt_codes:
        return [{
            "cpt_code":            "99999",
            "procedure_name":      "Procedure (extracted from document)",
            "category":            "General",
            "requires_prior_auth": True,
            "prerequisites":       [],
            "coverage_criteria":   text[:500].strip() if text else "No text could be extracted.",
            "estimated_cost":      "SAR — see document",
        }]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    rules = []
    for cpt in cpt_codes:
        meta = _PROCEDURE_META.get(cpt)

        # Find paragraph context for this CPT code
        ctx_paragraphs = [p for p in paragraphs if cpt in p]
        context_text   = " ".join(ctx_paragraphs)[:800] if ctx_paragraphs else ""
        ctx_lower      = context_text.lower()

        # Detect PA requirement from context
        if any(kw in ctx_lower for kw in _NO_PA_KEYWORDS):
            requires_pa = False
        elif any(kw in ctx_lower for kw in _PA_KEYWORDS):
            requires_pa = True
        else:
            requires_pa = meta["requires_pa"] if meta else True

        if meta:
            name          = meta["name"]
            category      = meta["category"]
            prerequisites = list(meta["prerequisites"])
            cost          = meta["cost"]
        else:
            name          = f"Procedure CPT-{cpt}"
            category      = "General"
            cost          = "SAR — see document"
            prerequisites = []
            for line in context_text.splitlines():
                line = _CID_RE.sub("", line).strip(" •-–\t")
                if line and len(line) > 10 and any(
                    kw in line.lower() for kw in
                    ["required", "must", "failed", "documented", "completed",
                     "prior", "age", "bmi", "history", "x-ray", "mri", "therapy"]
                ):
                    prerequisites.append(line[:200])
            prerequisites = prerequisites[:4]

        if requires_pa:
            criteria = "Prior authorization is mandatory before procedure."
            if prerequisites:
                criteria += " Patient must meet the following conditions: " + \
                            "  ".join(f"- {p}" for p in prerequisites)
        else:
            criteria = "No prior authorization required."

        rules.append({
            "cpt_code":            cpt,
            "procedure_name":      name,
            "category":            category,
            "requires_prior_auth": requires_pa,
            "prerequisites":       prerequisites,
            "coverage_criteria":   criteria,
            "estimated_cost":      cost,
        })

    return rules


def _rules_from_text(text: str, insurer: str) -> list[dict]:
    """
    Master rule extractor.

    Strategy 1 (primary):   Structured Rule-header parser — parses the exact
                            layout of our generated policy PDFs. Extracts CPT
                            codes only from Rule headers, so policy IDs (e.g.
                            POL-80692) and license numbers are never confused
                            with procedure codes.

    Strategy 2 (fallback):  (CPT: XXXXX) inline pattern scanner — for third-
                            party documents. Uses explicit CPT patterns, not the
                            bare \\b\\d{5}\\b regex, to avoid false positives.

    Always returns at least one rule.
    """
    # Try structured parser first
    rules = _parse_rules_structured(text)
    if rules:
        return rules

    # Fall back to inline pattern scanner
    return _parse_rules_fallback(text, insurer)


# ── URL / dict helpers ────────────────────────────────────────────────────────

def build_download_url(base_url: str, policy_id: int) -> str:
    return f"{base_url.rstrip('/')}/api/download/policy/{policy_id}"


def build_policy_dict(record: Policy, base_url: str) -> dict:
    d = record.to_dict()
    d["download_url"] = build_download_url(base_url, record.id)
    return d


# ── Service functions ─────────────────────────────────────────────────────────

async def generate_policy(
    db:       AsyncSession,
    insurer:  Optional[str],
    base_url: str,
) -> dict:
    """
    Auto-generate a synthetic policy, persist it, and update the rules cache.

    The PDF is built on-demand from coverage_rules stored in the DB.
    No file is written to disk for generated policies.
    """
    policy_doc = generator.generate_policy_document(
        insurer_name=insurer, num_procedures=5
    )
    rules = extractor.extract_policy_rules(policy_doc)

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
    db:       AsyncSession,
    file:     UploadFile,
    insurer:  str,
    base_url: str,
) -> dict:
    """
    Accept a policy document (PDF / DOCX / TXT), extract every coverage rule
    using the structured rule-block parser, persist the record, and update the
    rules cache.

    The structured parser correctly handles:
      - Policy ID numbers (e.g. POL-80692) — never confused with CPT codes.
      - (cid:127) PDF bullet characters — stripped cleanly.
      - ECG / Cardiology "Requires PA: No" — correctly sets requires_prior_auth=False.
      - Multi-prerequisite rules — all bullet items captured individually.
      - Coverage Criteria text — extracted verbatim from the document.
    """
    if not _allowed_file(file.filename or ""):
        ext = Path(file.filename or "").suffix.lower()
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext!r} is not allowed. Upload a PDF, DOCX, or TXT file.",
        )

    ext        = Path(file.filename or "").suffix.lower().lstrip(".")
    file_bytes = await file.read()

    extracted_text = _extract_file_text(file_bytes, ext)
    file.file      = io.BytesIO(file_bytes)

    original_name, saved_path = await _save_upload(file, subfolder="policies")

    coverage_rules = _rules_from_text(extracted_text, insurer)
    policy_id      = f"POL-{uuid.uuid4().hex[:6].upper()}"

    print(
        f"[policy-upload] '{original_name}' — "
        f"extracted {len(extracted_text):,} chars, "
        f"{len(coverage_rules)} rule(s) for {insurer}"
    )
    for r in coverage_rules:
        print(
            f"  CPT {r['cpt_code']} — {r['procedure_name']} — "
            f"PA={r['requires_prior_auth']} — prereqs={r['prerequisites']}"
        )

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