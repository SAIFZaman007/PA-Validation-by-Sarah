"""
Submission service — business logic for PA request validation.

Two modes share one code path:
  Mode 1 — Auto-generate (no file, likelihood from dropdown)
  Mode 2 — File upload   (clinical document PDF or DOCX)

Both modes run the same ML pipeline and persist to validation_results.
"""

import io
import re
import textwrap
import uuid
import datetime
from pathlib import Path
from typing  import Optional, Tuple

from fastapi             import HTTPException, UploadFile
from sqlalchemy          import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config  import settings, CPT_PATTERN
from core.models  import Policy, ValidationResult

from modules.services.ml_pipeline import (
    engineer,
    extractor,
    generator,
    get_policy_rules,
    router,
)


# ── File helpers ──────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext in settings.ALLOWED_EXTENSIONS


async def _save_upload(file: UploadFile, subfolder: str) -> Tuple[str, str]:
    """Save an uploaded file with a UUID prefix. Returns (original_name, absolute_path)."""
    original  = file.filename or "upload"
    safe      = re.sub(r"[^\w.\-]", "_", original)
    unique    = f"{uuid.uuid4().hex}_{safe}"
    dest_dir  = settings.upload_dir / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / unique

    contents  = await file.read()
    dest_path.write_bytes(contents)
    file.file = io.BytesIO(contents)

    return original, str(dest_path)


def _extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from an uploaded document.

    Supports:
      - PDF  -> pdfplumber (page-by-page extraction)
      - TXT  -> direct UTF-8 decode
      - DOCX -> python-docx paragraph extraction (if installed)

    Returns an empty string on any extraction failure.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join(
                    (page.extract_text() or "") for page in pdf.pages
                )
        except Exception as exc:
            print(f"[PDF] Extraction failed — {exc}")
            return ""

    if ext == ".txt":
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception:
            return ""

    if ext == ".docx":
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            return ""
        except Exception as exc:
            print(f"[DOCX] Extraction failed — {exc}")
            return ""

    return ""


def _assess_extraction_quality(
    extracted_text: str,
    cpt_codes: list,
    icd10_codes: list,
    filename: str,
) -> dict:
    """
    Assess whether the uploaded file contained useful clinical data.

    Returns a dict with:
      extraction_status  — "success" | "partial" | "failed"
      extraction_warning — human-readable message, or None
      chars_extracted    — raw character count

    Rules:
      failed  — no text at all (image-only, binary, password-protected file)
      partial — text exists but zero CPT/ICD-10 codes found (irrelevant document)
      success — text AND at least one clinical code found
    """
    chars = len(extracted_text.strip())

    if chars == 0:
        return {
            "extraction_status":  "failed",
            "extraction_warning": (
                f"No readable text could be extracted from '{filename}'. "
                "The file may be scanned, image-only, password-protected, or "
                "in an unsupported format.  Please upload a text-selectable PDF, "
                "DOCX, or TXT file that contains clinical prior-authorisation data."
            ),
            "chars_extracted": 0,
        }

    if not cpt_codes and not icd10_codes:
        return {
            "extraction_status":  "partial",
            "extraction_warning": (
                f"Text was extracted from '{filename}' ({chars:,} characters), "
                "but no CPT procedure codes or ICD-10 diagnosis codes were found.  "
                "This document does not appear to be a clinical prior-authorisation "
                "record.  Please upload a document that contains valid CPT and/or "
                "ICD-10 codes so the ML pipeline can produce a meaningful result."
            ),
            "chars_extracted": chars,
        }

    return {
        "extraction_status":  "success",
        "extraction_warning": None,
        "chars_extracted":    chars,
    }


def _build_clinical_context_from_text(extracted_text: str, base_policy: dict) -> tuple:
    """
    Enrich the PA request context with data parsed from the uploaded document.

    Extracts CPT and ICD-10 codes from the text and attaches them as hints
    so the generator produces a request that reflects the document content.
    """
    import re as _re
    from core.config import ICD10_PATTERN

    cpt_codes   = list(set(_re.findall(CPT_PATTERN,   extracted_text)))[:5]
    icd10_codes = list(set(_re.findall(ICD10_PATTERN, extracted_text)))[:5]

    context = dict(base_policy)

    if cpt_codes:
        context["extracted_cpt_codes"]   = cpt_codes
    if icd10_codes:
        context["extracted_icd10_codes"] = icd10_codes
    if extracted_text:
        context["clinical_notes"] = extracted_text[:2000]

    return context, cpt_codes, icd10_codes


# ── Reasoning builder ─────────────────────────────────────────────────────────

def _build_reasoning(
    routing: dict,
    source: str,
    extraction_quality: Optional[dict] = None,
) -> str:
    """
    Produce a plain-English reasoning statement that explains how reliable
    the ML routing decision is and how it was reached.
    """
    confidence   = routing["confidence"]
    tier         = routing["tier"]
    approve_prob = routing["approve_probability"]
    deny_prob    = routing["deny_probability"]

    # Confidence band
    if confidence >= 0.90:
        confidence_label = "HIGH"
        reliability_note = (
            "The result carries very high reliability and is suitable for "
            "automated processing without additional review."
        )
    elif confidence >= 0.75:
        confidence_label = "MODERATE"
        reliability_note = (
            "The result carries moderate reliability.  A brief human "
            "spot-check is advisable before acting on this decision."
        )
    else:
        confidence_label = "LOW"
        reliability_note = (
            "Confidence is below the auto-routing threshold.  This case "
            "has been escalated for manual clinical review."
        )

    # Routing tier phrase
    tier_phrases = {
        "auto_approve":  "automatically approved",
        "auto_deny":     "automatically denied",
        "manual_review": "escalated for manual clinical review",
    }
    tier_text = tier_phrases.get(tier, tier)

    parts = [
        f"Decision confidence: {confidence_label} ({confidence * 100:.1f}%).",
        f"This PA request was {tier_text}.",
        f"Approval probability: {approve_prob * 100:.1f}% | "
        f"Denial probability: {deny_prob * 100:.1f}%.",
        reliability_note,
    ]

    # Source context
    if source == "file_upload" and extraction_quality:
        chars = extraction_quality.get("chars_extracted", 0)
        parts.append(
            f"The uploaded clinical document was successfully parsed "
            f"({chars:,} characters extracted) and contributed to the "
            f"validation decision."
        )
    elif source == "auto_generated":
        parts.append(
            "This result is based on a synthetically generated PA request "
            "and is intended for demonstration and testing purposes only."
        )

    return "  ".join(parts)


# ── PDF generators ────────────────────────────────────────────────────────────

def _pdf_styles():
    """Return a dict of shared ReportLab ParagraphStyles."""
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib        import colors

    styles = getSampleStyleSheet()
    return {
        "label": ParagraphStyle(
            "PALabel", parent=styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#555555"),
        ),
        "title": ParagraphStyle(
            "PATitle", parent=styles["Title"],
            fontSize=18, textColor=colors.HexColor("#1a3a5c"), spaceAfter=4,
        ),
        "heading": ParagraphStyle(
            "PAHeading", parent=styles["Heading2"],
            fontSize=11, textColor=colors.HexColor("#1a3a5c"),
            spaceBefore=12, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "PABody", parent=styles["Normal"], fontSize=10, leading=14,
        ),
        "small": ParagraphStyle(
            "PASmall", parent=styles["Normal"],
            fontSize=8, textColor=colors.HexColor("#777777"), leading=11,
        ),
    }


def _table_style(header_bg=None):
    """Return a standard TableStyle for data tables."""
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle

    rows = [
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",     (0, 0), (0, -1), colors.HexColor("#555555")),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.HexColor("#f5f7fa"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]
    if header_bg:
        rows.insert(0, ("BACKGROUND", (0, 0), (-1, 0), header_bg))
    return TableStyle(rows)


def generate_policy_pdf_bytes(record) -> bytes:
    """
    Render a formatted PDF for a policy record (generated or uploaded).

    For generated policies:  builds from coverage_rules stored in the DB.
    For uploaded policies:   this is not called — the original file is served.

    Returns raw PDF bytes ready to stream as a Response.
    """
    from reportlab.lib.pagesizes  import A4
    from reportlab.platypus       import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib            import colors
    from reportlab.lib.units      import cm

    buf    = io.BytesIO()
    st     = _pdf_styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title=f"Policy {record.policy_id}",
        author="PA Validation System",
    )

    story = []

    # Header
    story.append(Paragraph("PA Validation System", st["label"]))
    story.append(Paragraph("Insurance Policy Document", st["title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3a5c")))
    story.append(Spacer(1, 0.3*cm))

    # Policy metadata
    created = (
        record.created_at.strftime("%Y-%m-%d %H:%M UTC")
        if record.created_at else "N/A"
    )
    meta_data = [
        ["Policy ID",      record.policy_id],
        ["Insurer",        record.insurer or "N/A"],
        ["Effective Date", record.effective_date or "N/A"],
        ["Version",        record.version or "N/A"],
        ["Source",         record.source.title()],
        ["Rules Count",    str(record.num_rules)],
        ["Created At",     created],
    ]
    meta_table = Table(meta_data, colWidths=[4*cm, 12*cm])
    meta_table.setStyle(_table_style())
    story.append(meta_table)
    story.append(Spacer(1, 0.5*cm))

    # All coverage rules section — all 5 rules, full detail
    story.append(Paragraph("Coverage Rules — All Procedures", st["heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
    story.append(Spacer(1, 0.2*cm))

    coverage_rules = record.coverage_rules or []
    if not coverage_rules:
        story.append(Paragraph("No coverage rules defined.", st["body"]))
    else:
        for i, rule in enumerate(coverage_rules, 1):
            prereqs     = rule.get("prerequisites") or []
            prereq_text = "\n".join(f"• {p}" for p in prereqs) if prereqs else "None"

            story.append(Paragraph(
                f"Rule {i}:  {rule.get('procedure_name', 'N/A')}  "
                f"(CPT: {rule.get('cpt_code', 'N/A')})",
                st["heading"],
            ))

            rule_data = [
                ["Category",          rule.get("category", "N/A")],
                ["Requires PA",       "Yes" if rule.get("requires_prior_auth") else "No"],
                ["Estimated Cost",    rule.get("estimated_cost", "N/A")],
                ["Coverage Criteria", rule.get("coverage_criteria", "N/A")],
                ["Prerequisites",     prereq_text],
            ]

            rule_table = Table(rule_data, colWidths=[4*cm, 12*cm])
            rule_table.setStyle(_table_style())
            story.append(rule_table)
            story.append(Spacer(1, 0.3*cm))

    # Disclaimer footer
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "This document was auto-generated by the PA Validation System for testing and "
        "demonstration purposes only.  It does not constitute a legally binding insurance policy.  "
        "PA Validation System  |  Queen Mary University of London  |  2026",
        st["small"],
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def generate_validation_report_bytes(record) -> bytes:
    """
    Render a formatted PDF validation result report for a submission.

    Includes:
      - result_id highlighted prominently (so users know the right ID)
      - Colour-coded routing decision banner
      - All policy coverage rules evaluated (all 5 CPT blocks)
      - Per-rule PA requirement status for the submitted procedure
      - Patient, procedure, diagnosis, reasoning

    Used by GET /api/download/request/{result_id}.
    Returns raw PDF bytes.
    """
    from reportlab.lib.pagesizes  import A4
    from reportlab.platypus       import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib            import colors
    from reportlab.lib.units      import cm

    buf = io.BytesIO()
    st  = _pdf_styles()

    # Decision colours per tier
    tier_colors = {
        "auto_approve":  colors.HexColor("#1a7a3e"),
        "auto_deny":     colors.HexColor("#a31b1b"),
        "manual_review": colors.HexColor("#8a6800"),
    }
    tier_backgrounds = {
        "auto_approve":  colors.HexColor("#eaf7ef"),
        "auto_deny":     colors.HexColor("#fceaea"),
        "manual_review": colors.HexColor("#fdf8e6"),
    }
    tier_labels = {
        "auto_approve":  "AUTO-APPROVE  —  APPROVED",
        "auto_deny":     "AUTO-DENY  —  DENIED",
        "manual_review": "MANUAL REVIEW  —  PENDING HUMAN REVIEW",
    }

    tier_color     = tier_colors.get(record.routing_tier, colors.black)
    tier_bg        = tier_backgrounds.get(record.routing_tier, colors.white)
    decision_label = tier_labels.get(record.routing_tier, record.routing_action)

    decision_style = _pdf_styles()["heading"].__class__(
        "PADecision", parent=_pdf_styles()["heading"],
        fontSize=13, textColor=tier_color, spaceAfter=0, spaceBefore=0,
    )
    # Rebuild with correct parent
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    styles_base = getSampleStyleSheet()
    decision_style = ParagraphStyle(
        "PADecision", parent=styles_base["Heading2"],
        fontSize=13, textColor=tier_color, spaceAfter=0, spaceBefore=0,
    )

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title=f"Validation Result — {record.request_id}",
        author="PA Validation System",
    )

    story = []

    # Header
    story.append(Paragraph("PA Validation System", st["label"]))
    story.append(Paragraph("Prior Authorisation Validation Report", st["title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3a5c")))
    story.append(Spacer(1, 0.3*cm))

    # Reference IDs — result_id first and highlighted
    ref_data = [
        ["result_id  (use for PDF download)", str(record.id)],
        ["Request ID",                        record.request_id or "N/A"],
        ["Submitted",                         record.timestamp or "N/A"],
        ["Source",                            record.source.replace("_", " ").title()],
        ["Insurer",                           record.insurer or "N/A"],
    ]
    ref_table = Table(ref_data, colWidths=[5.5*cm, 10.5*cm])
    ref_ts = _table_style()
    # Highlight result_id row
    from reportlab.platypus import TableStyle as TS
    ref_table.setStyle(TS(list(ref_ts._cmds) + [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0fb")),
        ("TEXTCOLOR",  (1, 0), (1, 0),  colors.HexColor("#1a3a5c")),
        ("FONTNAME",   (1, 0), (1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",   (1, 0), (1, 0),  12),
    ]))
    story.append(ref_table)
    story.append(Spacer(1, 0.5*cm))

    # Decision banner
    story.append(Paragraph("Routing Decision", st["heading"]))
    decision_table = Table(
        [[Paragraph(decision_label, decision_style)]],
        colWidths=[16*cm],
    )
    decision_table.setStyle(TS([
        ("BACKGROUND",    (0, 0), (-1, -1), tier_bg),
        ("BOX",           (0, 0), (-1, -1), 1.5, tier_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
    ]))
    story.append(decision_table)
    story.append(Spacer(1, 0.3*cm))

    # Probability scores
    prob_data = [
        ["Confidence",           f"{record.confidence * 100:.1f}%"],
        ["Approval Probability", f"{record.approve_probability * 100:.1f}%"],
        ["Denial Probability",   f"{record.deny_probability * 100:.1f}%"],
        ["ML Explanation",       record.explanation or "N/A"],
    ]
    prob_table = Table(prob_data, colWidths=[5.5*cm, 10.5*cm])
    prob_table.setStyle(_table_style())
    story.append(prob_table)
    story.append(Spacer(1, 0.3*cm))

    # Reasoning
    stored_routing = record.full_routing_json or {}
    reasoning_text = _build_reasoning(stored_routing, source=record.source or "auto_generated")
    story.append(Paragraph("Reasoning", st["heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(reasoning_text, st["body"]))

    # Patient information
    story.append(Paragraph("Patient Information", st["heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
    story.append(Spacer(1, 0.15*cm))
    patient_data = [
        ["Name",   record.patient_name   or "N/A"],
        ["Age",    str(record.patient_age) if record.patient_age else "N/A"],
        ["Gender", record.patient_gender  or "N/A"],
    ]
    patient_table = Table(patient_data, colWidths=[5.5*cm, 10.5*cm])
    patient_table.setStyle(_table_style())
    story.append(patient_table)
    story.append(Spacer(1, 0.3*cm))

    # Requested procedure + diagnosis
    story.append(Paragraph("Requested Procedure & Diagnosis", st["heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
    story.append(Spacer(1, 0.15*cm))
    proc_data = [
        ["CPT Code",   record.cpt_code           or "N/A"],
        ["Procedure",  record.procedure_name      or "N/A"],
        ["Category",   record.procedure_category  or "N/A"],
        ["ICD-10",     record.icd10_code          or "N/A"],
        ["Diagnosis",  record.diagnosis_desc      or "N/A"],
    ]
    proc_table = Table(proc_data, colWidths=[5.5*cm, 10.5*cm])
    proc_table.setStyle(_table_style())
    story.append(proc_table)
    story.append(Spacer(1, 0.3*cm))

    # ── All policy coverage rules evaluated ───────────────────────────────
    # Pull the full coverage_rules list from full_request_json (which contains
    # the policy context passed to generate_pa_request).  This gives us all
    # 5 CPT blocks, not just the single matched one.
    stored_request = record.full_request_json or {}
    policy_rules   = stored_request.get("coverage_rules", [])
    requested_cpt  = record.cpt_code

    if policy_rules:
        story.append(Paragraph(
            f"Policy Coverage Rules Evaluated ({len(policy_rules)} total)",
            st["heading"],
        ))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
        story.append(Spacer(1, 0.15*cm))

        for i, rule in enumerate(policy_rules, 1):
            cpt       = rule.get("cpt_code", "N/A")
            proc_name = rule.get("procedure_name", "N/A")
            cat       = rule.get("category", "N/A")
            req_pa    = "Yes" if rule.get("requires_prior_auth") else "No"
            cost      = rule.get("estimated_cost", "N/A")
            prereqs   = rule.get("prerequisites") or []
            prereq_str= "\n".join(f"• {p}" for p in prereqs) if prereqs else "None required"

            # Mark which rule was the one matched to this PA request
            is_matched = (cpt == requested_cpt)
            rule_label = (
                f"Rule {i}: {proc_name}  (CPT: {cpt})  ← MATCHED TO THIS REQUEST"
                if is_matched else
                f"Rule {i}: {proc_name}  (CPT: {cpt})"
            )
            story.append(Paragraph(rule_label, st["heading"]))

            rule_rows = [
                ["Category",     cat],
                ["Requires PA",  req_pa],
                ["Cost Est.",    cost],
                ["Prerequisites",prereq_str],
            ]
            rt = Table(rule_rows, colWidths=[4*cm, 12*cm])
            extra = []
            if is_matched:
                extra = [("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffde8"))]
            rt.setStyle(TS(list(_table_style()._cmds) + extra))
            story.append(rt)
            story.append(Spacer(1, 0.3*cm))

    # Uploaded document note
    if record.source == "file_upload" and record.uploaded_filename:
        story.append(Paragraph("Uploaded Clinical Document", st["heading"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
        story.append(Spacer(1, 0.15*cm))
        story.append(Paragraph(f"Filename: {record.uploaded_filename}", st["body"]))

    # Footer
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "DISCLAIMER: This report is system-generated by an ML pipeline. "
        "For clinical or administrative decisions always consult a qualified healthcare "
        "professional or authorised reviewer.  "
        "PA Validation System  |  Queen Mary University of London  |  2026",
        st["small"],
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# Alias kept for any code that imports the old name
def get_result_report_bytes(record) -> bytes:
    """Alias for generate_validation_report_bytes — returns PDF bytes."""
    return generate_validation_report_bytes(record)


# ── Statistics helper ─────────────────────────────────────────────────────────

async def live_stats(db: AsyncSession) -> dict:
    """Calculate live routing statistics from the database."""
    total = (await db.execute(
        select(func.count(ValidationResult.id))
    )).scalar() or 0

    auto_approved = (await db.execute(
        select(func.count(ValidationResult.id))
        .where(ValidationResult.routing_tier == "auto_approve")
    )).scalar() or 0

    auto_denied = (await db.execute(
        select(func.count(ValidationResult.id))
        .where(ValidationResult.routing_tier == "auto_deny")
    )).scalar() or 0

    manual_review = (await db.execute(
        select(func.count(ValidationResult.id))
        .where(ValidationResult.routing_tier == "manual_review")
    )).scalar() or 0

    return {
        "total_requests": total,
        "auto_approved":  auto_approved,
        "auto_denied":    auto_denied,
        "manual_review":  manual_review,
    }


# ── Core submission logic ─────────────────────────────────────────────────────

async def _get_latest_policy_context(db: AsyncSession) -> dict:
    """Fetch the most recently created policy. Raises HTTP 400 when none exist."""
    result = await db.execute(
        select(Policy).order_by(Policy.created_at.desc()).limit(1)
    )
    latest = result.scalar_one_or_none()

    if latest is None:
        raise HTTPException(
            status_code=400,
            detail="No policies found. Generate or upload a policy first.",
        )

    return {
        "policy_id":      latest.policy_id,
        "insurer":        latest.insurer,
        "coverage_rules": latest.coverage_rules,   # all 5 rules included
    }


async def submit_autogenerate(
    db:         AsyncSession,
    likelihood: str,
    base_url:   str,
) -> dict:
    """
    Mode 1 — Generate a synthetic PA request, run ML pipeline, persist result.

    The policy context passed to the generator includes all coverage_rules so
    they are stored in full_request_json and appear in the downloaded PDF report.
    """
    if router is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run: python src/ml/train_model_simple.py",
        )

    policy_context = await _get_latest_policy_context(db)

    # generate_pa_request picks ONE rule from the policy for the PA request
    # (the one with requires_prior_auth=True) but we pass the full policy so
    # all coverage_rules end up in the persisted full_request_json
    pa_request = generator.generate_pa_request(
        policy=policy_context,
        approval_likelihood=likelihood,
    )

    # Embed the full policy coverage_rules into the request for audit/reporting
    pa_request["coverage_rules"] = policy_context["coverage_rules"]

    features  = engineer.create_features(pa_request, get_policy_rules())
    routing   = router.route_case(features)
    reasoning = _build_reasoning(routing, source="auto_generated")

    record = await _persist_result(
        db, pa_request, routing, source="auto_generated",
        uploaded_filename=None, uploaded_file_path=None,
    )

    stats        = await live_stats(db)
    download_url = f"{base_url.rstrip('/')}/api/download/request/{record.id}"

    return {
        "success":            True,
        "result_id":          record.id,
        "download_url":       download_url,
        "routing":            routing,
        "reasoning":          reasoning,
        "request":            pa_request,
        "features":           features,
        "stats":              stats,
        "record_id":          record.id,   # backward-compatibility alias
        "extraction_status":  None,
        "extraction_warning": None,
    }


async def submit_file_upload(
    db:         AsyncSession,
    file:       UploadFile,
    likelihood: str,
    base_url:   str,
) -> dict:
    """
    Mode 2 — Upload a clinical document, extract its data, run ML pipeline.

    The full policy coverage_rules are embedded in the request so all 5 CPT
    blocks appear in the downloaded validation result PDF.
    """
    if router is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run: python src/ml/train_model_simple.py",
        )

    if not _allowed_file(file.filename or ""):
        raise HTTPException(
            status_code=400,
            detail=(
                f"File type not allowed.  "
                f"Accepted formats: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}.  "
                "Please upload a PDF, DOCX, or TXT file containing clinical data."
            ),
        )

    # Read file bytes once — reuse for extraction and disk save
    file_bytes    = await file.read()
    file.file     = io.BytesIO(file_bytes)
    original_name = file.filename or "upload"

    # Extract and assess before saving — reject garbage before touching disk
    extracted_text = _extract_text_from_file(file_bytes, original_name)
    policy_context = await _get_latest_policy_context(db)
    clinical_context, cpt_codes, icd10_codes = _build_clinical_context_from_text(
        extracted_text, policy_context
    )
    quality = _assess_extraction_quality(
        extracted_text, cpt_codes, icd10_codes, original_name
    )

    if quality["extraction_status"] == "failed":
        raise HTTPException(
            status_code=422,
            detail={
                "error":              "unreadable_file",
                "extraction_status":  "failed",
                "message":            quality["extraction_warning"],
                "guidance": (
                    "Upload a text-selectable PDF, DOCX, or TXT file that "
                    "contains clinical prior-authorisation data."
                ),
            },
        )

    if quality["extraction_status"] == "partial":
        raise HTTPException(
            status_code=422,
            detail={
                "error":              "irrelevant_file",
                "extraction_status":  "partial",
                "message":            quality["extraction_warning"],
                "guidance": (
                    "Please upload a clinical PA document containing valid CPT "
                    "procedure codes and/or ICD-10 diagnosis codes."
                ),
            },
        )

    # File is valid — save it to disk now
    file.file = io.BytesIO(file_bytes)
    original_name, saved_path = await _save_upload(file, subfolder="requests")

    print(
        f"[upload] Extracted {quality['chars_extracted']:,} chars, "
        f"CPT: {cpt_codes}, ICD-10: {icd10_codes} from '{original_name}'"
    )

    pa_request = generator.generate_pa_request(
        policy=clinical_context,
        approval_likelihood=likelihood,
    )
    pa_request["source"]                 = "file_upload"
    pa_request["filename"]               = original_name
    pa_request["extracted_text_preview"] = (
        (extracted_text[:300] + "…") if extracted_text else None
    )

    # Embed the full policy coverage_rules for audit and PDF reporting
    pa_request["coverage_rules"] = policy_context["coverage_rules"]

    features  = engineer.create_features(pa_request, get_policy_rules())
    routing   = router.route_case(features)
    reasoning = _build_reasoning(routing, source="file_upload", extraction_quality=quality)

    record = await _persist_result(
        db, pa_request, routing, source="file_upload",
        uploaded_filename=original_name, uploaded_file_path=saved_path,
    )

    stats        = await live_stats(db)
    download_url = f"{base_url.rstrip('/')}/api/download/request/{record.id}"

    return {
        "success":            True,
        "result_id":          record.id,
        "download_url":       download_url,
        "routing":            routing,
        "reasoning":          reasoning,
        "request":            pa_request,
        "features":           features,
        "stats":              stats,
        "record_id":          record.id,
        "extraction_status":  quality["extraction_status"],
        "extraction_warning": quality["extraction_warning"],
        "chars_extracted":    quality["chars_extracted"],
        "cpt_codes_found":    cpt_codes,
        "icd10_codes_found":  icd10_codes,
    }


async def _persist_result(
    db:                 AsyncSession,
    pa_request:         dict,
    routing:            dict,
    source:             str,
    uploaded_filename:  Optional[str],
    uploaded_file_path: Optional[str],
) -> ValidationResult:
    """Write a ValidationResult row and return the refreshed ORM object."""
    patient = pa_request.get("patient", {})
    proc    = pa_request.get("requested_procedure", {})
    diag    = pa_request.get("diagnosis", {})

    record = ValidationResult(
        request_id          = pa_request.get("request_id", ""),
        timestamp           = pa_request.get("timestamp", ""),
        insurer             = pa_request.get("insurer", ""),
        patient_name        = patient.get("name"),
        patient_age         = patient.get("age"),
        patient_gender      = patient.get("gender"),
        cpt_code            = proc.get("cpt_code"),
        procedure_name      = proc.get("name"),
        procedure_category  = proc.get("category"),
        icd10_code          = diag.get("icd10_code"),
        diagnosis_desc      = diag.get("description"),
        routing_tier        = routing["tier"],
        routing_action      = routing["action"],
        routing_color       = routing["color"],
        confidence          = routing["confidence"],
        approve_probability = routing["approve_probability"],
        deny_probability    = routing["deny_probability"],
        explanation         = routing["explanation"],
        source              = source,
        uploaded_filename   = uploaded_filename,
        uploaded_file_path  = uploaded_file_path,
        # full_request_json now contains coverage_rules (all 5 CPT blocks)
        full_request_json   = pa_request,
        full_routing_json   = routing,
    )

    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


# ── History ───────────────────────────────────────────────────────────────────

async def list_submissions(
    db:       AsyncSession,
    base_url: str,
    limit:    int = 200,
    tier:     Optional[str] = None,
) -> list:
    """Return stored validation results, newest first, with download_url."""
    limit  = min(limit, 1000)
    query  = select(ValidationResult).order_by(ValidationResult.created_at.desc())

    if tier:
        query = query.where(ValidationResult.routing_tier == tier)

    result  = await db.execute(query.limit(limit))
    records = result.scalars().all()

    rows = []
    for r in records:
        d = r.to_dict()
        d["download_url"] = f"{base_url.rstrip('/')}/api/download/request/{r.id}"
        rows.append(d)

    return rows


async def get_result_or_404(db: AsyncSession, result_id: int) -> ValidationResult:
    """Fetch a ValidationResult by DB primary key or raise HTTP 404."""
    result = await db.execute(
        select(ValidationResult).where(ValidationResult.id == result_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No submission found with result_id={result_id}.  "
                "Use the 'result_id' value from the /api/submit-request response, "
                "not the request_id (PA-XXXXXX) string field."
            ),
        )
    return record


async def reset_all_results(db: AsyncSession) -> None:
    """Delete all ValidationResult rows. FOR DEMO / TESTING ONLY."""
    from sqlalchemy import delete
    await db.execute(delete(ValidationResult))
    await db.commit()