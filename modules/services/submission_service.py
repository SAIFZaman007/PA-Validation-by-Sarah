"""
Submission service — business logic for PA request validation.

Single mode: Upload a prescription PDF, select an insurer (policy is auto-
loaded), cross-check prescription against policy, run ML pipeline, return
a detailed validation report with download link.

The prediction is deterministic for the same prescription + policy pair
(same features → same model output). Reasoning is generated from the
stored routing result, so downloading the PDF later gives the same content.
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
    """Save uploaded file with UUID prefix. Returns (original_name, abs_path)."""
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
    """Extract plain text from PDF / TXT / DOCX. Returns empty string on failure."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join((page.extract_text() or "") for page in pdf.pages)
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
    chars = len(extracted_text.strip())

    if chars == 0:
        return {
            "extraction_status":  "failed",
            "extraction_warning": (
                f"No readable text could be extracted from '{filename}'. "
                "The file may be scanned, image-only, or password-protected. "
                "Please upload a text-selectable PDF, DOCX, or TXT file."
            ),
            "chars_extracted": 0,
        }

    if not cpt_codes and not icd10_codes:
        return {
            "extraction_status":  "partial",
            "extraction_warning": (
                f"Text was extracted from '{filename}' ({chars:,} characters), "
                "but no CPT procedure codes or ICD-10 diagnosis codes were found. "
                "This document does not appear to be a clinical prior-authorisation record."
            ),
            "chars_extracted": chars,
        }

    return {
        "extraction_status":  "success",
        "extraction_warning": None,
        "chars_extracted":    chars,
    }


def _build_clinical_context_from_text(extracted_text: str, base_policy: dict) -> tuple:
    """Enrich the context dict with CPT/ICD-10 codes extracted from the document."""
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
    policy_insurer: Optional[str] = None,
) -> str:
    """
    Produce a comprehensive industry-standard reasoning statement explaining
    the ML routing decision with clinical rationale, confidence metrics, and
    actionable next steps.
    """
    confidence   = routing.get("confidence", 0.5)
    tier         = routing.get("tier", "manual_review")
    approve_prob = routing.get("approve_probability", 0.5)
    deny_prob    = routing.get("deny_probability", 0.5)
    action       = routing.get("action", "REVIEW").upper()

    if confidence >= 0.90:
        confidence_tier, reliability_class = "VERY HIGH", "CONFIDENCE_A"
        recommendation = "suitable for autonomous processing"
    elif confidence >= 0.80:
        confidence_tier, reliability_class = "HIGH", "CONFIDENCE_B"
        recommendation = "recommended for priority automated processing with standard audit trail"
    elif confidence >= 0.70:
        confidence_tier, reliability_class = "MODERATE-HIGH", "CONFIDENCE_C"
        recommendation = "suitable for processing pending brief supervisor validation"
    elif confidence >= 0.60:
        confidence_tier, reliability_class = "MODERATE", "CONFIDENCE_D"
        recommendation = "recommended for clinical team review before processing"
    else:
        confidence_tier, reliability_class = "LOW", "CONFIDENCE_E"
        recommendation = "requires full clinical review and physician evaluation"

    decision_statement = f"This PA request has been {action.lower()} by the system."

    if tier == "auto_approve":
        clinical_basis = (
            f"The prescription meets established coverage criteria with high clinical index. "
            f"Approval probability score: {approve_prob * 100:.1f}%. "
            f"Clinical indicators support medical necessity for the requested procedure."
        )
    elif tier == "auto_deny":
        clinical_basis = (
            f"The prescription does not meet established coverage criteria or clinical guidelines. "
            f"Denial probability score: {deny_prob * 100:.1f}%. "
            f"Clinical indicators suggest insufficient justification for the requested procedure."
        )
    else:
        clinical_basis = (
            f"The prescription presents mixed clinical indicators requiring expert judgment. "
            f"Approval probability: {approve_prob * 100:.1f}% | Denial probability: {deny_prob * 100:.1f}%. "
            f"Recommend clinical review to evaluate nuanced factors and patient-specific circumstances."
        )

    confidence_statement = (
        f"Confidence: {confidence_tier} ({confidence * 100:.2f}%, "
        f"Classification: {reliability_class}). "
        f"Result {recommendation}."
    )

    insurer_note = f" Policy insurer: {policy_insurer}." if policy_insurer else ""

    if source == "prescription_upload":
        quality_note = (
            f"Request Type: Prescription PDF Upload.{insurer_note} "
            f"Prescription was cross-checked against the selected insurer's active policy. "
            f"Decision is based on extracted clinical codes and policy coverage rules."
        )
        if extraction_quality:
            chars = extraction_quality.get("chars_extracted", 0)
            status = extraction_quality.get("extraction_status", "unknown")
            quality_note += (
                f" Document extraction: {status.upper()} ({chars:,} characters extracted)."
            )
    else:
        quality_note = (
            f"Request Type: Auto-Generated (Demonstration).{insurer_note} "
            f"This result is based on synthetic PA request data for testing purposes."
        )

    next_steps = {
        "auto_approve": "APPROVED FOR PROCESSING — Proceed with care authorisation per coverage plan terms.",
        "auto_deny":    "DENIED — Refer to patient/provider with specific denial rationale and appeal process information.",
        "manual_review":"ESCALATED FOR CLINICAL REVIEW — Assign to medical reviewer within 24 hours.",
    }
    next_step_text = next_steps.get(tier, "REVIEW REQUIRED — Assign for clinical assessment.")

    parts = [
        f"\n DECISION: {decision_statement}",
        f"\n CLINICAL BASIS: {clinical_basis}",
        f"\n CONFIDENCE: {confidence_statement}",
        f"\n DATA QUALITY: {quality_note}",
        f"\n RECOMMENDED ACTION: {next_step_text}",
    ]
    return " ".join(parts)


# ── PDF generators ────────────────────────────────────────────────────────────

def _pdf_styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib        import colors
    styles = getSampleStyleSheet()
    return {
        "label": ParagraphStyle("PALabel", parent=styles["Normal"],
                                fontSize=9, textColor=colors.HexColor("#555555")),
        "title": ParagraphStyle("PATitle", parent=styles["Title"],
                                fontSize=18, textColor=colors.HexColor("#1a3a5c"), spaceAfter=4),
        "heading": ParagraphStyle("PAHeading", parent=styles["Heading2"],
                                  fontSize=11, textColor=colors.HexColor("#1a3a5c"),
                                  spaceBefore=12, spaceAfter=4),
        "body":  ParagraphStyle("PABody", parent=styles["Normal"], fontSize=10, leading=14),
        "small": ParagraphStyle("PASmall", parent=styles["Normal"],
                                fontSize=8, textColor=colors.HexColor("#777777"), leading=11),
    }


def _table_style(header_bg=None):
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
    Render a formatted PDF for a policy record.

    For generated policies: builds from coverage_rules stored in the DB.
    For uploaded policies:  this is not called — the original file is served.
    """
    from reportlab.lib.pagesizes  import A4
    from reportlab.platypus       import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib            import colors
    from reportlab.lib.units      import cm

    buf = io.BytesIO()
    st  = _pdf_styles()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title=f"Policy {record.policy_id}", author="PA Validation System",
    )

    story = []
    story.append(Paragraph("PA Validation System", st["label"]))
    story.append(Paragraph("Insurance Policy Document", st["title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3a5c")))
    story.append(Spacer(1, 0.3*cm))

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

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "This document was auto-generated by the PA Validation System for testing and "
        "demonstration purposes only. It does not constitute a legally binding insurance policy. "
        "PA Validation System  |  Queen Mary University of London  |  2026",
        st["small"],
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def generate_validation_report_bytes(record) -> bytes:
    """
    Render a formatted PDF validation result report.

    Includes: result_id, routing decision banner, probability scores, reasoning,
    patient/procedure/diagnosis data, all policy rules evaluated.
    """
    from reportlab.lib.pagesizes  import A4
    from reportlab.platypus       import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib            import colors
    from reportlab.lib.units      import cm
    from reportlab.platypus       import TableStyle as TS

    buf = io.BytesIO()
    st  = _pdf_styles()

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

    styles_base    = getSampleStyleSheet()
    decision_style = ParagraphStyle(
        "PADecision", parent=styles_base["Heading2"],
        fontSize=13, textColor=tier_color, spaceAfter=0, spaceBefore=0,
    )

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
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

    # Reference block
    ref_data = [
        ["result_id  (use for PDF download)", str(record.id)],
        ["Request ID",                        record.request_id or "N/A"],
        ["Submitted",                         record.timestamp or "N/A"],
        ["Source",                            record.source.replace("_", " ").title()],
        ["Insurer",                           record.insurer or "N/A"],
    ]
    if record.policy_id_used:
        ref_data.append(["Policy Used",  record.policy_id_used])
    if record.uploaded_filename:
        ref_data.append(["Uploaded File", record.uploaded_filename])

    ref_table = Table(ref_data, colWidths=[5.5*cm, 10.5*cm])
    ref_table.setStyle(TS(list(_table_style()._cmds) + [
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
    reasoning_text = _build_reasoning(
        stored_routing,
        source=record.source or "prescription_upload",
        policy_insurer=record.policy_insurer,
    )
    story.append(Paragraph("Reasoning", st["heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(reasoning_text, st["body"]))

    # Patient
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

    # Procedure + Diagnosis
    story.append(Paragraph("Requested Procedure & Diagnosis", st["heading"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#aaaaaa")))
    story.append(Spacer(1, 0.15*cm))
    proc_data = [
        ["CPT Code",  record.cpt_code           or "N/A"],
        ["Procedure", record.procedure_name      or "N/A"],
        ["Category",  record.procedure_category  or "N/A"],
        ["ICD-10",    record.icd10_code          or "N/A"],
        ["Diagnosis", record.diagnosis_desc      or "N/A"],
    ]
    proc_table = Table(proc_data, colWidths=[5.5*cm, 10.5*cm])
    proc_table.setStyle(_table_style())
    story.append(proc_table)
    story.append(Spacer(1, 0.3*cm))

    # All policy coverage rules
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

            is_matched = (cpt == requested_cpt)
            rule_label = (
                f"Rule {i}: {proc_name}  (CPT: {cpt})  ← MATCHED TO THIS REQUEST"
                if is_matched else
                f"Rule {i}: {proc_name}  (CPT: {cpt})"
            )
            story.append(Paragraph(rule_label, st["heading"]))

            rule_rows = [
                ["Category",      cat],
                ["Requires PA",   req_pa],
                ["Cost Est.",     cost],
                ["Prerequisites", prereq_str],
            ]
            rt = Table(rule_rows, colWidths=[4*cm, 12*cm])
            extra = [("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffde8"))] if is_matched else []
            rt.setStyle(TS(list(_table_style()._cmds) + extra))
            story.append(rt)
            story.append(Spacer(1, 0.3*cm))

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


# Backward-compatible alias
def get_result_report_bytes(record) -> bytes:
    return generate_validation_report_bytes(record)


# ── Statistics ────────────────────────────────────────────────────────────────

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

    # Insurer breakdown
    from sqlalchemy import text as sa_text
    insurer_rows = (await db.execute(
        select(ValidationResult.insurer, func.count(ValidationResult.id))
        .group_by(ValidationResult.insurer)
        .order_by(func.count(ValidationResult.id).desc())
    )).all()

    by_insurer = [{"insurer": row[0] or "Unknown", "count": row[1]} for row in insurer_rows]

    return {
        "total_requests": total,
        "auto_approved":  auto_approved,
        "auto_denied":    auto_denied,
        "manual_review":  manual_review,
        "by_insurer":     by_insurer,
    }


# ── Policy lookup ─────────────────────────────────────────────────────────────

async def _get_policy_by_insurer(db: AsyncSession, insurer: str) -> Policy:
    """
    Fetch the most recent policy for the given insurer.
    Raises HTTP 400 when no policy exists for that insurer.
    """
    result = await db.execute(
        select(Policy)
        .where(Policy.insurer == insurer)
        .order_by(Policy.created_at.desc())
        .limit(1)
    )
    policy = result.scalar_one_or_none()

    if policy is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No policy found for insurer '{insurer}'. "
                f"Please generate or upload a policy for this insurer first via "
                f"POST /api/generate-policy or POST /api/upload-policy."
            ),
        )
    return policy


async def _get_all_insurers_with_policies(db: AsyncSession) -> list:
    """Return distinct insurers that have at least one policy in the DB."""
    result = await db.execute(
        select(Policy.insurer).distinct().order_by(Policy.insurer)
    )
    return [row[0] for row in result.all()]


# ── Core submission logic ─────────────────────────────────────────────────────

async def submit_prescription_upload(
    db:       AsyncSession,
    file:     UploadFile,
    insurer:  str,
    base_url: str,
) -> dict:
    """
    Upload a prescription PDF, extract the exact patient and clinical data from it,
    cross-check against the selected insurer's policy, run the ML pipeline,
    and persist the result.

    The prescription data in the response and stored record always reflects the
    ACTUAL content of the uploaded file — not randomly generated data.
    The prediction is deterministic for the same prescription + policy pair.
    """
    if router is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run: python src/ml/train_model.py",
        )

    if not _allowed_file(file.filename or ""):
        raise HTTPException(
            status_code=400,
            detail=(
                f"File type not allowed. "
                f"Accepted formats: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}."
            ),
        )

    # Load the policy for the selected insurer
    policy_record  = await _get_policy_by_insurer(db, insurer)
    policy_context = {
        "policy_id":      policy_record.policy_id,
        "insurer":        policy_record.insurer,
        "coverage_rules": policy_record.coverage_rules,
    }

    # Read file bytes once
    file_bytes    = await file.read()
    file.file     = io.BytesIO(file_bytes)
    original_name = file.filename or "upload"

    extracted_text = _extract_text_from_file(file_bytes, original_name)

    # Use the structured prescription parser (same one used by upload-prescription)
    # so both routes always return identical patient/clinical data for the same file.
    from modules.services.prescription_service import (
        _parse_prescription_from_text,
        _fallback_cpt,
        _fallback_icd10,
    )
    parsed = _parse_prescription_from_text(extracted_text, insurer)

    # Extract CPT and ICD-10 for quality assessment (using parsed values, not raw regex)
    cpt_codes   = [parsed["cpt_code"]]   if parsed.get("cpt_code")   else []
    icd10_codes = [parsed["icd10_code"]] if parsed.get("icd10_code") else []

    quality = _assess_extraction_quality(
        extracted_text, cpt_codes, icd10_codes, original_name
    )

    if quality["extraction_status"] == "failed":
        raise HTTPException(
            status_code=422,
            detail={
                "error":             "unreadable_file",
                "extraction_status": "failed",
                "message":           quality["extraction_warning"],
                "guidance": (
                    "Upload a text-selectable PDF, DOCX, or TXT file containing "
                    "clinical prior-authorisation data."
                ),
            },
        )

    if quality["extraction_status"] == "partial":
        raise HTTPException(
            status_code=422,
            detail={
                "error":             "irrelevant_file",
                "extraction_status": "partial",
                "message":           quality["extraction_warning"],
                "guidance": (
                    "Upload a clinical PA document containing valid CPT procedure codes "
                    "and/or ICD-10 diagnosis codes."
                ),
            },
        )

    # Save file to disk after validation passes
    file.file = io.BytesIO(file_bytes)
    original_name, saved_path = await _save_upload(file, subfolder="requests")

    print(
        f"[submit] '{original_name}' → policy '{policy_record.policy_id}' ({insurer}) | "
        f"patient={parsed['patient_name']!r}, CPT={parsed['cpt_code']}, "
        f"ICD10={parsed['icd10_code']}, chars={quality['chars_extracted']:,}"
    )

    # Build the PA request dict from the ACTUAL parsed prescription data.
    # This replaces the old generator.generate_pa_request() call which was
    # producing completely random (and therefore wrong) patient information.
    import datetime as _dt
    pa_request = {
        "request_id": f"PA-{uuid.uuid4().hex[:6].upper()}",
        "timestamp":  _dt.datetime.now().isoformat(),
        "source":     "prescription_upload",
        "filename":   original_name,
        "insurer":    insurer,
        "patient": {
            "patient_id":   parsed.get("patient_id"),
            "name":         parsed["patient_name"],
            "age":          parsed["patient_age"],
            "gender":       parsed["patient_gender"],
            "insurance_id": parsed.get("insurance_id"),
        },
        "requested_procedure": {
            "cpt_code": parsed["cpt_code"],
            "name":     parsed["procedure_name"],
            "category": parsed["procedure_category"],
        },
        "diagnosis": {
            "icd10_code":  parsed["icd10_code"],
            "description": parsed["diagnosis_desc"],
        },
        "clinical_info": {
            "bmi":                                parsed.get("bmi"),
            "conservative_therapy_duration_weeks": parsed.get("conservative_therapy_weeks"),
            "imaging_completed":                  parsed.get("imaging_completed", False),
            "prerequisites_met":                  parsed.get("prerequisites_met", []),
        },
        "requesting_physician": {
            "name":           parsed.get("physician_name"),
            "specialty":      parsed.get("physician_specialty"),
            "license_number": parsed.get("physician_license"),
        },
        "coverage_rules":          policy_context["coverage_rules"],
        "extracted_text_preview":  (extracted_text[:300] + "…") if extracted_text else None,
    }

    # Get policy rules from cache (warmed at startup) or extract inline
    policy_rules = get_policy_rules()
    if not policy_rules:
        policy_rules = extractor.extract_policy_rules(
            {"coverage_rules": policy_context["coverage_rules"]}
        )

    features  = engineer.create_features(pa_request, policy_rules)
    routing   = router.route_case(features)
    reasoning = _build_reasoning(
        routing,
        source="prescription_upload",
        extraction_quality={
            **quality,
            "cpt_codes_found":   cpt_codes,
            "icd10_codes_found": icd10_codes,
        },
        policy_insurer=insurer,
    )

    record = await _persist_result(
        db, pa_request, routing,
        source="prescription_upload",
        uploaded_filename=original_name,
        uploaded_file_path=saved_path,
        policy_id_used=policy_record.policy_id,
        policy_insurer=insurer,
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
        "extraction_status":  quality["extraction_status"],
        "extraction_warning": quality["extraction_warning"],
        "chars_extracted":    quality["chars_extracted"],
        "cpt_codes_found":    cpt_codes,
        "icd10_codes_found":  icd10_codes,
        "policy_used": {
            "policy_id": policy_record.policy_id,
            "insurer":   policy_record.insurer,
            "num_rules": policy_record.num_rules,
        },
    }


async def _persist_result(
    db:                 AsyncSession,
    pa_request:         dict,
    routing:            dict,
    source:             str,
    uploaded_filename:  Optional[str],
    uploaded_file_path: Optional[str],
    policy_id_used:     Optional[str] = None,
    policy_insurer:     Optional[str] = None,
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
        policy_id_used      = policy_id_used,
        policy_insurer      = policy_insurer,
        source              = source,
        uploaded_filename   = uploaded_filename,
        uploaded_file_path  = uploaded_file_path,
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
                f"No submission found with result_id={result_id}. "
                "Use the 'result_id' value from the /api/submit-request response."
            ),
        )
    return record


async def reset_all_results(db: AsyncSession) -> None:
    """Delete all ValidationResult rows. FOR DEMO / TESTING ONLY."""
    from sqlalchemy import delete
    await db.execute(delete(ValidationResult))
    await db.commit()


async def get_available_insurers(db: AsyncSession) -> list:
    """Return list of insurers that have at least one policy in the DB."""
    result = await db.execute(
        select(Policy.insurer, Policy.policy_id, Policy.num_rules)
        .distinct(Policy.insurer)
        .order_by(Policy.insurer, Policy.created_at.desc())
    )
    rows = result.all()
    seen = {}
    for insurer, policy_id, num_rules in rows:
        if insurer not in seen:
            seen[insurer] = {"insurer": insurer, "policy_id": policy_id, "num_rules": num_rules}
    return list(seen.values())