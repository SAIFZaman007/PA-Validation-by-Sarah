"""
SQLAlchemy ORM models for the PA Validation System.

Tables:
  Policy            — Insurance policies (generated or uploaded).
  Prescription      — Patient prescriptions (generated or uploaded).
  ValidationResult  — ML audit record for every PA submission.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float,
    Integer, JSON, String, Text,
)

from core.database import Base


# ── Policy ────────────────────────────────────────────────────────────────────

class Policy(Base):

    __tablename__ = "policies"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    policy_id      = Column(String(64),  unique=True, nullable=False, index=True)
    insurer        = Column(String(128), nullable=False)
    effective_date = Column(String(32),  nullable=False, default="2024-01-01")
    version        = Column(String(32),  nullable=False, default="2024.1")

    # "generated" or "uploaded"
    source    = Column(String(32),  nullable=False, default="generated")
    filename  = Column(String(256), nullable=True)
    file_path = Column(String(512), nullable=True)

    coverage_rules = Column(JSON,    nullable=False, default=list)
    num_rules      = Column(Integer, nullable=False, default=0)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "policy_id":      self.policy_id,
            "insurer":        self.insurer,
            "effective_date": self.effective_date,
            "version":        self.version,
            "source":         self.source,
            "filename":       self.filename,
            "num_rules":      self.num_rules,
            "coverage_rules": self.coverage_rules,
            "created_at":     self.created_at.isoformat(),
        }

    def to_json_export(self) -> str:
        doc = {
            "policy_id":      self.policy_id,
            "insurer":        self.insurer,
            "effective_date": self.effective_date,
            "version":        self.version,
            "source":         self.source,
            "coverage_rules": self.coverage_rules,
            "exported_at":    datetime.now(timezone.utc).isoformat(),
        }
        return json.dumps(doc, indent=2, ensure_ascii=False)


# ── Prescription ──────────────────────────────────────────────────────────────

class Prescription(Base):

    __tablename__ = "prescriptions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    prescription_id = Column(String(64), unique=True, nullable=False, index=True)

    # "generated" or "uploaded"
    source    = Column(String(32),  nullable=False, default="generated")
    filename  = Column(String(256), nullable=True)
    file_path = Column(String(512), nullable=True)

    # ── Patient (matches synthetic_pa_requests.json schema exactly) ───────────
    patient_id     = Column(String(64),  nullable=True)
    patient_name   = Column(String(128), nullable=False)
    patient_age    = Column(Integer,     nullable=False)
    patient_gender = Column(String(32),  nullable=False)
    insurance_id   = Column(String(64),  nullable=True)
    insurer        = Column(String(128), nullable=False)

    # ── Procedure & diagnosis ─────────────────────────────────────────────────
    cpt_code           = Column(String(16),  nullable=False)
    procedure_name     = Column(String(256), nullable=False)
    procedure_category = Column(String(128), nullable=False)
    icd10_code         = Column(String(32),  nullable=False)
    diagnosis_desc     = Column(String(256), nullable=False)

    # ── Clinical info ─────────────────────────────────────────────────────────
    bmi                        = Column(Float,   nullable=True)
    conservative_therapy_weeks = Column(Integer, nullable=True)
    imaging_completed          = Column(Boolean, nullable=True, default=False)
    prerequisites_met          = Column(JSON,    nullable=True, default=list)

    # ── Physician ─────────────────────────────────────────────────────────────
    physician_name      = Column(String(128), nullable=True)
    physician_specialty = Column(String(128), nullable=True)
    physician_license   = Column(String(64),  nullable=True)

    # For generated records only
    approval_likelihood_label = Column(String(32), nullable=True)

    # Full structured payload (for PDF rebuild without re-querying)
    full_data = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        return {
            "id":                          self.id,
            "prescription_id":             self.prescription_id,
            "source":                      self.source,
            "filename":                    self.filename,
            "patient_id":                  self.patient_id,
            "patient_name":                self.patient_name,
            "patient_age":                 self.patient_age,
            "patient_gender":              self.patient_gender,
            "insurance_id":                self.insurance_id,
            "insurer":                     self.insurer,
            "cpt_code":                    self.cpt_code,
            "procedure_name":              self.procedure_name,
            "procedure_category":          self.procedure_category,
            "icd10_code":                  self.icd10_code,
            "diagnosis_desc":              self.diagnosis_desc,
            "bmi":                         self.bmi,
            "conservative_therapy_weeks":  self.conservative_therapy_weeks,
            "imaging_completed":           self.imaging_completed,
            "prerequisites_met":           self.prerequisites_met,
            "physician_name":              self.physician_name,
            "physician_specialty":         self.physician_specialty,
            "physician_license":           self.physician_license,
            "approval_likelihood_label":   self.approval_likelihood_label,
            "created_at":                  self.created_at.isoformat(),
        }


# ── ValidationResult ──────────────────────────────────────────────────────────

class ValidationResult(Base):
    __tablename__ = "validation_results"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64),  nullable=False, index=True)
    timestamp  = Column(String(64),  nullable=False)
    insurer    = Column(String(128), nullable=True)

    # ── Patient demographics ───────────────────────────────────────────────────
    patient_name   = Column(String(128), nullable=True)
    patient_age    = Column(Integer,     nullable=True)
    patient_gender = Column(String(32),  nullable=True)

    # ── Requested procedure ───────────────────────────────────────────────────
    cpt_code           = Column(String(16),  nullable=True)
    procedure_name     = Column(String(256), nullable=True)
    procedure_category = Column(String(128), nullable=True)

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    icd10_code     = Column(String(32),  nullable=True)
    diagnosis_desc = Column(String(256), nullable=True)

    # ── ML routing decision ───────────────────────────────────────────────────
    routing_tier        = Column(String(32),  nullable=False, index=True)
    routing_action      = Column(String(64),  nullable=False)
    routing_color       = Column(String(16),  nullable=False)
    confidence          = Column(Float,       nullable=False)
    approve_probability = Column(Float,       nullable=False)
    deny_probability    = Column(Float,       nullable=False)
    explanation         = Column(Text,        nullable=True)

    # ── Policy used for this validation ──────────────────────────────────────
    policy_id_used = Column(String(64),  nullable=True)
    policy_insurer = Column(String(128), nullable=True)

    # ── Submission metadata ───────────────────────────────────────────────────
    source             = Column(String(32),  nullable=False, default="prescription_upload")
    uploaded_filename  = Column(String(256), nullable=True)
    uploaded_file_path = Column(String(512), nullable=True)

    # Full raw data
    full_request_json = Column(JSON, nullable=True)
    full_routing_json = Column(JSON, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        return {
            "id":                  self.id,
            "request_id":          self.request_id,
            "timestamp":           self.timestamp,
            "insurer":             self.insurer,
            "patient_name":        self.patient_name,
            "patient_age":         self.patient_age,
            "patient_gender":      self.patient_gender,
            "cpt_code":            self.cpt_code,
            "procedure_name":      self.procedure_name,
            "procedure_category":  self.procedure_category,
            "icd10_code":          self.icd10_code,
            "diagnosis_desc":      self.diagnosis_desc,
            "routing_tier":        self.routing_tier,
            "routing_action":      self.routing_action,
            "routing_color":       self.routing_color,
            "confidence":          self.confidence,
            "approve_probability": self.approve_probability,
            "deny_probability":    self.deny_probability,
            "explanation":         self.explanation,
            "policy_id_used":      self.policy_id_used,
            "policy_insurer":      self.policy_insurer,
            "source":              self.source,
            "uploaded_filename":   self.uploaded_filename,
            "created_at":          self.created_at.isoformat(),
            "full_request_json":   self.full_request_json,
            "full_routing_json":   self.full_routing_json,
        }