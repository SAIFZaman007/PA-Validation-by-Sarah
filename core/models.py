"""
SQLAlchemy ORM models for the PA Validation System.

Two tables:

  Policy
    Every insurance policy — auto-generated for testing or uploaded as a
    real PDF.  The coverage_rules JSON column stores the structured rules
    extracted by PolicyExtractor.  These rules are also cached in memory
    inside the ML pipeline service so the database is not hit on every
    single PA submission.

  ValidationResult
    A permanent audit record of every PA request and its ML routing
    decision.  This is what powers the submissions history on the dashboard.

Both models are database-agnostic.  SQLite works locally with zero setup.
Switching to PostgreSQL only requires changing DATABASE_URL in .env.
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
    source   = Column(String(32), nullable=False, default="generated")

    # Populated only when source == "uploaded"
    filename  = Column(String(256), nullable=True)
    file_path = Column(String(512), nullable=True)

    # Structured coverage rules as a JSON array
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
    # routing_tier is indexed — the dashboard filters by tier frequently.
    routing_tier        = Column(String(32),  nullable=False, index=True)
    routing_action      = Column(String(64),  nullable=False)
    routing_color       = Column(String(16),  nullable=False)
    confidence          = Column(Float,       nullable=False)
    approve_probability = Column(Float,       nullable=False)
    deny_probability    = Column(Float,       nullable=False)
    explanation         = Column(Text,        nullable=True)

    # ── Submission metadata ───────────────────────────────────────────────────
    source             = Column(String(32),  nullable=False, default="auto_generated")
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
            "source":              self.source,
            "uploaded_filename":   self.uploaded_filename,
            "created_at":          self.created_at.isoformat(),
            "full_request_json":   self.full_request_json,
            "full_routing_json":   self.full_routing_json,
        }