"""add prescriptions table and policy link to validation_results

Revision ID: 0002_prescriptions
Revises: 9792ad489538
Create Date: 2026-04-25

Uses raw SQL with IF NOT EXISTS / DO NOTHING guards so this migration is
fully idempotent — safe whether or not create_tables() already ran, and
safe to run multiple times.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text, inspect

revision      = "0002_prescriptions"
down_revision = "9792ad489538"
branch_labels = None
depends_on    = None


def _table_exists(conn, table_name: str) -> bool:
    return inspect(conn).has_table(table_name)


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    cols = [c["name"] for c in inspect(conn).get_columns(table_name)]
    return column_name in cols


def upgrade():
    bind = op.get_bind()

    # ── prescriptions table ───────────────────────────────────────────────────
    if not _table_exists(bind, "prescriptions"):
        op.create_table(
            "prescriptions",
            sa.Column("id",              sa.Integer(),    primary_key=True, autoincrement=True),
            sa.Column("prescription_id", sa.String(64),   unique=True, nullable=False),
            sa.Column("source",          sa.String(32),   nullable=False, server_default="generated"),
            sa.Column("filename",        sa.String(256),  nullable=True),
            sa.Column("file_path",       sa.String(512),  nullable=True),
            sa.Column("patient_id",      sa.String(64),   nullable=True),
            sa.Column("patient_name",    sa.String(128),  nullable=False, server_default="Unknown"),
            sa.Column("patient_age",     sa.Integer(),    nullable=False, server_default="0"),
            sa.Column("patient_gender",  sa.String(32),   nullable=False, server_default="Unknown"),
            sa.Column("insurance_id",    sa.String(64),   nullable=True),
            sa.Column("insurer",         sa.String(128),  nullable=False, server_default="Unknown"),
            sa.Column("cpt_code",        sa.String(16),   nullable=False, server_default="00000"),
            sa.Column("procedure_name",  sa.String(256),  nullable=False, server_default="Unknown"),
            sa.Column("procedure_category", sa.String(128), nullable=False, server_default="General"),
            sa.Column("icd10_code",      sa.String(32),   nullable=False, server_default="Z00.00"),
            sa.Column("diagnosis_desc",  sa.String(256),  nullable=False, server_default="Unknown"),
            sa.Column("bmi",             sa.Float(),      nullable=True),
            sa.Column("conservative_therapy_weeks", sa.Integer(), nullable=True),
            sa.Column("imaging_completed", sa.Boolean(), nullable=True),
            sa.Column("prerequisites_met", sa.JSON(),    nullable=True),
            sa.Column("physician_name",  sa.String(128), nullable=True),
            sa.Column("physician_specialty", sa.String(128), nullable=True),
            sa.Column("physician_license",   sa.String(64),  nullable=True),
            sa.Column("approval_likelihood_label", sa.String(32), nullable=True),
            sa.Column("full_data",       sa.JSON(),      nullable=True),
            sa.Column("created_at",      sa.DateTime(timezone=True), nullable=False,
                      server_default=sa.func.now()),
        )
        # Index on prescription_id
        op.create_index("ix_prescriptions_prescription_id", "prescriptions", ["prescription_id"], unique=True)
    else:
        # Table already exists — add any missing columns individually
        missing_cols = {
            "patient_id":                 sa.Column("patient_id",                 sa.String(64),   nullable=True),
            "insurance_id":               sa.Column("insurance_id",               sa.String(64),   nullable=True),
            "bmi":                        sa.Column("bmi",                        sa.Float(),      nullable=True),
            "conservative_therapy_weeks": sa.Column("conservative_therapy_weeks", sa.Integer(),    nullable=True),
            "imaging_completed":          sa.Column("imaging_completed",          sa.Boolean(),    nullable=True),
            "prerequisites_met":          sa.Column("prerequisites_met",          sa.JSON(),       nullable=True),
            "physician_name":             sa.Column("physician_name",             sa.String(128),  nullable=True),
            "physician_specialty":        sa.Column("physician_specialty",        sa.String(128),  nullable=True),
            "physician_license":          sa.Column("physician_license",          sa.String(64),   nullable=True),
            "approval_likelihood_label":  sa.Column("approval_likelihood_label",  sa.String(32),   nullable=True),
            "full_data":                  sa.Column("full_data",                  sa.JSON(),       nullable=True),
        }
        for col_name, col_def in missing_cols.items():
            if not _column_exists(bind, "prescriptions", col_name):
                op.add_column("prescriptions", col_def)

    # ── validation_results: add new columns if they don't exist ───────────────
    if _table_exists(bind, "validation_results"):
        new_vr_cols = {
            "policy_id_used": sa.Column("policy_id_used", sa.String(64),  nullable=True),
            "policy_insurer": sa.Column("policy_insurer", sa.String(128), nullable=True),
        }
        for col_name, col_def in new_vr_cols.items():
            if not _column_exists(bind, "validation_results", col_name):
                op.add_column("validation_results", col_def)


def downgrade():
    bind = op.get_bind()

    if _table_exists(bind, "validation_results"):
        if _column_exists(bind, "validation_results", "policy_insurer"):
            op.drop_column("validation_results", "policy_insurer")
        if _column_exists(bind, "validation_results", "policy_id_used"):
            op.drop_column("validation_results", "policy_id_used")

    if _table_exists(bind, "prescriptions"):
        op.drop_table("prescriptions")