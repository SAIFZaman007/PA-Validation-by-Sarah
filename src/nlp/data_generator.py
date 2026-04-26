"""
Synthetic data generator for the PA Validation System.
Produces realistic insurance policies and PA requests for development/testing.
"""

import json
import random
from datetime import datetime
from pathlib import Path


# Medical procedure catalogue

PROCEDURES = {
    "27447": {
        "name":         "Total Knee Arthroplasty",
        "category":     "Orthopedic Surgery",
        "requires_pa":  True,
        "prerequisites":["Failed conservative therapy for 6 months", "BMI < 40"],
        "cost_range":   (15000, 25000),
    },
    "99213": {
        "name":         "Office Visit - Established Patient",
        "category":     "Primary Care",
        "requires_pa":  False,
        "prerequisites":[],
        "cost_range":   (100, 200),
    },
    "70450": {
        "name":         "CT Head without Contrast",
        "category":     "Radiology",
        "requires_pa":  True,
        "prerequisites":["Clinical indication documented"],
        "cost_range":   (500, 1500),
    },
    "29827": {
        "name":         "Arthroscopy Shoulder",
        "category":     "Orthopedic Surgery",
        "requires_pa":  True,
        "prerequisites":["Failed physical therapy for 12 weeks", "MRI documented tear"],
        "cost_range":   (8000, 15000),
    },
    "93000": {
        "name":         "Electrocardiogram (ECG)",
        "category":     "Cardiology",
        "requires_pa":  False,
        "prerequisites":[],
        "cost_range":   (50, 150),
    },
    "45378": {
        "name":         "Colonoscopy",
        "category":     "Gastroenterology",
        "requires_pa":  True,
        "prerequisites":["Age >= 45 OR family history OR symptoms"],
        "cost_range":   (2000, 4000),
    },
    "88305": {
        "name":         "Tissue Examination",
        "category":     "Pathology",
        "requires_pa":  False,
        "prerequisites":[],
        "cost_range":   (200, 500),
    },
    "73721": {
        "name":         "MRI Lower Extremity",
        "category":     "Radiology",
        "requires_pa":  True,
        "prerequisites":["Conservative therapy failed", "X-ray completed"],
        "cost_range":   (1500, 3000),
    },
}

# ICD-10 diagnosis codes
DIAGNOSES = {
    "M17.11":   "Unilateral primary osteoarthritis, right knee",
    "M25.561":  "Pain in right knee",
    "M75.100":  "Unspecified rotator cuff tear, unspecified shoulder",
    "I10":      "Essential hypertension",
    "E11.9":    "Type 2 diabetes mellitus without complications",
    "M54.5":    "Low back pain",
    "R07.9":    "Chest pain, unspecified",
    "K92.1":    "Gastrointestinal hemorrhage",
    "S83.511A": "Sprain of anterior cruciate ligament of right knee",
}

INSURERS    = ["Bupa Arabia", "Tawuniya", "Al-Rajhi Takaful", "Medgulf", "Solidarity Saudi Takaful"]
FIRST_NAMES = ["Ahmed", "Mohammed", "Fatima", "Sara", "Abdullah", "Noura", "Khalid", "Huda"]
LAST_NAMES  = ["Al-Otaibi", "Al-Ghamdi", "Al-Zahrani", "Al-Harbi", "Al-Malki", "Al-Qahtani"]

# Probability of meeting each prerequisite per approval likelihood
_PREREQ_PROB = {"high": 0.95, "medium": 0.60, "low": 0.20}


# Internal helpers

def _coverage_criteria(proc: dict) -> str:
    """Build a human-readable coverage criteria string from procedure metadata."""
    lines = []
    if proc["requires_pa"]:
        lines.append("Prior authorization is mandatory before procedure.")
    else:
        lines.append("No prior authorization required.")

    if proc["prerequisites"]:
        lines.append("Patient must meet the following conditions:")
        for p in proc["prerequisites"]:
            lines.append(f"  - {p}")

    if proc["category"] == "Orthopedic Surgery":
        lines.append("Documented physical examination required.")
        lines.append("Imaging studies must be submitted with request.")

    return " ".join(lines)


def _prereq_compliance(prerequisites: list, likelihood: str) -> list:
    """
    Return the subset of prerequisites that are 'met' for a given likelihood.
    
    Ensures a realistic distribution:
    - High approval: 90-100% of prerequisites met
    - Medium approval: 50-70% of prerequisites met
    - Low approval: 10-30% of prerequisites met
    """
    if not prerequisites:
        return []
    
    # Defining how many prerequisites should be met based on likelihood
    num_prereqs = len(prerequisites)
    
    if likelihood == "high":
        # For high approval: meet 80-100% of prerequisites
        min_met = max(1, int(num_prereqs * 0.80))
        max_met = num_prereqs
        num_to_meet = random.randint(min_met, max_met)
    elif likelihood == "medium":
        # For medium approval: meet 40-70% of prerequisites
        min_met = max(1, int(num_prereqs * 0.40))
        max_met = max(1, int(num_prereqs * 0.70))
        num_to_meet = random.randint(min_met, max_met)
    else:  # low
        # For low approval: meet 0-30% of prerequisites
        max_met = max(0, int(num_prereqs * 0.30))
        num_to_meet = random.randint(0, max_met)
    
    # Randomly select which prerequisites are met
    if num_to_meet > 0:
        met_prereqs = random.sample(prerequisites, min(num_to_meet, num_prereqs))
    else:
        met_prereqs = []
    
    return met_prereqs


# Public API

def generate_policy(insurer: str = None, num_procedures: int = 5) -> dict:
    """
    Generate one synthetic insurance policy document.

    Args:
        insurer:        Insurance company name. Random if None.
        num_procedures: How many procedures to include.

    Returns:
        Policy dict with coverage_rules list.
    """
    if insurer is None:
        insurer = random.choice(INSURERS)

    cpt_codes = random.sample(list(PROCEDURES.keys()), min(num_procedures, len(PROCEDURES)))

    rules = []
    for cpt in cpt_codes:
        proc = PROCEDURES[cpt]
        rules.append({
            "cpt_code":          cpt,
            "procedure_name":    proc["name"],
            "category":          proc["category"],
            "requires_prior_auth": proc["requires_pa"],
            "prerequisites":     proc["prerequisites"],
            "coverage_criteria": _coverage_criteria(proc),
            "estimated_cost":    f"SAR {proc['cost_range'][0]:,} - {proc['cost_range'][1]:,}",
        })

    return {
        "policy_id":      f"POL-{random.randint(10000, 99999)}",
        "insurer":        insurer,
        "effective_date": "2026-01-01",
        "version":        "2026.1",
        "coverage_rules": rules,
    }


def generate_pa_request(policy: dict = None, approval_likelihood: str = "random") -> dict:
    """
    Generate one synthetic PA request.

    Args:
        policy:             Policy to reference. Generated randomly if None.
        approval_likelihood: 'high', 'medium', 'low', or 'random'.

    Returns:
        PA request dict.
    """
    if policy is None:
        policy = generate_policy(num_procedures=3)

    if approval_likelihood == "random":
        approval_likelihood = random.choice(["high", "medium", "low"])

    # Prefer procedures that actually require PA
    pa_rules = [r for r in policy["coverage_rules"] if r["requires_prior_auth"]]
    rule = random.choice(pa_rules) if pa_rules else random.choice(policy["coverage_rules"])

    patient = {
        "patient_id":   f"PAT-{random.randint(100000, 999999)}",
        "name":         f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "age":          random.randint(25, 75),
        "gender":       random.choice(["Male", "Female"]),
        "insurance_id": f"INS-{random.randint(1000000, 9999999)}",
    }

    diag_code = random.choice(list(DIAGNOSES.keys()))
    prereqs_met = _prereq_compliance(rule["prerequisites"], approval_likelihood)

    # Generate clinical info based on approval likelihood
    # High approval: better compliance with clinical parameters
    # Medium approval: moderate compliance
    # Low approval: poor compliance
    if approval_likelihood == "high":
        therapy_weeks = random.randint(8, 24)
        imaging_completed = random.choice([True, True, False])  # 67% True
        bmi = round(random.uniform(20, 35), 1)  # Better BMI
    elif approval_likelihood == "medium":
        therapy_weeks = random.randint(4, 16)
        imaging_completed = random.choice([True, False])  # 50% True
        bmi = round(random.uniform(25, 38), 1)  # Medium BMI
    else:  # low
        therapy_weeks = random.randint(0, 6)
        imaging_completed = random.choice([False, False, True])  # 33% True
        bmi = round(random.uniform(30, 45), 1)  # Higher BMI

    return {
        "request_id":   f"PA-{random.randint(100000, 999999)}",
        "timestamp":    datetime.now().isoformat(),
        "patient":      patient,
        "insurer":      policy["insurer"],
        "requested_procedure": {
            "cpt_code": rule["cpt_code"],
            "name":     rule["procedure_name"],
            "category": rule["category"],
        },
        "diagnosis": {
            "icd10_code":  diag_code,
            "description": DIAGNOSES[diag_code],
        },
        "clinical_info": {
            "prerequisites_met": prereqs_met,
            "conservative_therapy_duration_weeks": therapy_weeks,
            "imaging_completed": imaging_completed,
            "bmi": bmi,
        },
        "requesting_physician": {
            "name":           f"Dr. {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "specialty":      rule["category"],
            "license_number": f"MD-{random.randint(10000, 99999)}",
        },
        "approval_likelihood_label": approval_likelihood,
    }


def generate_dataset(num_policies: int = 5, requests_per_policy: int = 20):
    """
    Generate a full dataset of policies and PA requests.

    Returns:
        (policies list, requests list)
    """
    policies = [generate_policy() for _ in range(num_policies)]
    requests = [
        generate_pa_request(policy=p)
        for p in policies
        for _ in range(requests_per_policy)
    ]
    return policies, requests


def save_dataset(policies: list, requests: list, output_dir: Path = None):
    """Save dataset to JSON files in output_dir."""
    if output_dir is None:
        from core.config import BACKEND_DIR
        output_dir = BACKEND_DIR / "data" / "synthetic"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pol_file = output_dir / "synthetic_policies.json"
    req_file = output_dir / "synthetic_pa_requests.json"

    with open(pol_file, "w", encoding="utf-8") as f:
        json.dump(policies, f, indent=2, ensure_ascii=False)

    with open(req_file, "w", encoding="utf-8") as f:
        json.dump(requests, f, indent=2, ensure_ascii=False)

    return pol_file, req_file


# Backwards-compatible class wrapper

class SyntheticDataGenerator:
    """
    Thin class wrapper kept for compatibility with app.py imports.
    All logic lives in the module-level functions above.
    """

    def generate_policy_document(self, insurer_name: str = None, num_procedures: int = 5) -> dict:
        return generate_policy(insurer=insurer_name, num_procedures=num_procedures)

    def generate_pa_request(self, policy: dict = None, approval_likelihood: str = "random") -> dict:
        return generate_pa_request(policy=policy, approval_likelihood=approval_likelihood)

    def generate_dataset(self, num_policies: int = 5, num_requests_per_policy: int = 20):
        return generate_dataset(num_policies=num_policies, requests_per_policy=num_requests_per_policy)

    def save_dataset(self, policies: list, requests: list, output_dir=None):
        return save_dataset(policies, requests, output_dir)


# CLI entry point

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("Generating dataset …")
    pols, reqs = generate_dataset(num_policies=5, requests_per_policy=20)
    pol_f, req_f = save_dataset(pols, reqs)
    print(f"Saved {len(pols)} policies → {pol_f}")
    print(f"Saved {len(reqs)} requests → {req_f}")