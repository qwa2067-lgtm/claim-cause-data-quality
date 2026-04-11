"""
generate_claims.py
------------------
Generates 1,000 synthetic TPD claims with realistic but messy claim cause fields.
Simulates the real-world problem of inconsistent manual data entry in insurance
administration systems.

Outputs:
    claims_data.csv — the raw messy dataset (what the tool receives as input)

The 'true_cause' column represents the correct FSC category (used later to
measure classifier accuracy and produce the "after" experience study).
In a real system this ground truth would not exist — it has to be derived
by the classifier.

Usage:
    python3 generate_claims.py

Requirements:
    pip install pandas numpy faker
"""

import random
import numpy as np
import pandas as pd
from faker import Faker
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

RANDOM_SEED   = 42
N_CLAIMS      = 1000
OUTPUT_FILE   = Path(__file__).parent / "claims_data.csv"

# FSC cause categories and their true share of TPD claims
# Loosely based on Australian industry experience
FSC_CATEGORIES = {
    "Cancer / Neoplasm":        0.28,
    "Musculoskeletal":          0.22,
    "Mental Health":            0.18,
    "Cardiovascular":           0.12,
    "Neurological":             0.08,
    "Accident / Injury":        0.07,
    "Other / Unknown":          0.05,
}

# ── Messy cause entries per category ──────────────────────────────────────────
# Each list simulates how different staff members might enter the same condition.
# Includes: abbreviations, misspellings, free text, mixed formats, ambiguous terms.

MESSY_CAUSES = {
    "Cancer / Neoplasm": [
        "Cancer", "cancer", "CANCER", "Ca", "CA", "ca.",
        "Breast CA", "breast cancer", "Lung CA", "lung ca",
        "Caner",                        # common misspelling
        "Canncer",
        "Neoplasm", "neoplasm", "NEOPL",
        "Malignancy", "malignancy", "malig",
        "Carcinoma", "carcinoma", "carc",
        "Tumour", "tumor", "TUMOUR",
        "Lymphoma", "lymphoma",
        "Leukaemia", "leukemia",
        "Melanoma", "melanoma",
        "Metastatic disease",
        "Ca - lung", "Ca - breast", "Ca - bowel",
        "Oncology", "oncology dx",
        "Terminal illness - cancer",
        "Stage 4 cancer",
    ],
    "Musculoskeletal": [
        "Musculoskeletal", "MSK", "msk",
        "Back injury", "back pain", "Back Pain", "BACK",
        "Spine", "spinal", "SPINAL", "spinal injury",
        "Disc", "disc injury", "disc prolapse", "herniated disc",
        "Arthritis", "arthritis", "OA", "RA", "rheumatoid",
        "Osteoarthritis",
        "Knee", "knee injury", "knee replacement",
        "Hip", "hip replacement", "hip injury",
        "Shoulder", "shoulder injury", "rotator cuff",
        "Chronic pain", "chronic pain syndrome",
        "Degenerative disc disease", "DDD",
        "Scoliosis",
        "Musculo-skeletal",
        "Muscle disorder",
        "Joint disease",
        "Orthopaedic",
    ],
    "Mental Health": [
        "Mental health", "Mental Health", "MENTAL HEALTH", "MH",
        "Depression", "depression", "DEPRESSION", "depresion",
        "Anxiety", "anxiety", "ANXIETY",
        "Psychiatric", "psychiatric", "psych",
        "PTSD", "ptsd", "Post traumatic stress",
        "Bipolar", "bipolar disorder",
        "Schizophrenia", "schizophrenia",
        "Psychological", "psychological condition",
        "Mental illness", "mental illness",
        "Stress", "work-related stress",
        "Burnout", "burn out",
        "Schizoaffective",
        "OCD",
        "Personality disorder",
        "Mood disorder",
        "Nervous breakdown",
        "Psych condition - not specified",
    ],
    "Cardiovascular": [
        "Cardiovascular", "CVD", "cvd",
        "Heart attack", "heart attack", "HEART ATTACK",
        "MI", "AMI", "myocardial infarction",
        "Heart failure", "cardiac failure", "CHF",
        "Stroke", "stroke", "STROKE", "CVA", "cva",
        "Coronary artery disease", "CAD", "cad",
        "Angina", "angina",
        "Atrial fibrillation", "AF", "A-fib",
        "Cardiac", "cardiac event",
        "Heart disease",
        "Bypass surgery",
        "Valve disease",
        "Hypertensive heart disease",
        "Aortic aneurysm",
    ],
    "Neurological": [
        "Neurological", "neuro", "NEURO",
        "Multiple sclerosis", "MS", "m.s.",
        "Parkinson", "Parkinsons", "Parkinson's disease", "PD",
        "Epilepsy", "epilepsy", "seizures",
        "Motor neurone disease", "MND", "mnd", "ALS",
        "Dementia", "dementia", "early onset dementia",
        "Brain injury", "TBI", "acquired brain injury", "ABI",
        "Neuropathy", "peripheral neuropathy",
        "Muscular dystrophy", "MD",
        "Cerebral palsy",
        "Nerve disorder",
        "Spinal cord injury",                 # ambiguous — could be MSK or Neuro
        "Chronic neurological condition",
    ],
    "Accident / Injury": [
        "Accident", "accident", "ACCIDENT",
        "Injury", "injury", "INJURY",
        "MVA", "mva", "motor vehicle accident",
        "Car accident", "car crash",
        "Workplace injury", "work injury", "WPI",
        "Fall", "fall injury", "fall from height",
        "Trauma", "trauma",                   # ambiguous — could be mental health
        "Burns", "severe burns",
        "Amputation",
        "Fracture", "fractures",              # ambiguous — could be MSK
        "Head injury",                        # ambiguous — could be Neuro
        "Spinal cord - accident",
        "Accidental",
        "Crush injury",
    ],
    "Other / Unknown": [
        "Other", "other", "OTHER",
        "Unknown", "unknown", "UNKNOWN", "unk",
        "N/A", "n/a", "NA", "nil",
        "Not stated", "not specified", "NS",
        "See file", "refer to file", "refer medical",
        "Various", "multiple conditions",
        "Chronic illness",                    # genuinely ambiguous
        "Systemic disease",
        "Autoimmune",                         # could map to several categories
        "Rare disease",
        "Congenital",
        "Misc",
        "TBC",
        "Pending",
        "",                                   # blank entries
    ],
}

# ── Claim generation ───────────────────────────────────────────────────────────

def generate_claims(n: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    fake = Faker("en_AU")
    Faker.seed(seed)

    categories = list(FSC_CATEGORIES.keys())
    weights    = list(FSC_CATEGORIES.values())

    records = []
    for i in range(n):
        # Demographics
        gender     = random.choice(["M", "F"])
        age        = int(np.random.normal(loc=48, scale=10))
        age        = max(20, min(65, age))
        entry_age  = max(18, age - random.randint(1, 15))
        policy_dur = age - entry_age

        # Sum insured — broadly realistic for Australian TPD
        sum_insured = random.choice([250_000, 300_000, 500_000, 750_000, 1_000_000])

        # True FSC cause — assigned by design (ground truth)
        true_cause = random.choices(categories, weights=weights)[0]

        # Messy cause — what was actually entered in the system
        messy_pool = MESSY_CAUSES[true_cause]

        # Occasionally inject cross-category confusion (realistic ~8% of claims)
        if random.random() < 0.08:
            wrong_category = random.choice([c for c in categories if c != true_cause])
            messy_pool = MESSY_CAUSES[wrong_category]

        raw_cause = random.choice(messy_pool)

        records.append({
            "claim_id":       f"CLM{i+1:04d}",
            "gender":         gender,
            "age_at_claim":   age,
            "entry_age":      entry_age,
            "policy_duration": policy_dur,
            "sum_insured":    sum_insured,
            "raw_cause":      raw_cause,       # messy — what the tool receives
            "true_cause":     true_cause,      # ground truth — used for validation
        })

    return pd.DataFrame(records)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Generating synthetic TPD claims...")
    df = generate_claims(N_CLAIMS, RANDOM_SEED)

    # Summary
    print(f"\nGenerated {len(df):,} claims.")
    print("\nTrue cause distribution:")
    dist = df["true_cause"].value_counts()
    for cause, count in dist.items():
        print(f"  {cause:<30} {count:>4}  ({count/len(df)*100:.1f}%)")

    print(f"\nSample raw cause entries (first 10):")
    for _, row in df.head(10).iterrows():
        print(f"  [{row['true_cause']:<30}]  raw: '{row['raw_cause']}'")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
    print("Next step: run classifier.py")


if __name__ == "__main__":
    main()
