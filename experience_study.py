"""
experience_study.py
-------------------
Produces a TPD claim experience study showing A/E ratios by cause —
before and after claim cause classification.

"Before" uses the raw_cause field as-is (messy, inconsistent).
"After"  uses the predicted_cause field from the classifier.

The comparison demonstrates the direct impact of data quality on
experience study results — the core actuarial insight of this project.

A/E ratio = Actual claims / Expected claims
  > 1.0  → worse experience than expected (more claims than assumed)
  < 1.0  → better experience than expected (fewer claims than assumed)
  = 1.0  → exactly as expected

Expected claims are derived from assumed incidence rates by cause,
loosely based on Australian TPD industry experience.

Outputs:
    experience_study.csv  — full A/E table, before and after

Usage:
    python3 experience_study.py

Requirements:
    pip install pandas numpy
"""

import pandas as pd
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

FOLDER      = Path(__file__).parent
INPUT_FILE  = FOLDER / "claims_classified.csv"
OUTPUT_FILE = FOLDER / "experience_study.csv"

# FSC categories
FSC_CATEGORIES = [
    "Cancer / Neoplasm",
    "Musculoskeletal",
    "Mental Health",
    "Cardiovascular",
    "Neurological",
    "Accident / Injury",
    "Other / Unknown",
]

# Expected incidence rate per 1,000 lives per year by cause (assumed basis)
# These represent the pricing assumption — what the actuary expected to see.
# Loosely calibrated to Australian TPD industry experience.
EXPECTED_RATE = {
    "Cancer / Neoplasm":   0.85,
    "Musculoskeletal":     0.65,
    "Mental Health":       0.55,
    "Cardiovascular":      0.35,
    "Neurological":        0.25,
    "Accident / Injury":   0.20,
    "Other / Unknown":     0.15,
}

# Assumed exposure: number of lives under observation (policy years)
# In a real study this would come from the in-force data.
# We use a fixed number here to keep the example self-contained.
EXPOSED_LIVES = 5_000

# ── Helpers ────────────────────────────────────────────────────────────────────

def normalise_raw_cause(raw: str) -> str:
    """
    Attempt a very basic normalisation of the raw cause field
    to produce the 'before' grouping — simulating what happens
    when no classifier is used and someone tries to group manually
    with inconsistent data.

    This intentionally produces a messy, incomplete grouping to
    show the 'before' state clearly.
    """
    if not isinstance(raw, str) or raw.strip() == "":
        return "Other / Unknown"

    r = raw.lower().strip()

    # Only the most obvious matches — leaves many in Other / Unknown
    if any(w in r for w in ["cancer", "ca ", " ca", "neoplasm", "tumour", "tumor", "lymphoma"]):
        return "Cancer / Neoplasm"
    if any(w in r for w in ["back", "spine", "arthritis", "knee", "hip", "shoulder", "msk"]):
        return "Musculoskeletal"
    if any(w in r for w in ["depression", "anxiety", "mental health", "psychiatric", "ptsd"]):
        return "Mental Health"
    if any(w in r for w in ["heart", "cardiac", "stroke", "coronary", "cardiovascular"]):
        return "Cardiovascular"
    if any(w in r for w in ["ms", "parkinson", "epilepsy", "dementia", "neurological"]):
        return "Neurological"
    if any(w in r for w in ["accident", "injury", "mva", "fall"]):
        return "Accident / Injury"

    return "Other / Unknown"


def build_ae_table(cause_series: pd.Series, label: str) -> pd.DataFrame:
    """
    Build an A/E table for a given cause series.
    Returns a DataFrame with one row per FSC category.
    """
    actual_counts = cause_series.value_counts()

    rows = []
    for cat in FSC_CATEGORIES:
        actual   = int(actual_counts.get(cat, 0))
        expected = round(EXPECTED_RATE[cat] * EXPOSED_LIVES / 1000, 1)
        ae_ratio = round(actual / expected, 2) if expected > 0 else None
        rows.append({
            "cause":              cat,
            f"actual_{label}":    actual,
            f"expected_{label}":  expected,
            f"ae_{label}":        ae_ratio,
        })

    df = pd.DataFrame(rows)

    # Add totals row
    total_actual   = df[f"actual_{label}"].sum()
    total_expected = df[f"expected_{label}"].sum()
    total_ae       = round(total_actual / total_expected, 2) if total_expected > 0 else None
    totals = pd.DataFrame([{
        "cause":              "TOTAL",
        f"actual_{label}":    total_actual,
        f"expected_{label}":  total_expected,
        f"ae_{label}":        total_ae,
    }])
    return pd.concat([df, totals], ignore_index=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TPD Claim Experience Study — Before vs After")
    print("=" * 60)

    # Load classified claims
    print(f"\nLoading {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  {len(df):,} claims loaded.")

    # Build before cause using naive grouping (no classifier)
    df["before_cause"] = df["raw_cause"].apply(normalise_raw_cause)

    # After cause = classifier output
    df["after_cause"] = df["predicted_cause"]

    # Build A/E tables
    before_ae = build_ae_table(df["before_cause"], "before")
    after_ae  = build_ae_table(df["after_cause"],  "after")

    # Merge into one comparison table
    study = before_ae.merge(after_ae, on="cause")

    # Add change column
    study["ae_change"] = (study["ae_after"] - study["ae_before"]).round(2)

    # Save
    study.to_csv(OUTPUT_FILE, index=False)

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n--- Before classification (raw cause, naive grouping) ---")
    print(f"{'Cause':<30} {'Actual':>8} {'Expected':>10} {'A/E':>8}")
    print("-" * 60)
    for _, row in before_ae.iterrows():
        marker = " ◄ TOTAL" if row["cause"] == "TOTAL" else ""
        ae_str = f"{row['ae_before']:>8.2f}" if row["ae_before"] is not None else f"{'N/A':>8}"
        print(
            f"{row['cause']:<30} "
            f"{row['actual_before']:>8} "
            f"{row['expected_before']:>10.1f} "
            f"{ae_str}"
            f"{marker}"
        )

    print("\n--- After classification (predicted cause) ---")
    print(f"{'Cause':<30} {'Actual':>8} {'Expected':>10} {'A/E':>8}")
    print("-" * 60)
    for _, row in after_ae.iterrows():
        marker = " ◄ TOTAL" if row["cause"] == "TOTAL" else ""
        ae_str = f"{row['ae_after']:>8.2f}" if row["ae_after"] is not None else f"{'N/A':>8}"
        print(
            f"{row['cause']:<30} "
            f"{row['actual_after']:>8} "
            f"{row['expected_after']:>10.1f} "
            f"{ae_str}"
            f"{marker}"
        )

    print("\n--- Impact of classification: A/E change by cause ---")
    print(f"{'Cause':<30} {'Before A/E':>12} {'After A/E':>12} {'Change':>10}")
    print("-" * 68)
    for _, row in study.iterrows():
        if row["cause"] == "TOTAL":
            print("-" * 68)
        direction = "▲" if row["ae_change"] > 0 else "▼" if row["ae_change"] < 0 else "─"
        print(
            f"{row['cause']:<30} "
            f"{row['ae_before']:>12.2f} "
            f"{row['ae_after']:>12.2f} "
            f"{direction} {abs(row['ae_change']):>7.2f}"
        )

    print(f"\nSaved to {OUTPUT_FILE}")

    # Key insight
    other_before = before_ae[before_ae["cause"] == "Other / Unknown"]["actual_before"].values[0]
    other_after  = after_ae[after_ae["cause"] == "Other / Unknown"]["actual_after"].values[0]
    rescued = other_before - other_after
    print(f"\n── Key insight ──────────────────────────────────────────")
    print(f"  'Other / Unknown' before: {other_before} claims")
    print(f"  'Other / Unknown' after:  {other_after} claims")
    print(f"  {rescued} claims rescued from 'Other' and correctly categorised.")
    print(f"  These claims were previously invisible to the experience study.")
    print(f"────────────────────────────────────────────────────────")
    print("\nNext step: run app.py")


if __name__ == "__main__":
    main()
