# Claim Cause Data Quality Tool

A Streamlit dashboard demonstrating the impact of inconsistent claim cause coding on TPD actuarial experience studies — and how a three-layer classification pipeline restores data integrity.

---

## The Problem

In many life insurance administration systems, claim cause is entered manually by claims staff with no controlled vocabulary or validation. The same condition gets recorded dozens of different ways:

> `Cancer` · `ca` · `Breast CA` · `Caner` · `NEOPL` · `malignancy` · `Stage 4 cancer`

This corrupts the cause split in experience studies without affecting the total claim count — making the error **silent and easy to miss for years**.

The downstream impact touches every actuarial and operational function: pricing, reserving, reinsurance reporting, product design, capital modelling, industry submissions, and claims operations.

---

## The Approach

A three-layer classifier maps messy cause entries to FSC standard categories:

| Layer | Method | Purpose |
|-------|--------|---------|
| 1 | Rule-based + regex | Fast, transparent, auditable — handles clear cases |
| 2 | Fuzzy string matching | Catches misspellings and abbreviations |
| 3 | ML classifier (TF-IDF + Logistic Regression) | Handles ambiguous cases, returns confidence score |

Low-confidence classifications are flagged for actuary review. Corrections feed back as new training examples — the model gets smarter over time (**active learning**).

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| The Problem | What inconsistent cause coding looks like, and the full downstream impact |
| The Classifier | How the 3-layer pipeline works, accuracy metrics, feedback loop |
| Experience Study | Before vs after A/E ratios by cause — the actuarial output |
| Claims Explorer | Browse individual claims, filter by method / cause / correctness |

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1 — generate synthetic claims data
python3 generate_claims.py

# Step 2 — run the classifier
python3 classifier.py

# Step 3 — run the experience study
python3 experience_study.py

# Step 4 — launch the dashboard
streamlit run app.py
```

---

## Project Structure

```
ClaimCauseTool/
├── app.py                  # Streamlit dashboard
├── generate_claims.py      # Generates 1,000 synthetic TPD claims
├── classifier.py           # 3-layer claim cause classifier
├── experience_study.py     # Before/after A/E experience study
├── requirements.txt
└── .gitignore
```

Generated files (`claims_data.csv`, `claims_classified.csv`, `experience_study.csv`, `model/`) are excluded from the repository. Run the scripts in order to recreate them.

---

## Disclaimer

This tool uses **synthetic data** generated for illustration purposes only. It does not contain any real policyholder information. The FSC cause categories and expected incidence rates used are loosely based on published Australian industry experience and are not intended to represent any specific insurer's actual experience.

---


*Built by Amy Wang, FIAA.*

