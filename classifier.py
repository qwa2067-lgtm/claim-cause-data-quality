"""
classifier.py
-------------
3-layer claim cause classifier:

  Layer 1 — Rule-based matching
            Fast, transparent, auditable. Catches obvious cases using
            keyword rules defined by the actuary / domain expert.

  Layer 2 — Fuzzy string matching
            Catches misspellings, abbreviations, and near-matches that
            rules miss. Uses rapidfuzz for efficient similarity scoring.

  Layer 3 — ML text classifier (TF-IDF + Logistic Regression)
            Handles ambiguous cases that layers 1 and 2 couldn't resolve
            with confidence. Trained on labelled examples.

  Uncertain flag
            Claims where the ML model confidence is below threshold are
            flagged for human review. These corrections feed back into
            the training data (active learning loop).

Outputs:
    claims_classified.csv — original claims with predicted cause,
                            method used, confidence, and review flag

Usage:
    python3 classifier.py

Requirements:
    pip install pandas numpy scikit-learn rapidfuzz
"""

import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# ── Config ─────────────────────────────────────────────────────────────────────

FOLDER         = Path(__file__).parent
INPUT_FILE     = FOLDER / "claims_data.csv"
OUTPUT_FILE    = FOLDER / "claims_classified.csv"
MODEL_DIR      = FOLDER / "model"
MODEL_FILE     = MODEL_DIR / "classifier_pipeline.pkl"
FUZZY_THRESHOLD   = 80    # minimum similarity score (0-100) for fuzzy match
CONFIDENCE_THRESHOLD = 0.65  # ML confidence below this → flag for human review

# ── FSC categories ─────────────────────────────────────────────────────────────

FSC_CATEGORIES = [
    "Cancer / Neoplasm",
    "Musculoskeletal",
    "Mental Health",
    "Cardiovascular",
    "Neurological",
    "Accident / Injury",
    "Other / Unknown",
]

# ── Layer 1: Rule-based keyword matching ──────────────────────────────────────
# Each rule: list of keywords/phrases → FSC category
# Rules are checked in order — first match wins.
# Defined by domain expert (actuary). Transparent and auditable.

RULES = {
    "Cancer / Neoplasm": [
        "cancer", "carcinoma", "neoplasm", "malignancy", "malignant",
        "tumour", "tumor", "lymphoma", "leukaemia", "leukemia",
        "melanoma", "oncology", "metastatic", "sarcoma", "myeloma",
        r"\bca\b", r"\bcancer\b",
    ],
    "Musculoskeletal": [
        "musculoskeletal", "msk", "back pain", "back injury", "spine",
        "spinal", "disc", "arthritis", "osteoarthritis", "rheumatoid",
        "knee", "hip", "shoulder", "rotator", "scoliosis", "orthopaedic",
        "orthopedic", "chronic pain", "degenerative", "joint", "muscle disorder",
        r"\boa\b", r"\bra\b", r"\bddd\b",
    ],
    "Mental Health": [
        "mental health", "depression", "anxiety", "psychiatric", "ptsd",
        "post traumatic", "bipolar", "schizophrenia", "psychological",
        "mental illness", "burnout", "burn out", "nervous breakdown",
        "schizoaffective", r"\bocd\b", "personality disorder", "mood disorder",
        r"\bmh\b",
    ],
    "Cardiovascular": [
        "cardiovascular", "heart attack", "myocardial", r"\bami\b", r"\bmi\b",
        "heart failure", "cardiac failure", "stroke", r"\bcva\b",
        "coronary", r"\bcad\b", "angina", "atrial fibrillation",
        r"\ba-fib\b", r"\baf\b", "bypass", "valve", "aortic", "hypertensive heart",
        r"\bcvd\b",
    ],
    "Neurological": [
        "neurological", "multiple sclerosis", r"\bms\b", "parkinson",
        "epilepsy", "seizure", "motor neurone", r"\bmnd\b", r"\bals\b",
        "dementia", "acquired brain", r"\babi\b", r"\btbi\b", "neuropathy",
        "muscular dystrophy", "cerebral palsy", "nerve disorder",
        r"\bpd\b",
    ],
    "Accident / Injury": [
        "accident", r"\bmva\b", "motor vehicle", "workplace injury",
        "work injury", r"\bwpi\b", "fall injury", "fall from",
        "severe burns", "amputation", "crush injury", "accidental",
    ],
}


def rule_match(text: str) -> str | None:
    """
    Apply keyword rules to cleaned text.
    Returns FSC category string if matched, None otherwise.
    """
    text = text.lower().strip()
    for category, patterns in RULES.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return category
    return None


# ── Layer 2: Fuzzy string matching ────────────────────────────────────────────
# Build a flat reference list of known terms mapped to FSC categories.
# Use rapidfuzz to find closest match above threshold.

from generate_claims import MESSY_CAUSES  # reuse the same vocabulary

FUZZY_REFERENCE = []   # list of (term, fsc_category)
for category, terms in MESSY_CAUSES.items():
    for term in terms:
        if term.strip():   # skip blank entries
            FUZZY_REFERENCE.append((term.lower().strip(), category))

FUZZY_TERMS = [t for t, _ in FUZZY_REFERENCE]
FUZZY_CATS  = [c for _, c in FUZZY_REFERENCE]


def fuzzy_match(text: str) -> tuple[str | None, float]:
    """
    Find the closest known term using fuzzy string similarity.
    Returns (category, score) if above threshold, (None, score) otherwise.
    """
    text = text.lower().strip()
    if not text:
        return None, 0.0

    result = process.extractOne(text, FUZZY_TERMS, scorer=fuzz.WRatio)
    if result is None:
        return None, 0.0

    matched_term, score, idx = result
    if score >= FUZZY_THRESHOLD:
        return FUZZY_CATS[idx], score
    return None, score


# ── Layer 3: ML classifier (TF-IDF + Logistic Regression) ────────────────────

def build_training_data(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Build training data from claims where we have ground truth.
    In production, this would come from actuary-reviewed/corrected claims.
    Here we use the synthetic true_cause column.
    """
    X = df["raw_cause"].fillna("unknown").astype(str).tolist()
    y = df["true_cause"].tolist()
    return X, y


def train_model(X: list[str], y: list[str]) -> Pipeline:
    """
    Train a TF-IDF + Logistic Regression pipeline.
    TF-IDF converts text to numeric features.
    Logistic Regression classifies into FSC categories with probability scores.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",    # character n-grams — robust to misspellings
            ngram_range=(2, 4),    # bigrams to 4-grams
            min_df=1,
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
        )),
    ])

    # Cross-validation to report accuracy before saving
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"  ML model 5-fold CV accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    pipeline.fit(X, y)
    return pipeline


def ml_predict(pipeline: Pipeline, texts: list[str]) -> tuple[list[str], list[float]]:
    """
    Predict FSC category and return confidence (max class probability).
    """
    probs      = pipeline.predict_proba(texts)
    categories = pipeline.classes_
    predictions   = [categories[np.argmax(p)] for p in probs]
    confidences   = [float(np.max(p)) for p in probs]
    return predictions, confidences


# ── Main classification pipeline ──────────────────────────────────────────────

def classify_claims(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """
    Run all three layers on each claim.
    Records which layer made the final classification and confidence score.
    """
    results = []

    all_texts = df["raw_cause"].fillna("").astype(str).tolist()
    ml_preds, ml_confs = ml_predict(pipeline, all_texts)

    for pos, (i, row) in enumerate(df.iterrows()):
        raw = str(row["raw_cause"]).strip()

        # Layer 1 — rules
        pred = rule_match(raw)
        if pred:
            results.append({
                "predicted_cause": pred,
                "method":          "Rule-based",
                "confidence":      1.0,
                "review_flag":     False,
            })
            continue

        # Layer 2 — fuzzy
        pred, score = fuzzy_match(raw)
        if pred:
            results.append({
                "predicted_cause": pred,
                "method":          "Fuzzy match",
                "confidence":      round(score / 100, 2),
                "review_flag":     False,
            })
            continue

        # Layer 3 — ML
        pred       = ml_preds[pos]
        confidence = ml_confs[pos]
        review     = confidence < CONFIDENCE_THRESHOLD
        results.append({
            "predicted_cause": pred,
            "method":          "ML model",
            "confidence":      round(confidence, 2),
            "review_flag":     review,
        })

    result_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, result_df], axis=1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Claim Cause Classifier")
    print("=" * 60)

    # Load data
    print(f"\nLoading {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  {len(df):,} claims loaded.")

    # Train ML model
    print("\nTraining ML classifier...")
    X, y = build_training_data(df)
    pipeline = train_model(X, y)

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"  Model saved to {MODEL_FILE}")

    # Classify all claims
    print("\nClassifying claims...")
    classified = classify_claims(df, pipeline)

    # ── Results summary ────────────────────────────────────────────────────────
    print("\n--- Classification summary ---")

    method_counts = classified["method"].value_counts()
    for method, count in method_counts.items():
        print(f"  {method:<20} {count:>4} claims  ({count/len(df)*100:.1f}%)")

    review_count = classified["review_flag"].sum()
    print(f"\n  Flagged for human review: {review_count} claims ({review_count/len(df)*100:.1f}%)")

    # Accuracy vs ground truth
    correct = (classified["predicted_cause"] == classified["true_cause"]).sum()
    accuracy = correct / len(classified)
    print(f"\n--- Accuracy vs ground truth ---")
    print(f"  Overall accuracy: {accuracy:.1%}  ({correct}/{len(classified)} correct)")

    print(f"\n  Accuracy by FSC category:")
    for cat in sorted(classified["true_cause"].unique()):
        subset = classified[classified["true_cause"] == cat]
        cat_acc = (subset["predicted_cause"] == subset["true_cause"]).mean()
        print(f"    {cat:<30} {cat_acc:.1%}  (n={len(subset)})")

    # Save output
    classified.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
    print("Next step: run experience_study.py")


if __name__ == "__main__":
    main()
