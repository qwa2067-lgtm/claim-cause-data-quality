"""
app.py
------
Streamlit dashboard: Claim Cause Data Quality Tool
Demonstrates the impact of inconsistent claim cause coding on TPD
experience studies, and how a 3-layer classifier restores data integrity.

Usage:
    streamlit run app.py

Requirements:
    pip install streamlit pandas numpy scikit-learn rapidfuzz faker
"""

import pandas as pd
import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Claim Cause Data Quality Tool",
    page_icon="🔬",
    layout="wide",
)

FOLDER              = Path(__file__).parent
CLASSIFIED_FILE     = FOLDER / "claims_classified.csv"
EXPERIENCE_FILE     = FOLDER / "experience_study.csv"
MODEL_FILE          = FOLDER / "model" / "classifier_pipeline.pkl"

FSC_CATEGORIES = [
    "Cancer / Neoplasm",
    "Musculoskeletal",
    "Mental Health",
    "Cardiovascular",
    "Neurological",
    "Accident / Injury",
    "Other / Unknown",
]

CATEGORY_COLORS = {
    "Cancer / Neoplasm":   "#C0392B",
    "Musculoskeletal":     "#2980B9",
    "Mental Health":       "#8E44AD",
    "Cardiovascular":      "#E67E22",
    "Neurological":        "#16A085",
    "Accident / Injury":   "#D4AC0D",
    "Other / Unknown":     "#7F8C8D",
}

METHOD_COLORS = {
    "Rule-based":  "#2ECC71",
    "Fuzzy match": "#F39C12",
    "ML model":    "#3498DB",
}

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    # Auto-generate data files if they don't exist (e.g. on first Streamlit Cloud boot)
    if not CLASSIFIED_FILE.exists() or not EXPERIENCE_FILE.exists():
        import subprocess, sys
        with st.spinner("Setting up — generating and classifying claims data (first run only)..."):
            subprocess.run([sys.executable, str(FOLDER / "generate_claims.py")], check=True)
            subprocess.run([sys.executable, str(FOLDER / "classifier.py")],      check=True)
            subprocess.run([sys.executable, str(FOLDER / "experience_study.py")], check=True)
    claims     = pd.read_csv(CLASSIFIED_FILE)
    experience = pd.read_csv(EXPERIENCE_FILE)
    return claims, experience


# ── Helpers ────────────────────────────────────────────────────────────────────

def metric_card(label, value, sub="", color="#2C3E50"):
    return (
        f"<div style='background:{color};color:white;border-radius:8px;"
        f"padding:14px 16px;text-align:center;'>"
        f"<div style='font-size:2em;font-weight:bold;'>{value}</div>"
        f"<div style='font-size:0.8em;margin-top:2px;'>{sub}</div>"
        f"<div style='font-size:0.75em;margin-top:8px;font-weight:bold;'>{label}</div>"
        f"</div>"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.title("🔬 Claim Cause Data Quality Tool")
    st.caption(
        "TPD claim experience study — before and after claim cause classification. "
        "Demonstrates the impact of inconsistent data entry on actuarial experience studies."
    )

    with st.expander("ℹ️ About this tool", expanded=False):
        st.markdown(
            "**The problem:** In many life insurance administration systems, claim cause is entered "
            "manually by claims staff with no controlled vocabulary or validation. The same condition "
            "gets recorded dozens of different ways. This corrupts the cause split in experience studies "
            "without affecting the total claim count — making the error silent and easy to miss for years.\n\n"
            "**The approach:** This tool demonstrates a three-layer classification pipeline — "
            "rule-based matching, fuzzy string matching, and a machine learning classifier — "
            "that systematically maps messy cause entries to FSC standard categories. "
            "The before-and-after experience study output shows the direct actuarial impact of "
            "restoring data quality.\n\n"
            "**Scope:** This is a working prototype built for illustration using synthetic TPD claims data. "
            "It demonstrates the problem, the methodology, and the actuarial impact. "
            "A production implementation would connect directly to the administration system, "
            "include a formal actuary sign-off workflow, full audit trail, and ongoing model retraining "
            "as new labelled examples accumulate.\n\n"
            "*Built by Amy Wang, FIAA.*"
        )

    with st.expander("⚠️ Disclaimer", expanded=False):
        st.markdown(
            "This tool uses **synthetic data** generated for illustration purposes only. "
            "It does not contain any real policyholder information.\n\n"
            "The FSC cause categories, expected incidence rates, and A/E ratios shown are "
            "illustrative and loosely based on publicly available Australian TPD industry experience. "
            "They are not intended to represent any specific insurer's actual experience, "
            "and should not be used for pricing, reserving, or any other actuarial purpose.\n\n"
            "The classification pipeline is a working prototype. "
            "A production implementation would require formal validation, an actuary sign-off workflow, "
            "a full audit trail, and compliance with applicable data governance requirements."
        )

    claims, experience = load_data()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_problem, tab_classifier, tab_experience, tab_explorer = st.tabs([
        "📋 The Problem",
        "🤖 The Classifier",
        "📊 Experience Study",
        "🔍 Claims Explorer",
    ])

    # ── TAB 1: The Problem ─────────────────────────────────────────────────────
    with tab_problem:
        st.subheader("The Data Quality Problem")
        st.markdown(
            "In many life insurance administration systems, **claim cause is entered manually** "
            "by claims staff. There is no controlled vocabulary, no validation, and no consistency "
            "checks. The same condition gets recorded dozens of different ways."
        )

        st.markdown("---")
        st.markdown("### What this looks like in practice")
        st.markdown(
            "All of the entries below refer to the **same FSC cause category**. "
            "Without a classifier, grouping these correctly requires manual review of every claim."
        )

        example_col1, example_col2, example_col3 = st.columns(3)

        with example_col1:
            st.markdown(
                "<div style='background:#FADBD8;border-left:4px solid #C0392B;"
                "padding:12px;border-radius:4px;'>"
                "<div style='font-weight:bold;margin-bottom:8px;color:#C0392B;'>"
                "Cancer / Neoplasm</div>"
                "<div style='font-size:0.85em;line-height:1.8;'>"
                "Cancer · ca · CA · Breast CA · lung ca · <em>Caner</em> · "
                "<em>Canncer</em> · NEOPL · malignancy · malig · carc · "
                "TUMOUR · tumor · Ca - lung · Stage 4 cancer · oncology dx"
                "</div></div>",
                unsafe_allow_html=True
            )

        with example_col2:
            st.markdown(
                "<div style='background:#D6EAF8;border-left:4px solid #2980B9;"
                "padding:12px;border-radius:4px;'>"
                "<div style='font-weight:bold;margin-bottom:8px;color:#2980B9;'>"
                "Musculoskeletal</div>"
                "<div style='font-size:0.85em;line-height:1.8;'>"
                "MSK · msk · Back Pain · BACK · spinal · disc injury · "
                "disc prolapse · herniated disc · OA · RA · rheumatoid · "
                "knee replacement · rotator cuff · DDD · chronic pain syndrome"
                "</div></div>",
                unsafe_allow_html=True
            )

        with example_col3:
            st.markdown(
                "<div style='background:#E8DAEF;border-left:4px solid #8E44AD;"
                "padding:12px;border-radius:4px;'>"
                "<div style='font-weight:bold;margin-bottom:8px;color:#8E44AD;'>"
                "Mental Health</div>"
                "<div style='font-size:0.85em;line-height:1.8;'>"
                "MH · depresion · DEPRESSION · Anxiety · psych · PTSD · "
                "Post traumatic stress · bipolar disorder · "
                "Psych condition - not specified · work-related stress · "
                "burn out · nervous breakdown"
                "</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### The downstream impact")
        st.markdown(
            "Inconsistent cause coding doesn't affect the **total** claim count — "
            "it corrupts the **split by cause**. This has a direct flow-on effect:"
        )

        flow_cols = st.columns(5)
        steps = [
            ("📂", "Messy cause field", "Manual entry, no validation"),
            ("➡️", "Wrong grouping", "Claims pile up in 'Other'"),
            ("➡️", "Distorted A/E", "Experience looks fine overall but wrong by cause"),
            ("➡️", "Wrong assumptions", "Cause-specific incidence rates misstated"),
            ("➡️", "Mispriced product", "Premium too low or too high for certain causes"),
        ]
        for col, (icon, title, sub) in zip(flow_cols, steps):
            with col:
                st.markdown(
                    f"<div style='text-align:center;padding:10px;'>"
                    f"<div style='font-size:1.8em;'>{icon}</div>"
                    f"<div style='font-weight:bold;font-size:0.88em;margin-top:6px;'>{title}</div>"
                    f"<div style='font-size:0.78em;color:#666;margin-top:4px;'>{sub}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("### The scale of the problem in this dataset")

        before_other = int(
            experience.loc[experience["cause"] == "Other / Unknown", "actual_before"].values[0]
        )
        after_other  = claims["predicted_cause"].value_counts().get("Other / Unknown", 0)
        rescued      = before_other - after_other

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_card("Total claims", f"{len(claims):,}", "in dataset", "#2C3E50"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("In 'Other' before", f"{before_other:,}", "without classifier", "#C0392B"), unsafe_allow_html=True)
        with m3:
            st.markdown(metric_card("In 'Other' after", f"{after_other:,}", "with classifier", "#27AE60"), unsafe_allow_html=True)
        with m4:
            st.markdown(metric_card("Claims rescued", f"{rescued:,}", "correctly categorised", "#2980B9"), unsafe_allow_html=True)

    # ── TAB 2: The Classifier ──────────────────────────────────────────────────
    with tab_classifier:
        st.subheader("How the Classifier Works")
        st.markdown(
            "The classifier uses **three layers** applied in sequence. "
            "Each layer handles what the previous layer couldn't resolve."
        )

        st.markdown("---")

        layer_col1, layer_col2, layer_col3 = st.columns(3)

        with layer_col1:
            st.markdown(
                "<div style='background:#EAFAF1;border-left:4px solid #2ECC71;"
                "padding:14px;border-radius:4px;height:220px;'>"
                "<div style='font-weight:bold;font-size:1em;color:#27AE60;margin-bottom:8px;'>"
                "Layer 1 — Rule-based</div>"
                "<div style='font-size:0.85em;line-height:1.6;'>"
                "Keyword and regex rules defined by the actuary. "
                "Fast, fully transparent, and auditable. "
                "Every decision can be explained by pointing to a specific rule. "
                "<br><br><strong>Handles:</strong> clear cases, standard terminology"
                "</div></div>",
                unsafe_allow_html=True
            )

        with layer_col2:
            st.markdown(
                "<div style='background:#FEF9E7;border-left:4px solid #F39C12;"
                "padding:14px;border-radius:4px;height:220px;'>"
                "<div style='font-weight:bold;font-size:1em;color:#E67E22;margin-bottom:8px;'>"
                "Layer 2 — Fuzzy Matching</div>"
                "<div style='font-size:0.85em;line-height:1.6;'>"
                "Finds the closest known term even with misspellings or abbreviations. "
                "'Caner' → Cancer. 'depresion' → Mental Health. "
                "Uses similarity scoring — only classifies when confidence is above threshold. "
                "<br><br><strong>Handles:</strong> typos, abbreviations, variations"
                "</div></div>",
                unsafe_allow_html=True
            )

        with layer_col3:
            st.markdown(
                "<div style='background:#EBF5FB;border-left:4px solid #3498DB;"
                "padding:14px;border-radius:4px;height:220px;'>"
                "<div style='font-weight:bold;font-size:1em;color:#2980B9;margin-bottom:8px;'>"
                "Layer 3 — ML Classifier</div>"
                "<div style='font-size:0.85em;line-height:1.6;'>"
                "TF-IDF + Logistic Regression trained on labelled examples. "
                "Handles ambiguous cases that rules and fuzzy matching couldn't resolve. "
                "Returns a confidence score — low confidence claims are flagged for human review. "
                "<br><br><strong>Handles:</strong> ambiguous, novel, complex entries"
                "</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### Classification results on this dataset")

        method_counts = claims["method"].value_counts()
        total = len(claims)

        m_cols = st.columns(len(method_counts))
        for col, (method, count) in zip(m_cols, method_counts.items()):
            color = METHOD_COLORS.get(method, "#666")
            with col:
                st.markdown(
                    metric_card(method, f"{count:,}", f"{count/total*100:.1f}% of claims", color),
                    unsafe_allow_html=True
                )

        st.markdown("")

        review_count = claims["review_flag"].sum()
        if review_count > 0:
            st.warning(f"⚑ {review_count} claims flagged for human review (ML confidence below threshold).")
        else:
            st.success("✓ All claims classified with sufficient confidence. No human review required for this dataset.")

        st.markdown("---")
        st.markdown("### Accuracy vs ground truth")
        st.markdown(
            "_Accuracy = the share of claims where the classifier's predicted FSC category matches "
            "the known true category. It is calculated as: correct predictions ÷ total claims. "
            "In a real system, ground truth is not available — it has to be derived through actuary review. "
            "Here we use the synthetic true_cause field assigned at data generation to validate the classifier._"
        )

        correct  = (claims["predicted_cause"] == claims["true_cause"]).sum()
        accuracy = correct / total

        acc_col1, acc_col2 = st.columns([1, 2])
        with acc_col1:
            st.markdown(
                metric_card("Overall accuracy", f"{accuracy:.1%}", f"{correct}/{total} correct", "#27AE60"),
                unsafe_allow_html=True
            )

        with acc_col2:
            st.markdown("**Accuracy by FSC category:**")
            for cat in FSC_CATEGORIES:
                subset   = claims[claims["true_cause"] == cat]
                cat_acc  = (subset["predicted_cause"] == subset["true_cause"]).mean()
                color    = CATEGORY_COLORS.get(cat, "#666")
                bar_pct  = cat_acc * 100
                st.markdown(
                    f"<div style='margin-bottom:6px;'>"
                    f"<div style='font-size:0.82em;margin-bottom:2px;'>"
                    f"<strong>{cat}</strong> — {cat_acc:.1%} (n={len(subset)})</div>"
                    f"<div style='background:#f0f0f0;border-radius:4px;height:12px;'>"
                    f"<div style='background:{color};width:{bar_pct:.0f}%;height:12px;border-radius:4px;'>"
                    f"</div></div></div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        st.markdown("### The feedback loop — how the tool gets smarter over time")
        st.markdown(
            "Claims flagged for human review are sent to an actuary for confirmation or correction. "
            "Those corrections are added back as new labelled training examples. "
            "The ML model retrains — and gets better at the edge cases it previously struggled with. "
            "Over time, the human review workload shrinks. "
            "**The actuary's domain expertise becomes embedded in the model.**"
        )

        loop_cols = st.columns(4)
        loop_steps = [
            ("📥", "New claims arrive", "Raw, messy cause field"),
            ("🤖", "Classifier runs", "3 layers, uncertain cases flagged"),
            ("👩‍⚕️", "Actuary reviews flags", "Confirms or corrects"),
            ("🔄", "Model retrains", "Learns from corrections, fewer flags next time"),
        ]
        for col, (icon, title, sub) in zip(loop_cols, loop_steps):
            with col:
                st.markdown(
                    f"<div style='text-align:center;background:#f8f9fa;border-radius:8px;"
                    f"padding:14px;'>"
                    f"<div style='font-size:1.8em;'>{icon}</div>"
                    f"<div style='font-weight:bold;font-size:0.88em;margin-top:6px;'>{title}</div>"
                    f"<div style='font-size:0.78em;color:#666;margin-top:4px;'>{sub}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # ── TAB 3: Experience Study ────────────────────────────────────────────────
    with tab_experience:
        st.subheader("TPD Claim Experience Study — Before vs After")
        st.markdown(
            "The same 1,000 claims. The same total A/E. "
            "But the **cause-level picture changes completely** once data quality is restored."
        )

        study = experience.copy()
        study_body = study[study["cause"] != "TOTAL"]
        study_total = study[study["cause"] == "TOTAL"]

        st.markdown("---")
        st.markdown("### A/E ratio by cause")
        st.markdown("*A/E = Actual claims ÷ Expected claims based on pricing assumptions. 100 = exactly as expected.*")
        st.markdown("")

        # Build comparison bar chart using HTML
        for _, row in study_body.iterrows():
            cat   = row["cause"]
            color = CATEGORY_COLORS.get(cat, "#666")
            ae_b  = row["ae_before"]
            ae_a  = row["ae_after"]
            max_v = max(study_body[["ae_before", "ae_after"]].max().max(), 100) * 1.1

            st.markdown(
                f"<div style='margin-bottom:14px;'>"
                f"<div style='font-size:0.9em;font-weight:600;margin-bottom:4px;'>{cat}</div>"
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:3px;'>"
                f"<div style='width:80px;font-size:0.78em;color:#888;'>Before</div>"
                f"<div style='flex:1;background:#f0f0f0;border-radius:4px;height:16px;'>"
                f"<div style='background:#BDC3C7;width:{ae_b/max_v*100:.1f}%;height:16px;border-radius:4px;'></div>"
                f"</div>"
                f"<div style='width:60px;font-size:0.82em;text-align:right;color:#888;'>{ae_b:.1f}</div>"
                f"</div>"
                f"<div style='display:flex;align-items:center;gap:8px;'>"
                f"<div style='width:80px;font-size:0.78em;color:{color};font-weight:600;'>After</div>"
                f"<div style='flex:1;background:#f0f0f0;border-radius:4px;height:16px;'>"
                f"<div style='background:{color};width:{ae_a/max_v*100:.1f}%;height:16px;border-radius:4px;'></div>"
                f"</div>"
                f"<div style='width:60px;font-size:0.82em;text-align:right;font-weight:600;color:{color};'>{ae_a:.1f}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### Full A/E table")

        # Build HTML table
        header = (
            "<tr style='background:#2C3E50;color:white;'>"
            "<th style='padding:10px 12px;text-align:left;'>Cause</th>"
            "<th style='padding:10px 12px;text-align:right;'>Actual (Before)</th>"
            "<th style='padding:10px 12px;text-align:right;'>Expected</th>"
            "<th style='padding:10px 12px;text-align:right;'>A/E Before</th>"
            "<th style='padding:10px 12px;text-align:right;'>Actual (After)</th>"
            "<th style='padding:10px 12px;text-align:right;'>A/E After</th>"
            "<th style='padding:10px 12px;text-align:right;'>Change</th>"
            "</tr>"
        )

        rows_html = ""
        for i, row in study.iterrows():
            is_total = row["cause"] == "TOTAL"
            bg       = "#f0f4f8" if is_total else ("#ffffff" if i % 2 == 0 else "#f9f9f9")
            weight   = "font-weight:bold;" if is_total else ""
            change   = row["ae_change"]
            ch_color = "#C0392B" if change > 0 else "#27AE60" if change < 0 else "#666"
            ch_arrow = "▲" if change > 0 else "▼" if change < 0 else "─"

            rows_html += (
                f"<tr style='background:{bg};{weight}'>"
                f"<td style='padding:9px 12px;border-bottom:1px solid #e5e5e5;'>{row['cause']}</td>"
                f"<td style='padding:9px 12px;text-align:right;border-bottom:1px solid #e5e5e5;'>{int(row['actual_before'])}</td>"
                f"<td style='padding:9px 12px;text-align:right;border-bottom:1px solid #e5e5e5;'>{row['expected_before']:.1f}</td>"
                f"<td style='padding:9px 12px;text-align:right;border-bottom:1px solid #e5e5e5;color:#888;'>{row['ae_before']:.1f}</td>"
                f"<td style='padding:9px 12px;text-align:right;border-bottom:1px solid #e5e5e5;'>{int(row['actual_after'])}</td>"
                f"<td style='padding:9px 12px;text-align:right;border-bottom:1px solid #e5e5e5;font-weight:600;'>{row['ae_after']:.1f}</td>"
                f"<td style='padding:9px 12px;text-align:right;border-bottom:1px solid #e5e5e5;color:{ch_color};font-weight:600;'>{ch_arrow} {abs(change):.1f}</td>"
                f"</tr>"
            )

        table_html = (
            "<div style='overflow-x:auto;'>"
            "<table style='width:100%;border-collapse:collapse;font-family:sans-serif;font-size:0.88em;'>"
            f"<thead>{header}</thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("")
        st.info(
            "**Note:** The total A/E is identical before and after (66.7). "
            "The overall claim volume was never wrong — only the cause split was corrupted. "
            "This is why data quality problems at cause level can go undetected for years "
            "when only total experience is monitored."
        )

        st.markdown("---")
        st.markdown("### Why this matters beyond the numbers")
        st.markdown(
            "Data quality at cause level is not an IT problem. "
            "It sits at the base of every actuarial, financial, and operational process in the business. "
            "A silent error in cause coding propagates upward simultaneously — "
            "and because the **total claim count is always correct**, it can go undetected for years."
        )
        st.markdown("")

        impact_items = [
            ("📊", "Experience Studies",
             "The foundation everything else rests on. Wrong cause split → every downstream function inherits the error."),
            ("💰", "Pricing",
             "Wrong cause A/E → wrong cause-specific incidence assumptions → premiums too high or too low for specific conditions."),
            ("🏦", "Reserving",
             "TPD reserves use cause-specific recovery rates and duration assumptions. Wrong cause coding → wrong reserve adequacy → balance sheet and APRA capital impact."),
            ("🤝", "Reinsurance",
             "Most TPD treaties have cause-specific terms — different retentions, risk premiums, sometimes cause exclusions. Incorrect cause data in bordereau reporting is a treaty compliance issue and may result in under-recovery of reinsurance."),
            ("🛠️", "Product Design",
             "If Mental Health claims look artificially low, product managers may broaden definitions or remove exclusions based on false data — making the product more generous than the actual experience warrants."),
            ("🏛️", "Industry Submissions",
             "FSC and APRA collect cause data for industry benchmarking. If companies submit dirty cause data, the benchmark itself is corrupt — and every company that calibrates to it inherits the error."),
            ("📐", "Capital Modelling",
             "Under APRA's LAGIC framework, stress scenarios are sometimes cause-specific. Wrong cause distribution → wrong stress calibration → potentially insufficient capital held."),
            ("⚙️", "Claims Operations",
             "Many insurers route claims to specialist assessors by cause (e.g. mental health claims to psych-trained staff). Wrong coding means wrong routing — affecting claim outcomes and the customer experience."),
        ]

        row1 = st.columns(4)
        row2 = st.columns(4)
        for col, (icon, title, desc) in zip(list(row1) + list(row2), impact_items):
            with col:
                st.markdown(
                    f"<div style='background:#f8f9fa;border-radius:8px;padding:12px;"
                    f"margin-bottom:8px;height:170px;'>"
                    f"<div style='font-size:1.4em;'>{icon}</div>"
                    f"<div style='font-weight:bold;font-size:0.88em;margin-top:6px;"
                    f"margin-bottom:4px;'>{title}</div>"
                    f"<div style='font-size:0.78em;color:#444;line-height:1.5;'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # ── TAB 4: Claims Explorer ─────────────────────────────────────────────────
    with tab_explorer:
        st.subheader("Claims Explorer")
        st.markdown("Browse individual claims to see how the classifier processed each one.")

        st.markdown("---")

        col_filter1, col_filter2, col_filter3 = st.columns(3)
        with col_filter1:
            method_filter = st.multiselect(
                "Classification method",
                options=["Rule-based", "Fuzzy match", "ML model"],
                default=["Rule-based", "Fuzzy match", "ML model"]
            )
        with col_filter2:
            cause_filter = st.multiselect(
                "Predicted cause",
                options=FSC_CATEGORIES,
                default=FSC_CATEGORIES
            )
        with col_filter3:
            correct_filter = st.radio(
                "Correctness",
                ["All", "Correct only", "Incorrect only"],
                horizontal=True
            )

        filtered = claims[
            claims["method"].isin(method_filter) &
            claims["predicted_cause"].isin(cause_filter)
        ].copy()

        if correct_filter == "Correct only":
            filtered = filtered[filtered["predicted_cause"] == filtered["true_cause"]]
        elif correct_filter == "Incorrect only":
            filtered = filtered[filtered["predicted_cause"] != filtered["true_cause"]]

        st.markdown(f"Showing **{len(filtered):,}** claims")
        st.markdown("")

        display_cols = ["claim_id", "age_at_claim", "gender", "sum_insured",
                        "raw_cause", "predicted_cause", "true_cause", "method", "confidence"]

        display_df = filtered[display_cols].copy()
        display_df["correct"] = display_df["predicted_cause"] == display_df["true_cause"]
        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.0%}")

        st.dataframe(
            display_df.head(200),
            use_container_width=True,
            column_config={
                "claim_id":        st.column_config.TextColumn("Claim ID", width="small"),
                "age_at_claim":    st.column_config.NumberColumn("Age", width="small"),
                "gender":          st.column_config.TextColumn("Gender", width="small"),
                "sum_insured":     st.column_config.NumberColumn("Sum Insured", format="$%d"),
                "raw_cause":       st.column_config.TextColumn("Raw Cause (entered)", width="medium"),
                "predicted_cause": st.column_config.TextColumn("Predicted Cause", width="medium"),
                "true_cause":      st.column_config.TextColumn("True Cause", width="medium"),
                "method":          st.column_config.TextColumn("Method", width="small"),
                "confidence":      st.column_config.TextColumn("Confidence", width="small"),
                "correct":         st.column_config.CheckboxColumn("Correct?", width="small"),
            },
            hide_index=True,
        )

        if len(filtered) > 200:
            st.caption(f"Showing first 200 of {len(filtered):,} claims. Use filters to narrow results.")


if __name__ == "__main__":
    main()

