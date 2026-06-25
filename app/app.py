"""
TrustLens — Human-Centered XAI Study on Trust Calibration in AI-Assisted Loan Decisions
Streamlit application.

Design:
  - Between-subjects: each participant is randomly assigned ONE condition (A/B/C)
  - Per-case survey: trust, fairness, and accept/override captured for every case
  - Trust-shift measure: trust rated BEFORE and AFTER the explanation (B and C)
  - Six fixed study cases shown to all participants

Author: Vasudev Pant — MS Business Analytics, Arizona State University
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import os
import random
from datetime import datetime

# Condition rotation — sequential counterbalancing (A → B → C → A ...)
COUNTER_FILE = os.path.join("responses", "assignment_counter.json")
CONDITIONS    = ["A", "B", "C"]

def get_next_condition():
    """
    Assigns the next condition in strict rotation so every three participants
    receive one of each condition. Guarantees balanced cell sizes across the
    pilot study without relying on random chance.
    """
    os.makedirs("responses", exist_ok=True)
    try:
        with open(COUNTER_FILE) as f:
            count = json.load(f).get("count", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        count = 0
    condition = CONDITIONS[count % 3]
    with open(COUNTER_FILE, "w") as f:
        json.dump({"count": count + 1}, f)
    return condition

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(page_title="TrustLens Study", page_icon="🔍", layout="centered")

# ----------------------------------------------------------------------------
# Styling — quiet, academic, focused (one accent colour, generous whitespace)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { max-width: 760px; padding-top: 2rem; }
    h1, h2, h3 { color: #1a2e4a; font-family: 'Georgia', serif; }
    .stProgress > div > div > div > div { background-color: #1a7a6e; }

    .case-card {
        background: #f7f9fb; border: 1px solid #dce4ec; border-radius: 10px;
        padding: 1.2rem 1.5rem; margin: 1rem 0;
        animation: fadeIn 0.4s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .decision-approve {
        background:#e8f5e9; border-left:5px solid #2e7d32; padding:0.8rem 1.2rem;
        border-radius:6px; font-size:1.15rem; font-weight:600; color:#1b5e20;
    }
    .decision-reject {
        background:#fdecea; border-left:5px solid #c62828; padding:0.8rem 1.2rem;
        border-radius:6px; font-size:1.15rem; font-weight:600; color:#b71c1c;
    }
    .profile-label { color:#5a6b7b; font-size:0.85rem; }
    .profile-value { color:#1a2e4a; font-weight:600; font-size:1rem; }

    /* Case progress dots */
    .dots-row { display:flex; gap:8px; justify-content:center; margin: 0.5rem 0 1.2rem 0; }
    .dot {
        width:11px; height:11px; border-radius:50%; background:#dce4ec;
        transition: background 0.3s ease, transform 0.3s ease;
    }
    .dot.done { background:#1a7a6e; }
    .dot.current { background:#1a2e4a; transform: scale(1.3); }

    /* Confidence meter */
    .conf-wrap { margin-top:0.5rem; }
    .conf-track {
        background:#e3e8ee; border-radius:6px; height:10px; width:100%;
        overflow:hidden; margin-top:4px;
    }
    .conf-fill {
        height:100%; border-radius:6px;
        transition: width 0.6s ease;
    }
    .conf-fill.approve { background:#2e7d32; }
    .conf-fill.reject  { background:#c62828; }
    .conf-caption { font-size:0.8rem; color:#5a6b7b; margin-top:3px; }

    /* Sidebar study info */
    .sidebar-block {
        background:#f7f9fb; border:1px solid #dce4ec; border-radius:8px;
        padding:0.9rem 1rem; font-size:0.85rem; color:#3a4a5a; line-height:1.5;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load model artifacts (cached so they load once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # 1. Get the directory where app.py lives
    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Point directly to the 'model' folder inside it
    model_dir = os.path.join(app_dir, "model")
    
    # 3. Build absolute paths to each file
    model_path = os.path.join(model_dir, "xgboost_model.pkl")
    threshold_path = os.path.join(model_dir, "decision_threshold.pkl")
    features_path = os.path.join(model_dir, "feature_names.pkl")
    
    # Check if study_cases.json is also in the app directory or app/data folder
    data_path = os.path.join(app_dir, "study_cases.json")
    if not os.path.exists(data_path):
        data_path = os.path.join(app_dir, "data", "study_cases.json")

    # 4. Load the files
    model = joblib.load(model_path)
    threshold = joblib.load(threshold_path)
    features = joblib.load(features_path)
    
    with open(data_path, "r") as f:
        cases = json.load(f)
        
    explainer = shap.TreeExplainer(model)
    
    return model, threshold, features, cases, explainer



# Human-readable labels for the loan profile display
PROFILE_LABELS = {
    "checking_account_status": "Checking Account",
    "duration_months": "Loan Duration (months)",
    "credit_history": "Credit History",
    "purpose": "Loan Purpose",
    "credit_amount": "Credit Amount (DM)",
    "savings_account_bonds": "Savings Account",
    "employment_since": "Employment Duration",
    "installment_rate_percent_disposable_income": "Installment Rate (%)",
    "personal_status_sex": "Personal Status / Sex",
    "other_debtors_guarantors": "Other Debtors",
    "residence_since_years": "Residence (years)",
    "property": "Property",
    "age_years": "Age",
    "other_installment_plans": "Other Installment Plans",
    "housing": "Housing",
    "existing_credits_at_bank": "Existing Credits",
    "job": "Job",
    "people_liable_for_maintenance": "Dependents",
    "telephone": "Telephone",
    "foreign_worker": "Foreign Worker",
}

# Which profile fields to show participants (the salient ones)
DISPLAY_FIELDS = [
    "age_years", "personal_status_sex", "credit_amount", "duration_months",
    "purpose", "checking_account_status", "savings_account_bonds",
    "credit_history", "employment_since", "housing", "job",
]

# Lowercase, reader-friendly feature names used inside the prose explanation
TEXT_FEATURE_NAMES = {
    "checking_account_status": "checking account status",
    "duration_months": "loan duration",
    "credit_history": "credit history",
    "purpose": "loan purpose",
    "credit_amount": "requested credit amount",
    "savings_account_bonds": "savings holdings",
    "employment_since": "employment history",
    "installment_rate_percent_disposable_income": "installment-to-income rate",
    "personal_status_sex": "personal status",
    "other_debtors_guarantors": "guarantor arrangement",
    "residence_since_years": "length of residence",
    "property": "property holdings",
    "age_years": "age",
    "other_installment_plans": "other installment commitments",
    "housing": "housing situation",
    "existing_credits_at_bank": "existing credit lines",
    "job": "employment category",
    "people_liable_for_maintenance": "number of dependents",
    "telephone": "registered telephone",
    "foreign_worker": "residency status",
}

LIKERT = ["1 — Strongly disagree", "2 — Disagree", "3 — Neutral",
          "4 — Agree", "5 — Strongly agree"]


# ----------------------------------------------------------------------------
# Helper: generate text explanation from SHAP values
# ----------------------------------------------------------------------------
def text_explanation(shap_vals, feature_names, readable_row, decision, confidence,
                     case_number):
    """
    Produces a measured, analytical account of the model's recommendation. To
    avoid the six cases reading as one template with swapped values, the phrasing
    is varied per case: each case number deterministically selects a distinct set
    of sentence styles, so the wording differs case to case but stays identical
    across reruns of the same case.
    """
    shap_vals = np.asarray(shap_vals, dtype=float)
    top3 = np.argsort(np.abs(shap_vals))[::-1][:3]
    dw = "approval" if decision == "APPROVE" else "rejection"
    decverb = "approve the application" if decision == "APPROVE" else \
              "reject the application"

    supports, opposes = [], []
    for idx in top3:
        feat = TEXT_FEATURE_NAMES.get(feature_names[idx],
                                      feature_names[idx].replace("_", " "))
        val = readable_row.get(feature_names[idx], "")
        s = float(shap_vals[idx])
        toward_approve = s < 0
        item = (feat, val)
        if (decision == "APPROVE" and toward_approve) or \
           (decision == "REJECT" and not toward_approve):
            supports.append(item)
        else:
            opposes.append(item)

    def conf_phrase(c, article=True):
        if c >= 0.8:
            return "a high degree of confidence" if article else "high confidence"
        if c >= 0.65:
            return "moderate confidence"
        return "a limited degree of confidence" if article else "limited confidence"

    pct = int(confidence * 100)
    openers = [
        f"**Recommendation: {decision}.** This determination was made with "
        f"{conf_phrase(confidence)} ({pct}%).",
        f"**The model recommends {decision}.** Its confidence in this outcome is "
        f"{conf_phrase(confidence, article=False)} ({pct}%).",
        f"**Outcome: {decision}.** The system arrived at this with "
        f"{conf_phrase(confidence)} ({pct}%).",
        f"**Assessment: {decision}.** This conclusion carries "
        f"{conf_phrase(confidence)} ({pct}%).",
        f"**Decision: {decision}.** The model holds "
        f"{conf_phrase(confidence, article=False)} ({pct}%) in this recommendation.",
        f"**The recommendation is to {decverb}.** This was reached with "
        f"{conf_phrase(confidence)} ({pct}%).",
    ]
    primary = [
        "The assessment is anchored above all in the applicant's {f} ({v}), the "
        "factor carrying the greatest weight toward {dw}.",
        "Foremost among the drivers is the applicant's {f} ({v}), which the model "
        "treats as the principal basis for {dw}.",
        "The single strongest influence is the applicant's {f} ({v}), weighing "
        "decisively toward {dw}.",
        "What most shapes this outcome is the applicant's {f} ({v}), the leading "
        "consideration behind the case for {dw}.",
        "At the centre of the decision lies the applicant's {f} ({v}), the dominant "
        "factor pointing toward {dw}.",
        "The model leans most heavily on the applicant's {f} ({v}) in building its "
        "case for {dw}.",
    ]
    support = [
        "Reinforcing this, the applicant's {f} ({v}) points the same way.",
        "The applicant's {f} ({v}) adds weight in the same direction.",
        "This is supported further by the applicant's {f} ({v}).",
        "Their {f} ({v}) lends additional support to the same conclusion.",
        "A second consideration, the applicant's {f} ({v}), aligns with this judgement.",
        "The applicant's {f} ({v}) corroborates the leading factor.",
    ]
    counter = [
        "Set against this, the applicant's {f} ({v}) pulled in the opposite "
        "direction, though not enough to change the outcome.",
        "The model did weigh a counter-signal — the applicant's {f} ({v}) — but "
        "found it insufficient to alter the result.",
        "One factor resisted this conclusion, the applicant's {f} ({v}), yet its "
        "pull was ultimately overridden.",
        "A tension exists: the applicant's {f} ({v}) argued otherwise, but the "
        "dominant factors prevailed.",
        "Notably, the applicant's {f} ({v}) ran counter to the decision, without "
        "being decisive.",
        "The applicant's {f} ({v}) offered some resistance to this outcome, though "
        "it was ultimately outweighed.",
    ]

    v = (int(case_number) - 1) % 6  # deterministic variant per case
    parts = [openers[v]]
    if supports:
        parts.append(primary[v].format(f=supports[0][0], v=supports[0][1], dw=dw))
        if len(supports) > 1:
            parts.append(support[v].format(f=supports[1][0], v=supports[1][1]))
    if opposes:
        parts.append(counter[v].format(f=opposes[0][0], v=opposes[0][1]))
    return " ".join(parts)


# ----------------------------------------------------------------------------
# Helper: SHAP waterfall figure
# ----------------------------------------------------------------------------
def shap_waterfall(shap_vals, base_value, readable_row, feature_names,
                   decision, confidence, n=8):
    """
    Publication-quality SHAP waterfall, built manually with matplotlib for full
    control over fonts, axis labels, the f(x) and E[f(x)] reference lines, and
    consistent rendering inside Streamlit (the default SHAP plotter renders
    intermittently on reruns; this does not).
    """
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    shap_vals = np.asarray(shap_vals, dtype=float)
    order = np.argsort(np.abs(shap_vals))[::-1]
    top_idx = order[:n]
    other_idx = order[n:]
    other_sum = float(np.sum(shap_vals[other_idx])) if len(other_idx) else 0.0

    items = [(PROFILE_LABELS.get(feature_names[i], feature_names[i].replace("_", " ")),
              float(shap_vals[i]),
              str(readable_row.get(feature_names[i], ''))) for i in top_idx]
    if len(other_idx):
        items.append((f"{len(other_idx)} other features", other_sum, ""))
    items = items[::-1]  # strongest on top

    fx = base_value + float(np.sum(shap_vals))
    APPROVE_BLUE = "#1f6feb"
    REJECT_RED = "#d1242f"

    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    # Pre-compute cumulative positions and x-range
    all_vals = [base_value, fx]
    cumulative = base_value
    positions, yticks, yticklabels = [], [], []
    y = 0
    for label, val, fval in items:
        positions.append((y, cumulative, val))
        cumulative += val
        all_vals.append(cumulative)
        disp = f"{label}" + (f"  =  {fval}" if fval else "")
        yticks.append(y)
        yticklabels.append(disp)
        y += 1

    xmin, xmax = min(all_vals), max(all_vals)
    span = (xmax - xmin) or 1.0
    pad = span * 0.20
    ax.set_xlim(xmin - pad, xmax + pad)

    # Draw bars with number labels
    for (yy, start, val), (label, _, fval) in zip(positions, items):
        color = REJECT_RED if val > 0 else APPROVE_BLUE
        ax.barh(yy, val, left=start, color=color, edgecolor='white',
                height=0.62, zorder=3)
        sign = "+" if val >= 0 else ""
        if abs(val) >= span * 0.09:  # label inside only if bar is wide enough
            ax.text(start + val / 2, yy, f"{sign}{val:.2f}", ha='center', va='center',
                    color='white', fontsize=9, fontweight='bold', zorder=5)
        else:  # otherwise place it just outside the bar
            offset = span * 0.012
            ax.text(start + val + (offset if val >= 0 else -offset), yy,
                    f"{sign}{val:.2f}", ha='left' if val >= 0 else 'right',
                    va='center', color='#333', fontsize=8.5, fontweight='bold', zorder=5)

    # E[f(x)] baseline (dotted) and f(x) decision line (dashed-dotted)
    ax.axvline(base_value, color='#888', linestyle=':', linewidth=1.6, zorder=2)
    ax.text(base_value, y - 0.25, f" E[f(x)] = {base_value:.2f}", color='#555',
            fontsize=9, ha='left', va='bottom', style='italic')
    ax.axvline(fx, color='#1a2e4a', linestyle=(0, (2, 2)), linewidth=1.9, zorder=2)
    ax.text(fx, -0.95, f"f(x) = {fx:.2f}", color='#1a2e4a',
            fontsize=10, ha='center', va='top', fontweight='bold')

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9.5)
    ax.set_xlabel("Model risk score   ·   lower (left) = more likely APPROVE   |   "
                  "higher (right) = more likely REJECT", fontsize=9.5, color='#333', labelpad=8)
    ax.set_ylabel("Applicant features, ordered by influence (most influential at top)",
                  fontsize=9.5, color='#333', labelpad=8)
    ax.set_title(f"Why the AI chose {decision}   ·   {int(confidence*100)}% confident",
                 fontsize=13, fontweight='bold', color='#1a2e4a', pad=14)

    legend_elems = [
        mpatches.Patch(color=APPROVE_BLUE, label='Feature pushes toward APPROVE (lowers risk score)'),
        mpatches.Patch(color=REJECT_RED, label='Feature pushes toward REJECT (raises risk score)'),
        Line2D([0], [0], color='#888', linestyle=':',
               label='E[f(x)]  ·  average prediction across all applicants'),
        Line2D([0], [0], color='#1a2e4a', linestyle=(0, (2, 2)),
               label='f(x)  ·  model prediction for this applicant'),
    ]
    ax.legend(handles=legend_elems, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=2, fontsize=8.5, framealpha=0.95)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='-', alpha=0.15, zorder=0)
    ax.set_ylim(-1.7, y - 0.1)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Session state initialisation
# ----------------------------------------------------------------------------
def init_session():
    if "stage" not in st.session_state:
        st.session_state.stage = "welcome"
    if "condition" not in st.session_state:
        # Sequential counterbalancing: A → B → C → A → B → C ...
        # Guarantees balanced cell sizes across the pilot study regardless of
        # when participants arrive. The researcher key overrides this for testing.
        st.session_state.condition = get_next_condition()
    if "participant_id" not in st.session_state:
        st.session_state.participant_id = datetime.now().strftime("P%Y%m%d%H%M%S") + str(random.randint(10, 99))
    if "case_index" not in st.session_state:
        st.session_state.case_index = 0
    if "responses" not in st.session_state:
        st.session_state.responses = []


# ----------------------------------------------------------------------------
# Save a participant's responses to CSV
# ----------------------------------------------------------------------------
def save_responses():
    os.makedirs("responses", exist_ok=True)
    df = pd.DataFrame(st.session_state.responses)
    fname = f"responses/{st.session_state.participant_id}.csv"
    df.to_csv(fname, index=False)
    return fname


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    init_session()
    model, threshold, features, cases, explainer = load_artifacts()
    cond = st.session_state.condition

    # ---- WELCOME / CONSENT ------------------------------------------------
    if st.session_state.stage == "welcome":
        st.title("TrustLens")
        st.subheader("How do people decide when to trust an AI?")
        st.markdown("""
        Thank you for taking part in this research.

        Artificial intelligence increasingly advises people on high-stakes
        decisions — who receives a loan, who is shortlisted for a role, how risk
        is judged. Yet a good decision depends not only on how capable the AI is,
        but on whether the person using it knows **when to rely on it and when to
        think for themselves**. This study explores exactly that question.

        For the next few minutes, you will step into the role of a **loan officer**.
        You will review six real applicant profiles. For each, an AI system will
        recommend whether to **approve** or **reject** the application and show how
        **confident** it is. Some recommendations will also include an account of
        *why* the AI reached its decision.

        After each case you will share how much you would rely on the AI, and then
        make the final call yourself — accepting its recommendation or deciding
        differently. There are **no right or wrong answers**; your honest,
        considered judgement is precisely what makes this research valuable.
        """)

        st.markdown("""
        <div style="background:#f7f9fb;border-left:4px solid #1a7a6e;
        padding:0.9rem 1.2rem;border-radius:6px;font-size:0.92rem;color:#3a4a5a;">
        <b>What to expect</b><br>
        • About 10–15 minutes to complete<br>
        • Six short applicant cases to review<br>
        • Fully anonymous — no personal information is collected
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.caption("Participation is voluntary and your responses are used only for "
                   "this research. By continuing, you consent to take part.")
        st.info("When you are ready, click **Begin**.")

        # --- Researcher-only condition selector ---
        # Hidden by default. Appears ONLY when the URL includes ?researcher=trustlens
        # Participants and reviewers using the normal link never see this panel.
        if st.query_params.get("researcher", "") == "trustlens":
            with st.expander("Researcher options (testing only)"):
                st.caption("Visible only via the researcher access key. Participants "
                           "are assigned automatically by sequential rotation.")
                choice = st.radio(
                    "Assign condition",
                    ["Rotation (default)",
                     "A — Minimal (decision only)",
                     "B — Text explanation",
                     "C — Visual SHAP explanation"],
                    index=0, key="tester_cond")
                mapping = {"A — Minimal (decision only)": "A",
                           "B — Text explanation": "B",
                           "C — Visual SHAP explanation": "C"}
                if choice in mapping:
                    st.session_state.condition = mapping[choice]
                    st.caption(f"Condition forced to **{mapping[choice]}** for this session.")

        if st.button("Begin", type="primary"):
            st.session_state.stage = "study"
            st.rerun()

    # ---- STUDY ------------------------------------------------------------
    elif st.session_state.stage == "study":
        idx = st.session_state.case_index
        case = cases[idx]
        total = len(cases)

        # --- Sidebar: quiet study info, no hypothesis leakage ---
        with st.sidebar:
            st.markdown("### About this study")
            st.markdown(
                '<div class="sidebar-block">'
                f'You are reviewing loan case <b>{idx + 1} of {total}</b>.<br><br>'
                'For each case you will rate how much you trust the AI\'s '
                'recommendation, then accept or override it.<br><br>'
                'There are no right or wrong answers — please respond honestly.'
                '</div>', unsafe_allow_html=True)

        # --- Progress dots ---
        dots_html = '<div class="dots-row">'
        for i in range(total):
            cls = "dot done" if i < idx else ("dot current" if i == idx else "dot")
            dots_html += f'<div class="{cls}"></div>'
        dots_html += '</div>'
        st.markdown(dots_html, unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:#5a6b7b; font-size:0.85rem;'>"
                    f"Case {idx + 1} of {total}</p>", unsafe_allow_html=True)
        st.markdown(f"### Loan Applicant {idx + 1}")

        # --- Loan profile card ---
        st.markdown('<div class="case-card">', unsafe_allow_html=True)
        cols = st.columns(2)
        readable = case["readable"]
        for i, field in enumerate(DISPLAY_FIELDS):
            if field in readable:
                label = PROFILE_LABELS.get(field, field)
                raw_val = str(readable[field])
                # Capitalise the first letter for a polished, consistent presentation
                disp_val = raw_val[:1].upper() + raw_val[1:] if raw_val else raw_val
                with cols[i % 2]:
                    st.markdown(f'<div class="profile-label">{label}</div>'
                                f'<div class="profile-value">{disp_val}</div>',
                                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(
            "Amounts are shown in DM (Deutsche Mark), the former currency of Germany, "
            "which was replaced by the Euro (EUR) in 2002. The figures reflect the "
            "original German Credit dataset."
        )
        decision = case["ai_decision"]
        confidence = case["ai_confidence"]
        css_class = "decision-approve" if decision == "APPROVE" else "decision-reject"
        fill_class = "approve" if decision == "APPROVE" else "reject"
        conf_pct = int(confidence * 100)

        st.markdown(f'<div class="{css_class}">AI Recommendation: {decision}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="conf-wrap">'
            f'<div class="conf-track"><div class="conf-fill {fill_class}" '
            f'style="width:{conf_pct}%;"></div></div>'
            f'<div class="conf-caption">Confidence: {conf_pct}%</div>'
            f'</div>', unsafe_allow_html=True)
        st.write("")

        # --- INITIAL trust rating (before explanation) ---
        st.markdown("**Based on the recommendation alone, how much would you rely on "
                    "the AI's judgement for this applicant?**")
        st.caption("Consider how comfortable you would feel acting on this "
                   "recommendation before seeing any further detail.")
        initial_trust = st.radio("Initial trust", LIKERT, index=2, key=f"init_{idx}",
                                 label_visibility="collapsed")

        # --- Explanation (conditions B and C only; control condition A shows none) ---
        explanation_shown = False
        if cond in ("B", "C"):
            # A checkbox persists across Streamlit reruns (a button does not), which
            # is what guarantees the explanation stays visible once revealed. The
            # SHAP values are computed fresh and the chart is built manually, so it
            # renders identically every time for all six cases.
            reveal = st.checkbox("Show the AI's reasoning", key=f"reveal_{idx}")
            if reveal:
                explanation_shown = True
                st.divider()
                enc_df = pd.DataFrame([case["encoded"]], columns=features)
                sv = np.asarray(explainer.shap_values(enc_df))
                if sv.ndim > 1:
                    sv = sv[0]

                if cond == "B":
                    st.markdown("#### AI Explanation")
                    st.markdown(text_explanation(sv, features, case["readable"],
                                                 decision, confidence,
                                                 case["case_number"]))
                    st.caption(
                        "This explanation identifies the factors that most shaped the "
                        "AI's recommendation for this applicant and how each one "
                        "influenced the assessed credit risk."
                    )

                elif cond == "C":
                    st.markdown("#### AI Explanation")
                    base = float(np.ravel(explainer.expected_value)[0])
                    fig = shap_waterfall(sv, base, case["readable"], features,
                                         decision, confidence)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.markdown("**How to read this chart**")
                    st.markdown(
                        "- **Each bar is one applicant feature.** The value beside the "
                        "name is this applicant's actual value for it.\n"
                        "- **Blue bars** push the model toward **approval** (they lower "
                        "the risk score); **red bars** push it toward **rejection** "
                        "(they raise it).\n"
                        "- **Longer bars matter more** — the length shows how strongly "
                        "that feature influenced the outcome.\n"
                        "- The chart reads left to right, starting from **E[f(x)]** — "
                        "the model's *average* prediction across all applicants.\n"
                        "- Each feature then shifts the score until it reaches **f(x)** "
                        "(the dotted line) — the model's prediction *for this specific "
                        "applicant*.\n"
                        "- In short: **E[f(x)]** is the typical applicant; **f(x)** is "
                        "this one."
                    )

        # --- The rest of the survey appears once ready ---
        ready_for_survey = (cond == "A") or explanation_shown

        if ready_for_survey:
            st.divider()

            # FINAL trust (after explanation) — only meaningful for B and C
            if cond in ("B", "C"):
                st.markdown("**Now that you have seen the AI's reasoning, how much "
                            "would you rely on its judgement for this applicant?**")
                st.caption("It is completely fine if your view stayed the same or "
                           "changed — either tells us something useful.")
                final_trust = st.radio("Final trust", LIKERT, index=2, key=f"final_{idx}",
                                       label_visibility="collapsed")
            else:
                final_trust = initial_trust  # no explanation in Condition A

            st.markdown("**I have a clear sense of why the AI arrived at this "
                        "recommendation.**")
            understand = st.radio("Understand", LIKERT, index=2, key=f"und_{idx}",
                                  label_visibility="collapsed")

            st.markdown("**This recommendation reflects a fair assessment of the "
                        "applicant, independent of personal characteristics such as "
                        "age or gender.**")
            fairness = st.radio("Fairness", LIKERT, index=2, key=f"fair_{idx}",
                                label_visibility="collapsed")

            st.markdown("**Acting as the loan officer, what is your final call on "
                        "this application?**")
            st.caption("You may follow the AI's recommendation or substitute your "
                       "own judgement.")
            action = st.radio("Action", ["Accept the AI's recommendation",
                                          "Override and decide differently"],
                              key=f"act_{idx}", label_visibility="collapsed")

            if st.button("Next case →", type="primary", key=f"next_{idx}"):
                st.session_state.responses.append({
                    "participant_id": st.session_state.participant_id,
                    "condition": cond,
                    "case_number": case["case_number"],
                    "ai_decision": decision,
                    "ai_confidence": confidence,
                    "initial_trust": int(initial_trust[0]),
                    "final_trust": int(final_trust[0]),
                    "trust_shift": int(final_trust[0]) - int(initial_trust[0]),
                    "understand": int(understand[0]),
                    "fairness": int(fairness[0]),
                    "action": "override" if "Override" in action else "accept",
                    "timestamp": datetime.now().isoformat(),
                })
                st.session_state.case_index += 1
                if st.session_state.case_index >= total:
                    st.session_state.stage = "final"
                st.rerun()

    # ---- FINAL OPEN-ENDED + SUBMIT ----------------------------------------
    elif st.session_state.stage == "final":
        st.title("Two last reflections")
        st.markdown("You have reviewed all six cases — thank you. Before you finish, "
                    "we would value your perspective in your own words.")

        st.markdown("**1. When an AI gave you a recommendation, what made you trust it "
                    "or hold back? Did anything about how it explained itself change "
                    "your mind?**")
        open_response = st.text_area(
            "Reflection 1", height=140,
            placeholder="There are no wrong answers — even a sentence or two is "
                        "genuinely helpful.",
            label_visibility="collapsed")

        st.markdown("**2. Was there a moment when you wanted more explanation than you "
                    "were given? If so, what would have helped you decide?**")
        wanted_more = st.text_area(
            "Reflection 2", height=140,
            placeholder="For example, a particular case where you wished the AI had "
                        "told you more about its reasoning.",
            label_visibility="collapsed")

        if st.button("Submit my responses", type="primary"):
            for r in st.session_state.responses:
                r["open_response"] = open_response
                r["wanted_more_explanation"] = wanted_more
            fname = save_responses()
            st.session_state.stage = "thanks"
            st.rerun()

    # ---- THANK YOU --------------------------------------------------------
    elif st.session_state.stage == "thanks":
        st.title("Thank you")
        st.success("Your responses have been recorded.")
        st.markdown("""
        Thank you for contributing to this research.

        This study explored how different styles of AI explanation affect the way
        people trust and act on AI recommendations. Your participation helps us
        understand how to design AI systems that support better human decisions.
        """)
        st.balloons()


if __name__ == "__main__":
    main()
