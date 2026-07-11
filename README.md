# TrustLens

**Trust Calibration, Decision Agency, and Fairness Perception Across Explanation Modalities in AI-Assisted Loan Decision-Making**

## Live Study

**Try it here:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://trustlens-study.streamlit.app/)

TrustLens is a live, deployed human-subjects study. Visitors are assigned one of three explanation conditions (minimal, text, or visual) and review six AI-assisted loan decisions, rating their trust before and after seeing 
the AI's reasoning.

> ***Note for reviewers:*** ***Each visitor is randomly assigned one condition. To explore all three explanation styles, you may revisit the link in a new session.***
---

## Research Questions

**RQ1 — Trust Calibration** : ***How do different AI explanation modalities (minimal, textual, and visual) influence the accuracy of user trust calibration — defined as the alignment between expressed confidence and actual AI correctness — in high-stakes loan decision-making?***

**RQ2 — Decision Agency** : ***Does the type of AI explanation modality influence user override behavior, independent of stated trust levels, revealing a gap between what users say they trust and what they actually act on?***

**RQ3 — Fairness Perception** : ***How does exposure to feature-level SHAP explanations — which may surface demographic proxies — affect users' perceived algorithmic fairness, and does fairness perception moderate trust calibration?***

---

## The Three Explanation Conditions

| Condition | What the user sees |
|---|---|
| A — Minimal | Approve / Reject + confidence score only |
| B — Text | Natural language explanation of the decision |
| C — Visual | SHAP-based feature contribution chart |

---

## What TrustLens Measures?

Much research on AI explanations asks a single question: do explanations make people trust AI more? TrustLens takes a narrower but more revealing angle — looking at trust *quality* and *behavior* rather than trust *quantity*. It brings together four measures that are usually studied in isolation:

1. **Trust Calibration Metric** — a three-component measure (calibration score, overtrust index, undertrust index) capturing whether a user's confidence matches the AI's actual correctness on each case, rather than whether they trust the AI more or less overall.
2. **Trust Shift** — trust is measured *before* and *after* the explanation is revealed, quantifying how much each explanation format actually moves a participant's trust.
3. **Decision Agency (Override Behavior)** — captures what users *do*, not only what they *say*, surfacing the explanation–action gap.
4. **Fairness Perception under Explanation Exposure** — tests whether the visibility of demographic-proxy features in SHAP explanations affects perceived algorithmic fairness, and whether fairness perception moderates trust calibration.

---

## Tech Stack

- **Model** — XGBoost classifier
- **Explainability** — SHAP (TreeExplainer)
- **Dataset** — German Credit Dataset (UCI)
- **App** — Streamlit
- **Analysis** — pandas, scipy, seaborn, matplotlib

---

## Project Structure

```
trustlens/
├── data/              # Raw and processed dataset
├── models/            # Saved XGBoost model
├── notebooks/         # EDA, model training, SHAP exploration
├── app/               # Streamlit app
│   ├── conditions/    # One file per explanation condition
│   └── utils/         # Model, SHAP, and calibration helpers
├── study/             # Loan cases and participant responses
├── analysis/          # Statistical analysis and charts
└── report/            # Mini research report
```

---

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/trustlens.git
cd trustlens
python -m venv venv
venv\Scripts\activate.bat        # Windows
pip install -r requirements.txt
streamlit run app/main.py
```

---

## User Study Design

- **Participants** — 15 to 20 (Ongoing)
- **Task** — Review 6 loan applicant profiles and evaluate AI recommendations
- **Measures** — 7-item Likert survey + override decision + open-ended response
- **Analysis** — Kruskal-Wallis, Mann-Whitney pairwise, chi-square for override rates

---

## Related Publications (Co-Authored with Dr. Narendra Kumar)

- *"Transparency and Accountability in AI Decision-Making: Methods and Challenges"*
- *"Performance Impact of Machine Unlearning in AI Systems"*

Both accepted in: **Machine Unlearning: Ethical and Responsible AI, Machine Learning and Retraining (Both of the papers are peer reviewed and will be added as the chapters of this book (Currently, the book is under process))**

Wiley-Scrivener | Scopus-indexed | forthcoming

---

## Theoretical Grounding

- ***Schemmer et al. (2023) — Appropriate Reliance on AI Advice: Conceptualization and the Effect of Explanations***
- ***Vasconcelos et al. (2023) — Explanations Can Reduce Overreliance on AI Systems During Decision-Making***
- ***Buccinca et al. (2021) — Trust or Think: Cognitive Forcing Functions***

---

## Author

### Vasudev Pant

##### ***M.S. Business Analytics Candidate - W.P. Carey School of Business, Arizona State University***
##### ***Primary : vasudev21june@gmail.com***
##### ***Academic : vpant12@asu.edu***
##### ***Research focus: Human-Centered Trustworthy AI, Explainability and, Algorithmic Fairness, Trust Callibration***

---
