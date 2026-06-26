# TrustLens

**Trust Calibration, Decision Agency, and Fairness Perception Across Explanation Modalities in AI-Assisted Loan Decision-Making**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link-here)

---

## Research Questions

**RQ1 — Trust Calibration**
How do different AI explanation modalities (minimal, textual, and visual) influence the accuracy of user trust calibration — defined as the alignment between expressed confidence and actual AI correctness — in high-stakes loan decision-making?

**RQ2 — Decision Agency**
Does the type of AI explanation modality influence user override behavior, independent of stated trust levels, revealing a gap between what users say they trust and what they actually act on?

**RQ3 — Fairness Perception**
How does exposure to feature-level SHAP explanations — which may surface demographic proxies — affect users' perceived algorithmic fairness, and does fairness perception moderate trust calibration?

---

## The Three Explanation Conditions

| Condition | What the user sees |
|---|---|
| A — Minimal | Approve / Reject + confidence score only |
| B — Text | Natural language explanation of the decision |
| C — Visual | SHAP-based feature contribution chart |

---

## Key Novel Contributions

1. **Trust Calibration Score** — measures whether user confidence matches AI accuracy on a per-case basis, not just whether users trust AI more or less
2. **Override Behavior** — captures what users *do*, not just what they *say*, revealing the explanation-action gap
3. **Fairness Perception under Explanation Exposure** — examines whether SHAP feature visibility affects perceived algorithmic fairness and moderates trust

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

- **Participants** — 15 to 25
- **Task** — Review 6 loan applicant profiles and evaluate AI recommendations
- **Measures** — 7-item Likert survey + override decision + open-ended response
- **Analysis** — Kruskal-Wallis, Mann-Whitney pairwise, chi-square for override rates

---

## Related Publications (Co-Author)

- *"Transparency and Accountability in AI Decision-Making: Methods and Challenges"*
- *"Performance Impact of Machine Unlearning in AI Systems"*

Both accepted in: **Machine Unlearning: Ethical and Responsible AI**
Wiley-Scrivener | Scopus-indexed | forthcoming

---

## Theoretical Grounding

- Schemmer et al. (2023) — Appropriate Reliance on AI
- Vasconcelos et al. (2023) — Explanations and Appropriate Reliance
- Buccinca et al. (2021) — Trust or Think: Cognitive Forcing Functions

---

## Author

### Vasudev Pant

#### **MS Business Analytics**
#### **W.P. Carey School of Business**
#### **Arizona State University**
#### **Mail : vasudev21june@gmail.com**

**Research focus: Human-Centered Trustworthy AI, Explainability, Algorithmic Fairness**

---
