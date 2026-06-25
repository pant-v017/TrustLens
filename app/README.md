# TrustLens — Study Application

The interactive web application for the TrustLens study on trust calibration
in AI-assisted loan decision-making.

## What this app does

Each participant is randomly assigned to one of three explanation conditions and
reviews six loan cases. For every case the app records their trust (before and
after the explanation), comprehension, fairness perception, and their accept/override
decision.

- **Condition A (Minimal):** AI decision and confidence only
- **Condition B (Text):** decision plus a plain-language explanation
- **Condition C (Visual):** decision plus a SHAP waterfall chart

## Design

- **Between-subjects:** each participant experiences one condition only
- **Trust-shift measure:** trust is rated before and after the explanation (B and C)
- **Per-case survey:** trust, comprehension, fairness, and action captured per case

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
trustlens_app/
├── app.py                      # main application
├── requirements.txt
├── model/
│   ├── xgboost_model.pkl       # trained classifier
│   ├── decision_threshold.pkl  # optimised threshold
│   └── feature_names.pkl
├── data/
│   └── study_cases.json        # the six fixed study cases
└── responses/                  # per-participant CSV output (created at runtime)
```

## Data collected per case

| Field | Meaning |
|---|---|
| initial_trust | trust before the explanation (1–5) |
| final_trust | trust after the explanation (1–5) |
| trust_shift | final minus initial |
| understand | comprehension rating (1–5) |
| fairness | fairness perception (1–5) |
| action | accept or override |
| open_response | final free-text reflection |

Part of a PhD application portfolio in Human-Centered Computing / HCI.
