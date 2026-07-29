# Fraud Detection System — Synthetic Transaction Simulation & Real-Time Risk Scoring

## Problem Statement

Card and account fraud typically makes up less than 1-2% of all transactions, which
makes it a genuinely hard classification problem — accuracy is meaningless when 99%
of a "predict no fraud" model is already correct by default. This project builds a
fraud detection system from the ground up: a synthetic transaction generator (so the
imbalance ratio and fraud patterns are fully controlled and explainable), a
cost-sensitive classifier, an explainability layer, and a real-time scoring API with
drift monitoring.

The goal isn't just "detect fraud" — it's to catch fraud while minimizing the cost of
false alarms that annoy genuine customers, and to be able to explain *why* any given
transaction was flagged.

## Why Self-Built Data Instead of a Public Dataset

Most fraud detection portfolios reuse the same one or two public Kaggle datasets
(IEEE-CIS, PaySim). Instead, this project uses a custom simulator
(`data/simulator.py`) that generates:
- 2,000 synthetic customers, each with a stable spending profile (average
  transaction size, active hours, home location, preferred merchant categories)
- A month of normal transactions per customer following their own pattern
- Three labeled fraud patterns injected at a realistic ~1% rate:
  - **Card testing** — a burst of small transactions across different merchants in
    a short window (attacker verifying a stolen card works)
  - **Account takeover** — a sudden high-value transaction spike at an hour outside
    the customer's normal activity
  - **Geo-impossible** — two transactions from the same customer too far apart
    geographically to be physically possible in the time between them

Building the generator meant every design decision — the imbalance ratio, the fraud
patterns, the noise in "normal" behavior — is something I chose and can explain,
rather than inheriting from someone else's dataset.

**Verified output of the simulator (30 days, 2,000 customers, seed=42):**
- Total transactions: 151,402
- Fraud transactions: 1,475 (0.97%)
- Breakdown: card_testing — 946, geo_impossible — 302, account_takeover — 227

## Architecture

```
simulator.py → transactions.csv → EDA → feature engineering → model training
                                                                      │
                                                                      ▼
                                                          fraud_model.pkl
                                                                      │
                                                                      ▼
                                            FastAPI (/predict) ──► risk score + SHAP explanation
                                                                      │
                                                                      ▼
                                         Evidently drift check (month 1 vs. simulated month 2)
```

## Key Results

*(All numbers below are real output from running this pipeline, not estimates.)*

- **Model & metric:** XGBoost classifier — **Precision-Recall AUC: 0.867** on a
  time-based holdout (last 20% of days, never seen during training). Accuracy is
  not used as the primary metric since it's uninformative at ~1% fraud prevalence
  (a model that predicts "never fraud" would already be 99% accurate).
- **Imbalance handling approach:** Class weighting via XGBoost's `scale_pos_weight`
  (~102x), rather than SMOTE — simpler and more defensible than generating
  synthetic oversampled fraud on top of already-synthetic data.
- **Cost-sensitive threshold:** 0.82, chosen by minimizing an assumed cost of
  ₹5,000 per missed fraud vs. ₹50 per false alarm (a stated assumption, not a
  measured business figure — swap in real numbers in a real deployment). At this
  threshold: **88.5% recall, 66.1% precision** on fraud cases.
- **Explainability:** Each flagged transaction returns its top 3 contributing
  features via SHAP, in plain language (e.g. "distance from customer's usual
  location," "transaction at an unusual hour"). Verified against real fraud
  examples — geo-impossible fraud correctly surfaces distance as the top reason,
  card testing correctly surfaces amount and unfamiliar category.
- **Drift check:** Simulated a "month 2" with a shifted fraud rate (0.93% → 2.36%)
  and pattern mix. Monitoring correctly flagged drift on 3 of 6 features (50%),
  with the strongest shifts in `amount_vs_avg` and `distance_from_home_km` — see
  `monitoring/drift_report.html`.

## How to Run It Locally

```bash
# 1. Clone and set up environment
git clone <your-repo-url>
cd fraud-detection-system
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Generate the dataset
python data/simulator.py

# 3. Train the model
python src/train.py

# 4. Run the API locally
uvicorn api.main:app --reload
# Visit http://localhost:8000/docs for the interactive Swagger UI
```

## Live API

- **Endpoint:** [TBD — filled in after step 12 deployment]
- **Example request:**
```json
POST /predict
{
  "customer_id": "CUST_00042",
  "amount": 4500,
  "merchant_category": "electronics",
  "hour": 3,
  "lat": 19.07,
  "lon": 72.87
}
```
- **Example response (verified real output for a genuinely suspicious transaction):**
```json
{
  "risk_score": 0.9998,
  "flagged": true,
  "threshold_used": 0.82,
  "top_reasons": ["time of day", "distance from customer's usual location", "transaction amount"]
}
```

## What I'd Do With More Time

- Add a fourth fraud pattern (merchant-side collusion / repeated refund abuse)
- Extend the simulator to model seasonal spending shifts, not just daily patterns
- Add a proper monitoring dashboard (Streamlit or Grafana) instead of a static
  Evidently HTML report
- Retrain automatically when drift crosses a threshold, rather than a manual check

## Project Structure

```
fraud-detection-system/
├── data/               # simulator + generated datasets
├── notebooks/          # EDA, modeling, thresholding, explainability, drift
├── src/                # reusable preprocessing, training, explanation code
├── models/             # trained model artifact
├── api/                # FastAPI app
├── monitoring/         # drift reports
├── tests/              # API sanity tests
└── Dockerfile
```

## Tech Stack

Python, Pandas, NumPy, Faker, Scikit-learn, XGBoost, imbalanced-learn, SHAP,
Evidently, FastAPI, Docker, GitHub Actions (optional CI/CD)
