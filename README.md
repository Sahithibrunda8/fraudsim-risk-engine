# Fraud Detection System (with a self-built transaction simulator)

## Why I built this

Most fraud detection projects online use the same one or two Kaggle datasets (IEEE-CIS or PaySim). I wanted something I could fully explain in an interview — where I understand not just the model, but where every row of data came from and why.

So instead of downloading a dataset, I built my own transaction simulator. It generates synthetic customers, each with their own spending habits, and then injects three realistic fraud patterns into that data at a controlled rate (~1%, close to real-world fraud prevalence). This meant I got to make every decision myself: how imbalanced the data should be, what fraud actually looks like in the features, and later, how to simulate concept drift for the monitoring piece.

## The problem

Fraud is a genuinely hard classification problem because it's so rare — under 1-2% of transactions in most real systems. A model that just predicts "not fraud" every time would already be ~99% accurate and completely useless. So the real work here wasn't just training a classifier, it was:
- picking the right metric (precision-recall, not accuracy)
- deciding what threshold actually makes business sense (catching more fraud costs you in false alarms — so where's the right tradeoff?)
- being able to explain *why* a transaction got flagged, not just that it did

## The simulator

`data/simulator.py` creates 2,000 synthetic customers, each with:
- a home location
- an average spend and typical spending range
- active hours (most people don't shop at 3am)
- a few preferred merchant categories

Then it generates a month of normal transactions per customer, and injects three fraud patterns:
- **Card testing** — a burst of small transactions across different merchants in a short window (this is what it looks like when someone's testing whether a stolen card number still works)
- **Account takeover** — a sudden high-value transaction at an hour way outside the customer's normal pattern
- **Geo-impossible transactions** — two transactions from the same customer, too far apart geographically to be physically possible given the time between them

Running it gives:
- 151,402 transactions total
- 1,475 fraud transactions (0.97%)
- broken down: 946 card testing, 302 geo-impossible, 227 account takeover

## What I did with the data

1. **Feature engineering** — built a profile for each customer from their transaction history (average spend, home location, typical hours, common categories), then engineered features like how far a transaction is from home, how the amount compares to the customer's usual spend, and whether the merchant category is new for them. This same feature code is shared between training and the live API, so there's no mismatch between how the model was trained and how it scores real transactions.

2. **Handling the imbalance** — I went with class weighting (XGBoost's `scale_pos_weight`, around 102x) rather than SMOTE. I thought about this one — SMOTE would mean generating synthetic oversampled fraud on top of data that's already synthetic, which felt like a layer of made-up data on made-up data. Class weighting is simpler and I can actually defend it.

3. **Model** — XGBoost, evaluated on precision-recall AUC (0.867 on a time-based holdout — trained on the first 24 days, tested on the last 6, so nothing "future" leaks into training).

4. **Threshold — not the default 0.5** — I picked the threshold that minimizes actual cost, not the one that maximizes F1. I assumed a cost of ₹5,000 for a missed fraud case versus ₹50 for annoying a real customer with a false alarm (these are stated assumptions, not real company numbers — in an actual job I'd use real figures). That landed the threshold at 0.82, catching 88.5% of fraud with 66.1% precision.

5. **Explainability** — every flagged transaction comes back with the top 3 reasons it was flagged, using SHAP, but translated into plain English instead of a SHAP plot. I tested this against real fraud examples and it lines up with what you'd expect — geo-impossible fraud gets flagged mainly for distance from the customer's usual location, card testing gets flagged for amount and unfamiliar merchant category.

6. **Drift check** — since I control the simulator, I generated a second "month 2" dataset with a different fraud rate (2.36% instead of 0.93%) and pattern mix, and ran it through Evidently to see if monitoring would actually catch the shift. It flagged drift on 3 of 6 features, with the biggest shifts in transaction amount relative to customer average and distance from home — which makes sense given what changed.

7. **Deployment** — wrapped the whole thing in a FastAPI app that takes a transaction and returns a risk score, a flag, and the explanation. Containerized it with Docker and tested it running both locally and inside the container — same results both ways.

## Results, for real

- PR-AUC: 0.867
- At the chosen threshold (0.82): 88.5% recall, 66.1% precision on fraud cases
- Verified example — a transaction at 3am, far from the customer's home, for a large amount, scored 0.9998 risk and got flagged for "time of day" and "distance from customer's usual location"
- A transaction matching the customer's actual spending profile scored 0.05 risk and wasn't flagged

## How to run it

```bash
git clone https://github.com/Sahithibrunda8/fraudsim-risk-engine.git
cd fraudsim-risk-engine
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python data/simulator.py      # generates the dataset
python src/train.py           # trains the model
python notebooks/01_eda.py    # regenerates EDA plots
python notebooks/05_drift_check.py   # regenerates the drift report

uvicorn api.main:app --reload
# visit http://localhost:8000/docs
```

Or with Docker:
```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

## Live API

Deployed on Render (free tier — the first request after a period of inactivity can take 30-60 seconds while the instance spins back up, everything after that is fast):

**https://fraudsim-risk-engine.onrender.com/docs**

Verified live response for the example above:
```json
{
  "risk_score": 0.9998,
  "flagged": true,
  "threshold_used": 0.82,
  "top_reasons": ["time of day", "distance from customer's usual location", "transaction amount"]
}
```

## Example request/response

```json
POST /predict
{
  "customer_id": "CUST_00042",
  "amount": 90000,
  "merchant_category": "electronics",
  "timestamp": "2026-01-05T03:00:00",
  "lat": 28.6,
  "lon": 77.2
}
```

```json
{
  "risk_score": 0.9998,
  "flagged": true,
  "threshold_used": 0.82,
  "top_reasons": ["time of day", "distance from customer's usual location", "transaction amount"]
}
```

## What I'd add with more time

- A fourth fraud pattern — repeated refund abuse or merchant-side collusion
- Seasonal spending shifts in the simulator, not just daily patterns
- A real dashboard instead of a static HTML drift report
- Automatic retraining triggered when drift crosses a threshold, instead of a manual check

## Project structure

```
fraudsim-risk-engine/
├── data/               # simulator + generated dataset
├── notebooks/          # EDA and drift check scripts
├── src/                # shared preprocessing, training, explainability
├── models/             # trained model artifact
├── api/                # FastAPI app
├── monitoring/         # drift report
├── tests/               # API tests
└── Dockerfile
```

## Stack

Python, Pandas, NumPy, Faker, Scikit-learn, XGBoost, SHAP, Evidently, FastAPI, Docker