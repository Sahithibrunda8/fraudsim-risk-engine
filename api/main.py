"""
Real-time fraud scoring API.

Run locally:  uvicorn api.main:app --reload   (from the project root)
Docs:         http://localhost:8000/docs
"""

import pickle
import sys
import os
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import engineer_features, FEATURE_COLUMNS  # noqa: E402
from explain import FraudExplainer  # noqa: E402
from schemas import TransactionRequest, RiskResponse  # noqa: E402

app = FastAPI(title="Fraud Detection API")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_model.pkl")

# Load model + profiles once at startup, not per-request — reloading a
# pickle on every call would make the API far too slow for real-time use
with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
profiles = artifact["profiles"]
threshold = artifact["threshold"]
feature_columns = artifact["feature_columns"]

# background sample for SHAP, built once at startup
_sample_df = pd.DataFrame([
    {"customer_id": cid, "amount": p["avg_amount"], "merchant_category": "misc",
     "lat": p["home_lat"], "lon": p["home_lon"], "timestamp": "2026-01-01T12:00:00"}
    for cid, p in list(profiles.items())[:200]
])
_background_features = engineer_features(_sample_df, profiles)[feature_columns]
explainer = FraudExplainer(model, feature_columns, _background_features)


@app.get("/")
def health_check():
    return {"status": "ok", "model_pr_auc": artifact.get("pr_auc")}


@app.post("/predict", response_model=RiskResponse)
def predict(transaction: TransactionRequest):
    row = pd.DataFrame([{
        "customer_id": transaction.customer_id,
        "amount": transaction.amount,
        "merchant_category": transaction.merchant_category,
        "timestamp": transaction.timestamp,
        "lat": transaction.lat,
        "lon": transaction.lon,
    }])

    try:
        features = engineer_features(row, profiles)[feature_columns]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {e}")

    risk_score = float(model.predict_proba(features)[:, 1][0])
    flagged = risk_score >= threshold
    reasons = explainer.explain(features) if flagged else []

    return RiskResponse(
        risk_score=round(risk_score, 4),
        flagged=flagged,
        threshold_used=threshold,
        top_reasons=reasons,
    )
