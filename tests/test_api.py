"""
Basic sanity tests for the fraud API. Run with: pytest tests/test_api.py
(run from the project root, so the model artifact and imports resolve)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_expected_fields():
    response = client.post("/predict", json={
        "customer_id": "CUST_00042",
        "amount": 420,
        "merchant_category": "groceries",
        "timestamp": "2026-01-05T14:00:00",
        "lat": 9.80,
        "lon": 86.35,
    })
    assert response.status_code == 200
    body = response.json()
    assert "risk_score" in body
    assert "flagged" in body
    assert 0 <= body["risk_score"] <= 1


def test_predict_rejects_invalid_amount():
    response = client.post("/predict", json={
        "customer_id": "CUST_00042",
        "amount": -50,  # invalid — amount must be > 0
        "merchant_category": "groceries",
        "timestamp": "2026-01-05T14:00:00",
        "lat": 9.80,
        "lon": 86.35,
    })
    assert response.status_code == 422  # FastAPI validation error


def test_unknown_customer_does_not_crash():
    """A customer not seen during training should fall back to the
    default profile instead of raising an error."""
    response = client.post("/predict", json={
        "customer_id": "CUST_UNKNOWN_999",
        "amount": 1000,
        "merchant_category": "shopping",
        "timestamp": "2026-01-05T14:00:00",
        "lat": 20.0,
        "lon": 78.0,
    })
    assert response.status_code == 200
