"""
Training pipeline: raw transactions -> engineered features -> trained model
-> cost-sensitive threshold -> saved artifacts (model + profiles + threshold).

Run: python src/train.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, classification_report
)
from xgboost import XGBClassifier

from preprocessing import build_customer_profiles, engineer_features, FEATURE_COLUMNS


def pick_cost_sensitive_threshold(y_true, y_proba, cost_false_negative=5000, cost_false_positive=50):
    """
    Instead of picking the threshold that maximizes F1 (a purely statistical
    choice), pick the one that minimizes real business cost:
    - missing a fraud transaction (false negative) costs ~cost_false_negative
    - wrongly flagging a genuine transaction (false positive) costs
      ~cost_false_positive (customer friction, manual review time)
    These numbers are stated assumptions, not measured facts — swap in real
    figures if this were a real company.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold, best_cost = 0.5, float("inf")

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        fn = np.sum((preds == 0) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        total_cost = fn * cost_false_negative + fp * cost_false_positive
        if total_cost < best_cost:
            best_cost, best_threshold = total_cost, t

    return best_threshold, best_cost


def main():
    print("Loading data...")
    df = pd.read_csv("data/transactions.csv")

    # Time-based split: train on first 24 days, test on last 6 — avoids
    # leaking future information backward, which a random split would do
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].quantile(0.8)
    train_df = df[df["timestamp"] < cutoff].copy()
    test_df = df[df["timestamp"] >= cutoff].copy()
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    print("Building customer profiles from training data...")
    profiles = build_customer_profiles(train_df)

    print("Engineering features...")
    train_feat = engineer_features(train_df, profiles)
    test_feat = engineer_features(test_df, profiles)

    X_train, y_train = train_feat[FEATURE_COLUMNS], train_feat["is_fraud"]
    X_test, y_test = test_feat[FEATURE_COLUMNS], test_feat["is_fraud"]

    print(f"Train fraud rate: {y_train.mean()*100:.2f}%  |  Test fraud rate: {y_test.mean()*100:.2f}%")

    # Imbalance handling: class weighting (simpler and more defensible than
    # SMOTE for this size of dataset — SMOTE-generated synthetic fraud on
    # top of already-synthetic fraud adds a layer that's hard to justify
    # in an interview; class weighting is transparent and standard)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"scale_pos_weight (class imbalance ratio): {scale_pos_weight:.1f}")

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"\nTest PR-AUC: {pr_auc:.4f}")

    threshold, cost = pick_cost_sensitive_threshold(y_test.values, y_proba)
    print(f"Cost-sensitive threshold: {threshold:.2f} (estimated cost at this threshold: {cost:.0f})")

    y_pred = (y_proba >= threshold).astype(int)
    print("\nClassification report at chosen threshold:")
    print(classification_report(y_test, y_pred, digits=3))

    # Save everything the API needs
    with open("models/fraud_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "profiles": profiles,
            "threshold": float(threshold),
            "feature_columns": FEATURE_COLUMNS,
            "pr_auc": float(pr_auc),
        }, f)
    print("\nSaved model artifact to models/fraud_model.pkl")


if __name__ == "__main__":
    main()
