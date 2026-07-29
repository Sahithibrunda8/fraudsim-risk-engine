"""
Drift check — simulates a "month 2" with a shifted fraud rate and pattern
mix, then checks whether feature-distribution monitoring would catch the
shift. This demonstrates the monitoring piece of the project without
needing a real production system running for months.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from data.simulator import TransactionSimulator
from preprocessing import build_customer_profiles, engineer_features, FEATURE_COLUMNS

# Month 1: same as training data
sim1 = TransactionSimulator(n_customers=500, seed=42)
month1 = sim1.generate_dataset(pd.Timestamp("2026-01-01"), n_days=30, target_fraud_rate=0.01)

# Month 2: same customers, but a higher fraud rate and different pattern mix,
# simulating a real-world scenario where fraud tactics evolve
sim2 = TransactionSimulator(n_customers=500, seed=99)  # different seed = shifted behavior
month2 = sim2.generate_dataset(pd.Timestamp("2026-02-01"), n_days=30, target_fraud_rate=0.025)

profiles = build_customer_profiles(month1)
feat1 = engineer_features(month1, profiles)[FEATURE_COLUMNS]
feat2 = engineer_features(month2, profiles)[FEATURE_COLUMNS]

report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=feat1, current_data=feat2)
result.save_html("monitoring/drift_report.html")

print("Month 1 fraud rate:", month1["is_fraud"].mean())
print("Month 2 fraud rate:", month2["is_fraud"].mean())
print("\nDrift report saved to monitoring/drift_report.html")
