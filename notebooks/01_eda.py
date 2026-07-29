"""
EDA — run this first to understand the data before any modeling.
Paste these cells into notebooks/01_eda.ipynb, or run directly:
    python notebooks/01_eda.py
Saves plots to notebooks/eda_plots/ so you can drop them into your README.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("notebooks/eda_plots", exist_ok=True)
df = pd.read_csv("data/transactions.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour

print("Shape:", df.shape)
print("\nNulls:\n", df.isnull().sum())

# 1. Class balance
fraud_rate = df["is_fraud"].mean()
print(f"\nFraud rate: {fraud_rate*100:.2f}%")

# 2. Amount distribution, fraud vs non-fraud
fig, ax = plt.subplots(figsize=(8, 5))
df[df["is_fraud"] == 0]["amount"].clip(upper=5000).hist(bins=50, alpha=0.6, label="Legit", ax=ax)
df[df["is_fraud"] == 1]["amount"].clip(upper=5000).hist(bins=50, alpha=0.6, label="Fraud", ax=ax)
ax.set_xlabel("Amount (clipped at 5000)")
ax.set_ylabel("Count")
ax.set_title("Transaction amount: fraud vs legit")
ax.legend()
plt.savefig("notebooks/eda_plots/amount_distribution.png", dpi=100, bbox_inches="tight")
plt.close()

# 3. Hour of day, fraud vs non-fraud
fig, ax = plt.subplots(figsize=(8, 5))
df[df["is_fraud"] == 0]["hour"].value_counts(normalize=True).sort_index().plot(
    kind="bar", alpha=0.6, label="Legit", ax=ax, position=1, width=0.4
)
df[df["is_fraud"] == 1]["hour"].value_counts(normalize=True).sort_index().plot(
    kind="bar", alpha=0.6, label="Fraud", ax=ax, position=0, width=0.4, color="orange"
)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Proportion of transactions")
ax.set_title("Time of day: fraud vs legit")
ax.legend()
plt.savefig("notebooks/eda_plots/hour_distribution.png", dpi=100, bbox_inches="tight")
plt.close()

# 4. Fraud type breakdown
print("\nFraud type counts:\n", df[df["is_fraud"] == 1]["fraud_type"].value_counts())

# 5. Geographic spread
fig, ax = plt.subplots(figsize=(8, 6))
legit = df[df["is_fraud"] == 0].sample(min(5000, (df["is_fraud"] == 0).sum()), random_state=42)
fraud = df[df["is_fraud"] == 1]
ax.scatter(legit["lon"], legit["lat"], s=3, alpha=0.3, label="Legit")
ax.scatter(fraud["lon"], fraud["lat"], s=8, alpha=0.7, color="red", label="Fraud")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographic spread: fraud vs legit")
ax.legend()
plt.savefig("notebooks/eda_plots/geo_spread.png", dpi=100, bbox_inches="tight")
plt.close()

# 6. Transactions per customer sanity check
tx_per_customer = df.groupby("customer_id").size()
print(f"\nTransactions per customer: min={tx_per_customer.min()}, "
      f"median={tx_per_customer.median()}, max={tx_per_customer.max()}")

print("\nPlots saved to notebooks/eda_plots/")
