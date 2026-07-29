"""
Transaction Simulator for Fraud Detection Project
---------------------------------------------------
Generates realistic normal customer transactions, then injects
labeled fraud patterns at a controlled rate.

Why build this instead of downloading a fraud dataset:
- You control the imbalance ratio (real fraud is ~0.1-2% of transactions)
- You control drift (can generate a "month 2" with shifted fraud patterns
  later, to test monitoring)
- You can explain every design decision in an interview, because you made
  every design decision
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import uuid

fake = Faker()


class Customer:
    """A synthetic customer with a stable, realistic spending profile."""

    def __init__(self, customer_id, rng):
        self.customer_id = customer_id
        self.rng = rng

        # Home location — fraud will be judged partly on distance from this
        self.home_lat = rng.uniform(8.0, 28.0)   # rough India latitude range
        self.home_lon = rng.uniform(72.0, 88.0)  # rough India longitude range

        # Spending behavior — lognormal so most transactions are small,
        # a few are large, which mirrors real spending distributions
        self.avg_amount = rng.uniform(300, 5000)     # typical transaction size (INR)
        self.amount_std = self.avg_amount * 0.4

        # Typical active hours (most people transact 8am-11pm-ish)
        self.active_hour_start = rng.integers(6, 10)
        self.active_hour_end = rng.integers(20, 23)

        # Typical transaction frequency
        self.avg_daily_transactions = rng.poisson(1.5) + 1

        self.categories = rng.choice(
            ["groceries", "food_delivery", "shopping", "utilities",
             "travel", "electronics", "entertainment"],
            size=rng.integers(2, 5), replace=False
        ).tolist()


class TransactionSimulator:
    """Generates a labeled dataset of normal + fraudulent transactions."""

    def __init__(self, n_customers=2000, seed=42):
        self.rng = np.random.default_rng(seed)
        self.customers = [Customer(f"CUST_{i:05d}", self.rng) for i in range(n_customers)]

    # ---------- normal transaction generation ----------

    def _random_timestamp(self, day, customer):
        hour = self.rng.integers(customer.active_hour_start, customer.active_hour_end)
        minute = self.rng.integers(0, 60)
        return day.replace(hour=int(hour), minute=int(minute), second=int(self.rng.integers(0, 60)))

    def _nearby_location(self, customer, max_km_drift=15):
        # small random drift from home location, simulating local spending
        lat_drift = self.rng.normal(0, max_km_drift / 111)  # ~111km per degree lat
        lon_drift = self.rng.normal(0, max_km_drift / 111)
        return customer.home_lat + lat_drift, customer.home_lon + lon_drift

    def generate_normal_transactions(self, start_date, n_days):
        rows = []
        for customer in self.customers:
            for d in range(n_days):
                day = start_date + timedelta(days=d)
                n_tx = self.rng.poisson(customer.avg_daily_transactions)
                for _ in range(n_tx):
                    amount = max(50, self.rng.normal(customer.avg_amount, customer.amount_std))
                    lat, lon = self._nearby_location(customer)
                    rows.append({
                        "transaction_id": str(uuid.uuid4()),
                        "customer_id": customer.customer_id,
                        "timestamp": self._random_timestamp(day, customer),
                        "amount": round(amount, 2),
                        "merchant_category": self.rng.choice(customer.categories),
                        "lat": round(lat, 4),
                        "lon": round(lon, 4),
                        "is_fraud": 0,
                        "fraud_type": "none",
                    })
        return pd.DataFrame(rows)

    # ---------- fraud injection patterns ----------

    def inject_card_testing(self, df, n_incidents):
        """Fraud pattern: several small-value transactions in quick succession
        across different merchants — attacker testing if a stolen card works."""
        rows = []
        victims = self.rng.choice(self.customers, size=n_incidents, replace=False)
        base_dates = pd.to_datetime(df["timestamp"]).dt.date.unique()
        for customer in victims:
            day = pd.Timestamp(self.rng.choice(base_dates))
            start_time = day.replace(
                hour=int(self.rng.integers(0, 24)), minute=int(self.rng.integers(0, 60))
            )
            n_tx = self.rng.integers(4, 9)  # burst of small transactions
            for i in range(n_tx):
                ts = start_time + timedelta(minutes=int(self.rng.integers(1, 4)) * i)
                lat, lon = self._nearby_location(customer, max_km_drift=5)
                rows.append({
                    "transaction_id": str(uuid.uuid4()),
                    "customer_id": customer.customer_id,
                    "timestamp": ts,
                    "amount": round(self.rng.uniform(10, 100), 2),  # unusually small
                    "merchant_category": "misc",
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "is_fraud": 1,
                    "fraud_type": "card_testing",
                })
        return pd.DataFrame(rows)

    def inject_account_takeover(self, df, n_incidents):
        """Fraud pattern: sudden burst of high-value transactions, well above
        the customer's normal spending, at an unusual hour."""
        rows = []
        victims = self.rng.choice(self.customers, size=n_incidents, replace=False)
        base_dates = pd.to_datetime(df["timestamp"]).dt.date.unique()
        for customer in victims:
            day = pd.Timestamp(self.rng.choice(base_dates))
            unusual_hour = (customer.active_hour_start - 5) % 24  # off their normal hours
            ts = day.replace(hour=int(unusual_hour), minute=int(self.rng.integers(0, 60)))
            n_tx = self.rng.integers(1, 3)
            for i in range(n_tx):
                lat, lon = self._nearby_location(customer, max_km_drift=5)
                rows.append({
                    "transaction_id": str(uuid.uuid4()),
                    "customer_id": customer.customer_id,
                    "timestamp": ts + timedelta(minutes=i * 2),
                    "amount": round(customer.avg_amount * self.rng.uniform(5, 15), 2),  # spike
                    "merchant_category": "electronics",
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "is_fraud": 1,
                    "fraud_type": "account_takeover",
                })
        return pd.DataFrame(rows)

    def inject_geo_impossible(self, df, n_incidents):
        """Fraud pattern: two transactions from the same customer, far apart
        geographically, too close in time to be physically possible (velocity fraud)."""
        rows = []
        victims = self.rng.choice(self.customers, size=n_incidents, replace=False)
        base_dates = pd.to_datetime(df["timestamp"]).dt.date.unique()
        for customer in victims:
            day = pd.Timestamp(self.rng.choice(base_dates))
            ts1 = day.replace(hour=int(self.rng.integers(8, 20)), minute=int(self.rng.integers(0, 60)))
            ts2 = ts1 + timedelta(minutes=int(self.rng.integers(5, 30)))  # too soon for real travel

            lat1, lon1 = self._nearby_location(customer, max_km_drift=5)
            # second transaction far away — random distant point
            lat2 = customer.home_lat + self.rng.choice([-1, 1]) * self.rng.uniform(5, 15)
            lon2 = customer.home_lon + self.rng.choice([-1, 1]) * self.rng.uniform(5, 15)

            for ts, lat, lon in [(ts1, lat1, lon1), (ts2, lat2, lon2)]:
                rows.append({
                    "transaction_id": str(uuid.uuid4()),
                    "customer_id": customer.customer_id,
                    "timestamp": ts,
                    "amount": round(self.rng.normal(customer.avg_amount, customer.amount_std), 2),
                    "merchant_category": self.rng.choice(customer.categories),
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "is_fraud": 1,
                    "fraud_type": "geo_impossible",
                })
        return pd.DataFrame(rows)

    # ---------- orchestration ----------

    def generate_dataset(self, start_date, n_days, target_fraud_rate=0.01):
        normal_df = self.generate_normal_transactions(start_date, n_days)
        n_normal = len(normal_df)

        # solve for incident counts so total fraud rows land near target_fraud_rate
        # card_testing produces ~6 rows/incident, account_takeover ~2, geo_impossible exactly 2
        # -> weighted so total fraud ROWS (not incidents) hits the target rate
        n_fraud_target = int(n_normal * target_fraud_rate / (1 - target_fraud_rate))
        n_each = max(1, n_fraud_target // 10)

        card_testing_df = self.inject_card_testing(normal_df, n_each)
        takeover_df = self.inject_account_takeover(normal_df, n_each)
        geo_df = self.inject_geo_impossible(normal_df, n_each)

        full_df = pd.concat(
            [normal_df, card_testing_df, takeover_df, geo_df], ignore_index=True
        )
        full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
        return full_df


if __name__ == "__main__":
    sim = TransactionSimulator(n_customers=2000, seed=42)
    df = sim.generate_dataset(start_date=pd.Timestamp("2026-01-01"), n_days=30, target_fraud_rate=0.01)

    print(f"Total transactions: {len(df)}")
    print(f"Fraud transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(df["fraud_type"].value_counts())

    df.to_csv("transactions.csv", index=False)
    print("\nSaved to transactions.csv")
