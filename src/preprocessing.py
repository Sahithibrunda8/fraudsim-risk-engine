"""
Shared preprocessing — used by BOTH training and the live API.

Why this file exists as its own module instead of feature code living
inside a notebook: if training and the API compute features differently,
you get "train/serve skew" — a classic real-world bug where a model looks
great in testing and performs badly in production because the live
features don't match what it was trained on. Importing the same functions
in both places prevents that.
"""

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_customer_profiles(df):
    """
    Build a per-customer profile from historical (training) data:
    average spend, home location, typical active hours, common categories.

    This profile is what the API looks up at inference time — a single
    incoming transaction is judged against this profile.
    """
    profiles = {}
    for customer_id, group in df.groupby("customer_id"):
        profiles[customer_id] = {
            "avg_amount": group["amount"].mean(),
            "std_amount": group["amount"].std() if len(group) > 1 else group["amount"].mean() * 0.3,
            "home_lat": group["lat"].median(),
            "home_lon": group["lon"].median(),
            "typical_hour_start": pd.to_datetime(group["timestamp"]).dt.hour.quantile(0.1),
            "typical_hour_end": pd.to_datetime(group["timestamp"]).dt.hour.quantile(0.9),
            "common_categories": set(group["merchant_category"].value_counts().head(3).index),
        }
    return profiles


def engineer_features(df, profiles, default_profile=None):
    """
    Turn raw transaction rows into model-ready features, using each
    customer's profile. Falls back to a global default profile for
    customers not seen before (relevant for the API, not for training).
    """
    if default_profile is None:
        all_amounts = [p["avg_amount"] for p in profiles.values()]
        default_profile = {
            "avg_amount": np.mean(all_amounts),
            "std_amount": np.std(all_amounts),
            "home_lat": np.mean([p["home_lat"] for p in profiles.values()]),
            "home_lon": np.mean([p["home_lon"] for p in profiles.values()]),
            "typical_hour_start": 8,
            "typical_hour_end": 21,
            "common_categories": set(),
        }

    rows = []
    timestamps = pd.to_datetime(df["timestamp"])

    for i, row in df.reset_index(drop=True).iterrows():
        profile = profiles.get(row["customer_id"], default_profile)
        hour = timestamps.iloc[i].hour

        distance_km = haversine_km(
            row["lat"], row["lon"], profile["home_lat"], profile["home_lon"]
        )
        amount_vs_avg = row["amount"] / max(profile["avg_amount"], 1)
        is_unusual_hour = int(
            hour < profile["typical_hour_start"] or hour > profile["typical_hour_end"]
        )
        is_new_category = int(row["merchant_category"] not in profile["common_categories"])

        rows.append({
            "amount": row["amount"],
            "amount_vs_avg": amount_vs_avg,
            "distance_from_home_km": distance_km,
            "hour": hour,
            "is_unusual_hour": is_unusual_hour,
            "is_new_category": is_new_category,
        })

    features_df = pd.DataFrame(rows)
    if "is_fraud" in df.columns:
        features_df["is_fraud"] = df["is_fraud"].values
    return features_df


FEATURE_COLUMNS = [
    "amount", "amount_vs_avg", "distance_from_home_km",
    "hour", "is_unusual_hour", "is_new_category",
]
