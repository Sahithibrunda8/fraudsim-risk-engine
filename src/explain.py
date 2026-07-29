"""
Explainability — given a scored transaction, return the top contributing
features in plain language. Used by the API so every flagged transaction
comes with a reason, not just a number.
"""

import shap
import numpy as np

FEATURE_LABELS = {
    "amount": "transaction amount",
    "amount_vs_avg": "amount compared to this customer's usual spend",
    "distance_from_home_km": "distance from customer's usual location",
    "hour": "time of day",
    "is_unusual_hour": "transaction at an unusual hour for this customer",
    "is_new_category": "unfamiliar merchant category for this customer",
}


class FraudExplainer:
    def __init__(self, model, feature_columns, background_data):
        """
        background_data: a sample of training features, used as the
        reference point SHAP compares each prediction against.
        """
        self.feature_columns = feature_columns
        self.explainer = shap.TreeExplainer(model)
        self._background = background_data

    def explain(self, feature_row, top_n=3):
        """
        feature_row: a single-row DataFrame with the model's feature columns.
        Returns the top_n features pushing the prediction toward "fraud",
        in plain language.
        """
        shap_values = self.explainer.shap_values(feature_row)
        if isinstance(shap_values, list):  # some SHAP versions return a list per class
            shap_values = shap_values[-1]

        values = shap_values[0] if shap_values.ndim > 1 else shap_values
        contributions = list(zip(self.feature_columns, values))
        # sort by how much each feature pushed the score toward fraud (positive = toward fraud)
        contributions.sort(key=lambda x: x[1], reverse=True)

        reasons = []
        for feature, value in contributions[:top_n]:
            if value > 0:  # only report features that pushed toward "fraud"
                reasons.append(FEATURE_LABELS.get(feature, feature))
        return reasons if reasons else ["no single feature stood out — flagged on combined pattern"]
