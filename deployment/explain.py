
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor


# Locate the saved XGBoost model
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "fatigue_xgboost.json"
)


@lru_cache(maxsize=1)
def load_shap_explainer():
    # Load the same model used by the prediction backend
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))

    return shap.TreeExplainer(model)


def explain_prediction(features, worker_id="Worker"):
    # Validate the extracted feature data
    if not isinstance(features, pd.DataFrame) or len(features) != 1:
        raise ValueError(
            "Exactly one row of extracted features is required."
        )

    # Calculate SHAP values for the prediction
    explainer = load_shap_explainer()
    shap_result = explainer(features)

    shap_values = shap_result.values[0]

    # Group the 25 features into five sensor categories
    sensor_groups = [
        {
            "name": "Heart Rate",
            "prefix": "HR",
            "unit": "BPM",
            "description": "heart rate",
        },
        {
            "name": "Heart Rate Variability",
            "prefix": "HRV",
            "unit": "",
            "description": "heart rate variability",
        },
        {
            "name": "Skin Temperature",
            "prefix": "Temperature",
            "unit": "°C",
            "description": "skin temperature",
        },
        {
            "name": "Body Movement",
            "prefix": "Acceleration",
            "unit": "g",
            "description": "acceleration magnitude",
        },
        {
            "name": "Gyroscope Movement",
            "prefix": "Gyroscope",
            "unit": "°/s",
            "description": "absolute gyroscope X-axis reading",
        },
    ]

    explanations = []

    for sensor in sensor_groups:
        prefix = sensor["prefix"]

        # Find the five features belonging to this sensor
        indices = [
            i for i, column in enumerate(features.columns)
            if column.startswith(prefix + "_")
        ]

        # Calculate combined SHAP contribution
        contribution = float(np.sum(shap_values[indices]))

        # Read the original statistical feature values
        mean_value = features.iloc[0][f"{prefix}_Mean"]
        median_value = features.iloc[0][f"{prefix}_Median"]

        unit = sensor["unit"]
        description = sensor["description"]

        # Describe the sensor readings
        if pd.isna(mean_value):
            reading_text = (
                f"{worker_id} had no valid {description} "
                "readings in this 3-minute window."
            )
        else:
            reading_text = (
                f"{worker_id} had an average {description} "
                f"of {mean_value:.2f} {unit} "
                f"and a median of {median_value:.2f} {unit}."
            )

        # Describe the model's SHAP contribution
        if contribution > 0.005:
            impact_text = (
                f"These features increased the model's "
                f"predicted fatigue score by "
                f"{contribution:.2f} points."
            )

        elif contribution < -0.005:
            impact_text = (
                f"These features decreased the model's "
                f"predicted fatigue score by "
                f"{abs(contribution):.2f} points."
            )

        else:
            impact_text = (
                "These features had almost no net effect "
                "on the model's predicted fatigue score."
            )

        explanations.append({
            "sensor": sensor["name"],
            "text": f"{reading_text} {impact_text}",
            "contribution": contribution,
            "importance": abs(contribution),
        })

    # Show the strongest net sensor contributions first
    explanations.sort(
        key=lambda item: item["importance"],
        reverse=True
    )

    return explanations
