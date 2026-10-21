
from functools import lru_cache

import numpy as np
import pandas as pd
import shap

# Use the XGBoost model
from predict import model, SENSOR_COLUMNS


@lru_cache(maxsize=1)
def load_shap_explainer():
    # Create a SHAP explainer for the XGBoost model
    return shap.TreeExplainer(model)


def explain_prediction(features, worker_id="Worker"):
    # Validate the flattened input
    if not isinstance(features, pd.DataFrame):
        raise ValueError("Features must be a DataFrame.")

    if features.shape != (1, 1800):
        raise ValueError(
            "Expected one sample containing 1,800 values."
        )

    # Convert the features into an array
    feature_array = features.to_numpy(dtype=np.float32)

    # Calculate SHAP values
    explainer = load_shap_explainer()
    shap_result = explainer(feature_array)
    shap_values = shap_result.values[0]

    # Five sensor groups
    sensor_groups = [
        {
            "name": "Heart Rate",
            "column": "HR_Processed",
            "unit": "BPM",
            "description": "heart rate",
        },
        {
            "name": "Heart Rate Variability",
            "column": "HRV",
            "unit": "",
            "description": "heart rate variability",
        },
        {
            "name": "Skin Temperature",
            "column": "Temperature",
            "unit": "°C",
            "description": "skin temperature",
        },
        {
            "name": "Body Movement",
            "column": "Acceleration_Magnitude",
            "unit": "g",
            "description": "acceleration magnitude",
        },
        {
            "name": "Gyroscope Movement",
            "column": "Gyroscope_Proxy",
            "unit": "°/s",
            "description": "gyroscope movement",
        },
    ]

    explanations = []

    for sensor in sensor_groups:
        # Find this sensor's position among the five signals
        position = SENSOR_COLUMNS.index(sensor["column"])

        # Get its 360 readings from the flattened input
        indices = np.arange(position, 1800, 5)
        readings = feature_array[0, indices]

        # Combine SHAP values for this sensor
        contribution = float(np.sum(shap_values[indices]))

        # Calculate average and median sensor readings
        mean_value = float(np.mean(readings))
        median_value = float(np.median(readings))

        unit = sensor["unit"]
        description = sensor["description"]

        reading_text = (
            f"{worker_id} had an average {description} "
            f"of {mean_value:.2f} {unit} "
            f"and a median of {median_value:.2f} {unit}."
        )

        # Explain how this sensor influenced the prediction
        if contribution > 0.005:
            impact_text = (
                "These readings pushed the model's "
                "fatigue estimate higher."
            )

        elif contribution < -0.005:
            impact_text = (
                "These readings pushed the model's "
                "fatigue estimate lower."
            )

        else:
            impact_text = (
                "These readings had almost no net effect "
                "on the model's fatigue estimate."
            )

        explanations.append({
            "sensor": sensor["name"],
            "text": f"{reading_text} {impact_text}",
            "contribution": contribution,
            "importance": abs(contribution),
        })

    # Show the strongest net contributions first
    explanations.sort(
        key=lambda item: item["importance"],
        reverse=True
    )

    return explanations
