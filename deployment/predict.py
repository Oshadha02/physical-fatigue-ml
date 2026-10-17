
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# Locate the saved model and metadata files
MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_PATH = MODEL_DIR / "fatigue_xgboost.json"
METADATA_PATH = MODEL_DIR / "fatigue_model_metadata.json"


# Load model and metadata
model = XGBRegressor()
model.load_model(str(MODEL_PATH))

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)


# Required raw sensor columns
REQUIRED_COLUMNS = [
    "HR_Processed",
    "HRV",
    "Temperature",
    "IMU_5_ax_g_",
    "IMU_5_ay_g_",
    "IMU_5_az_g_",
    "IMU_5_gx_dps_",
]


def extract_features(sensor_data):
    # Verify the required sensor columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in sensor_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Each prediction requires one 3-minute window
    if len(sensor_data) != 360:
        raise ValueError(
            f"Expected 360 readings, received {len(sensor_data)}."
        )

    data = sensor_data[REQUIRED_COLUMNS].copy()

    # Convert readings to numeric values
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Apply training preprocessing rules
    data.loc[
        data["HR_Processed"] <= 0,
        "HR_Processed"
    ] = np.nan

    data.loc[
        data["HRV"] < 0,
        "HRV"
    ] = np.nan

    # Calculate acceleration magnitude
    data["Acceleration_Magnitude"] = np.sqrt(
        data["IMU_5_ax_g_"] ** 2
        + data["IMU_5_ay_g_"] ** 2
        + data["IMU_5_az_g_"] ** 2
    )

    # Provisional gyroscope signal
    data["Gyroscope_Proxy"] = data[
        "IMU_5_gx_dps_"
    ].abs()

    sensors = {
        "HR": "HR_Processed",
        "HRV": "HRV",
        "Temperature": "Temperature",
        "Acceleration": "Acceleration_Magnitude",
        "Gyroscope": "Gyroscope_Proxy",
    }

    # Calculate the 25 features used during training
    features = {}

    for name, column in sensors.items():
        values = data[column]

        features[f"{name}_Mean"] = values.mean()
        features[f"{name}_Median"] = values.median()
        features[f"{name}_STD"] = values.std()
        features[f"{name}_P10"] = values.quantile(0.10)
        features[f"{name}_P90"] = values.quantile(0.90)

    # Preserve the exact feature order used during training
    feature_df = pd.DataFrame([features])
    feature_df = feature_df[metadata["feature_columns"]]

    return feature_df


def predict_fatigue(sensor_data):
    # Extract features from the uploaded readings
    features = extract_features(sensor_data)

    # Predict a fatigue score between 0 and 10
    score = float(
        np.clip(model.predict(features)[0], 0, 10)
    )

    # Convert the score into a fatigue category
    if score < 4:
        level = "Low"
    elif score < 7:
        level = "Moderate"
    else:
        level = "High"

    return {
        "fatigue_score": round(score, 2),
        "fatigue_level": level,
        "features": features,
    }
