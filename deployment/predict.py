
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# Load the XGBoost model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "xgboost_3min_flattened.json"

model = XGBRegressor()
model.load_model(str(MODEL_PATH))


# Required columns from the uploaded CSV
REQUIRED_COLUMNS = [
    "HR_Processed",
    "HRV",
    "Temperature",
    "IMU_5_ax_g_",
    "IMU_5_ay_g_",
    "IMU_5_az_g_",
    "IMU_5_gx_dps_",
]

# Five model input signals, in order
SENSOR_COLUMNS = [
    "HR_Processed",
    "HRV",
    "Temperature",
    "Acceleration_Magnitude",
    "Gyroscope_Proxy",
]

# Training-derived missing-value replacements
SENSOR_MEDIANS = {
    "HR_Processed": 104.30,
    "HRV": 59.40,
    "Temperature": 34.53,
    "Acceleration_Magnitude": 1.03,
    "Gyroscope_Proxy": 4.44,
}


def extract_features(sensor_data):
    # Check the sensor columns
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in sensor_data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # One sample = 360 readings
    if len(sensor_data) != 360:
        raise ValueError(
            f"Expected 360 readings, received {len(sensor_data)}."
        )

    data = sensor_data[REQUIRED_COLUMNS].copy()

    # Convert readings into numbers
    for col in REQUIRED_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan)

    # Handle invalid sensor readings
    data.loc[data["HR_Processed"] <= 0, "HR_Processed"] = np.nan
    data.loc[data["HRV"] < 0, "HRV"] = np.nan

    # Calculate acceleration magnitude
    data["Acceleration_Magnitude"] = np.sqrt(
        data["IMU_5_ax_g_"] ** 2
        + data["IMU_5_ay_g_"] ** 2
        + data["IMU_5_az_g_"] ** 2
    )

    # Obtain the gyroscope signal
    data["Gyroscope_Proxy"] = data["IMU_5_gx_dps_"].abs()

    # Select the five model inputs
    data = data[SENSOR_COLUMNS]

    # Replace missing values using training medians
    data = data.fillna(SENSOR_MEDIANS)

    # Convert 360 × 5 readings into 1,800 values
    flattened = data.to_numpy(dtype=np.float32).reshape(1, 1800)

    # Give every flattened value a unique column name
    feature_names = [
        f"{sensor}_t{t:03d}"
        for t in range(360)
        for sensor in SENSOR_COLUMNS
    ]

    return pd.DataFrame(flattened, columns=feature_names)


def predict_fatigue(sensor_data):
    features = extract_features(sensor_data)

    # Use the XGBoost model
    score = float(np.clip(model.predict(features)[0], 0, 10))

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
