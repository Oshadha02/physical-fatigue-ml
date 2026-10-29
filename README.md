# Physical Fatigue Detection Using Wearable Sensor Data

A research prototype for estimating physical fatigue from wearable sensor readings. The project compares **Decision Tree, Random Forest, XGBoost, and LSTM** models and includes a **Streamlit** interface with **SHAP-based explanations**.

This component is being developed as part of the **Smart Wearable & IoT-Driven Construction Worker Safety System** research project (J26-IT-404). The current models were trained using an existing **manufacturing-task dataset**, not construction-site field measurements.

## How the system works

1. A user uploads a CSV containing **360 time-ordered sensor readings** (3 minutes at 2 Hz).
2. The system derives five signals: **heart rate, HRV, skin temperature, acceleration magnitude, and gyroscope movement**.
3. The trained model estimates a fatigue score on a **0–10 scale**.
4. The score is converted to **Low**, **Moderate**, or **High** fatigue.
5. **SHAP** explains which input signals contributed to the model's estimate.

| Predicted score | Fatigue category |
| --- | --- |
| `score < 4` | Low |
| `4 <= score < 7` | Moderate |
| `score >= 7` | High |

The category boundaries are research-based conventions for the 0–10 scale, not medical safety thresholds.

## Dataset and preparation

- **Source:** MXD / WorkerFatigue wearable-sensor dataset (Mohapatra et al., 2024).
- **Tasks:** Composite and Wire Harness.
- **Retained data:** 38 unique participants and 334 usable activity sessions.
- **Windowing:** 3-minute windows; 360 readings per sensor signal; **627 complete windows**.
- **Participant-based split:** 500 training windows (30 participants) and 127 test windows (8 different participants).
- **Regression target:** Estimated fatigue score; intermediate window scores were derived from session-level self-reports, rather than measured directly in every 3-minute window.

The tree-based models use one flattened row of **1,800 inputs per window** (`360 x 5`). LSTM uses the same readings as a time-ordered **`360 x 5` sequence**. The split is kept the same across all four models.

## Project structure

```text
Physical_Fatigue_ML/
├── data/                    # Source and processed data
├── deployment/
│   ├── app.py               # Streamlit user interface
│   ├── predict.py           # Input preparation and fatigue prediction
│   ├── explain.py           # SHAP explanations
│   └── models/              # Earlier deployment model and metadata
├── models/
│   ├── decision_tree_flattened.joblib
│   ├── random_forest_3min_flattened.joblib
│   ├── xgboost_3min_flattened.json
│   └── lstm_3min_sequence.pth
├── notebooks/
│   ├── 01_dataset_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 05b_window_based_decision_tree.ipynb
│   ├── 05c_window_based_random_forest.ipynb
│   ├── 05d_window_based_XGBoost.ipynb
│   ├── 05e_window_based_LSTM.ipynb
│   └── 07_explainable_ai.ipynb
├── results/
│   ├── raw_window_flattened_split.npz
│   └── raw_window_sequences_split.npz
└── README.md
```

## Run the application

From the project root, create or activate a Python virtual environment. On **Windows PowerShell**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scikit-learn xgboost shap streamlit joblib
python -m streamlit run deployment/app.py
```

Open the local URL displayed in the terminal (typically `http://localhost:8501`). Upload a CSV with exactly **360 rows**, ordered by time, containing these columns:

```text
HR_Processed
HRV
Temperature
IMU_5_ax_g_
IMU_5_ay_g_
IMU_5_az_g_
IMU_5_gx_dps_
```

**Deployment check:** Ensure `deployment/predict.py` loads `models/xgboost_3min_flattened.json`, and `deployment/explain.py` uses that *same model*. Both must use the exact sensor order, invalid-value handling, and flattening convention used during model training. The saved XGBoost JSON alone does not store the entire CSV-preparation pipeline.

## Training and saved datasets

- `03_preprocessing.ipynb` combines and cleans activity-session data.
- `05b_window_based_decision_tree.ipynb` creates the 3-minute windows, splits by participant, trains the Decision Tree, and saves both NPZ datasets.
- `05c_window_based_random_forest.ipynb` trains Random Forest using `results/raw_window_flattened_split.npz`.
- `05d_window_based_XGBoost.ipynb` trains XGBoost using the **same flattened split**.
- `05e_window_based_LSTM.ipynb` trains LSTM using `results/raw_window_sequences_split.npz`.

These NPZ files contain `X_train`, `X_test`, `y_train`, and `y_test`. **They are saved datasets, not trained models.**

## Evaluation and limitations

On the held-out participant test set, the flattened-model notebooks report:

| Model | Overall accuracy | High-fatigue recall |
| --- | ---: | ---: |
| Decision Tree | 48.03% | 5.26% (1/19) |
| Random Forest | 55.12% | 0% (0/19) |
| XGBoost | 55.91% | 0% (0/19) |

A later presentation reports **LSTM: 44.88% accuracy and 21.05% high-fatigue recall (4/19)**, while the uploaded saved LSTM notebook contains an earlier evaluation output. Re-run or document the final LSTM checkpoint before presenting that figure as reproducible.

**Important limitations:** The labels are estimates; the training data come from manufacturing tasks; the system has not been validated on construction workers using a deployed wearable prototype; and the models' ability to detect **High Fatigue** is currently inadequate for reliable safety alerts. This is a **research prototype**, not a validated medical or occupational safety device.

## Research reference

Mohapatra et al. (2024), *Wearable network for multilevel physical fatigue prediction in manufacturing workers*. Refer to the original dataset documentation and publication for dataset provenance, use conditions, and citation details.
