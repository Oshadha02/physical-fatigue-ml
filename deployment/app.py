
import streamlit as st
import pandas as pd

from predict import predict_fatigue, REQUIRED_COLUMNS
from explain import explain_prediction


# Configure the Streamlit application
st.set_page_config(
    page_title="Physical Fatigue Detection",
    page_icon="🦺",
    layout="wide"
)


# Application heading
st.title("Physical Fatigue Detection System")
st.write("Wearable Sensor-Based Fatigue Prediction")

st.divider()


# Step 1: Upload sensor data
st.subheader("1. Upload Sensor Data")

# worker id
worker_id = st.text_input(
    "Worker ID ",
    placeholder="Example: P21"
)

worker_name = worker_id.strip() or "The worker"

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload a CSV containing 360 sensor readings",
    type=["csv"]
)


if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        sensor_data = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully!")

        # Display sensor data preview
        st.write("Sensor Data Preview")

        st.dataframe(
            sensor_data.head(10),
            use_container_width=True
        )

        st.write("Total readings:", len(sensor_data))

        # Validate required sensor columns
        missing_columns = [
            column for column in REQUIRED_COLUMNS
            if column not in sensor_data.columns
        ]

        if missing_columns:
            st.error(
                f"Missing sensor columns: {missing_columns}"
            )

        elif len(sensor_data) != 360:
            st.error(
                "Please upload exactly 360 sensor readings."
            )

        else:
            # Step 2: Predict fatigue
            st.subheader("2. Predict Physical Fatigue")

            if st.button(
                "Predict Fatigue",
                type="primary"
            ):

                # Perform fatigue prediction
                with st.spinner("Analyzing sensor readings..."):
                    result = predict_fatigue(sensor_data)

                level = result["fatigue_level"]

                # Step 3: Display prediction results
                st.divider()
                st.subheader("3. Prediction Results")

                st.write(f"**Worker:** {worker_name}")



                
                # Display fatigue level in the center
                colors = {
                    "Low": "#beb338",
                    "Moderate": "#16a34a",
                    "High": "#dc2626"
                }

                level_color = colors.get(level, "#16a34a")

                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 30px;">
                        <h3>Fatigue Level</h3>
                        <div style="
                            background-color: {level_color};
                            color: white;
                            font-size: 42px;
                            font-weight: bold;
                            padding: 25px;
                            border-radius: 12px;
                        ">
                            {level}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                

                # Step 4: Natural-language Explainable AI
                st.divider()
                st.subheader("4. AI Fatigue Explanation")

                with st.spinner("Generating AI explanation..."):
                    explanations = explain_prediction(
                        result["features"],
                        worker_name
                    )

                st.write(
                    "The following explanations describe "
                    "how different wearable sensor features "
                    "influenced the model's prediction."
                )

                # Display explanations for each sensor
                for item in explanations:
                    st.markdown(
                        f"**{item['sensor']}**"
                    )

                    st.write(item["text"])

                    st.write("")



    except Exception as error:
        st.error(
            f"Error processing sensor data: {error}"
        )

else:
    st.info(
        "Upload sensor data to begin fatigue prediction."
    )
