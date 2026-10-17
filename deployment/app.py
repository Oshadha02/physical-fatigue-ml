
import streamlit as st
import pandas as pd

from predict import predict_fatigue, REQUIRED_COLUMNS


# Configure the application
st.set_page_config(
    page_title="Physical Fatigue Detection",
    page_icon="🦺",
    layout="wide"
)

st.title("Physical Fatigue Detection System")
st.write("Wearable Sensor-Based Fatigue Prediction")

st.divider()

# Upload sensor readings
st.subheader("1. Upload Sensor Data")

uploaded_file = st.file_uploader(
    "Upload a CSV containing 360 sensor readings",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        sensor_data = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully!")

        # Display uploaded data
        st.write("Sensor Data Preview")
        st.dataframe(sensor_data.head(10), use_container_width=True)

        st.write("Total readings:", len(sensor_data))

        missing_columns = [
            column for column in REQUIRED_COLUMNS
            if column not in sensor_data.columns
        ]

        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")

        elif len(sensor_data) != 360:
            st.error("Please upload exactly 360 sensor readings.")

        else:
            st.subheader("2. Predict Physical Fatigue")

            if st.button("Predict Fatigue", type="primary"):
                # Send readings to the Python backend
                with st.spinner("Analyzing sensor readings..."):
                    result = predict_fatigue(sensor_data)

                score = result["fatigue_score"]
                level = result["fatigue_level"]

                st.divider()
                st.subheader("3. Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Fatigue Score", f"{score}/10")

                with col2:
                    st.metric("Fatigue Level", level)

                if level == "Low":
                    st.success("Low Fatigue Detected")
                elif level == "Moderate":
                    st.warning("Moderate Fatigue Detected")
                else:
                    st.error("High Fatigue Detected")

                # Display extracted features
                with st.expander("View Extracted Features"):
                    st.dataframe(
                        result["features"],
                        use_container_width=True
                    )

                st.info(
                    "Fatigue predictions are experimental "
                    "and are not validated safety assessments."
                )

    except Exception as error:
        st.error(f"Error processing data: {error}")

else:
    st.info("Upload sensor data to begin fatigue prediction.")
