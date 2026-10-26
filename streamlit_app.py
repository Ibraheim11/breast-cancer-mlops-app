"""
Assignment 03 - Streamlit ML Application.

A lightweight interface that sends selected
feature values to the FastAPI prediction service.

Streamlit does not load or train the ML model.
"""

import json
import os
from pathlib import Path

import requests
import streamlit as st

# -----------------------------------------
# Configuration
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

FEATURES_PATH = PROJECT_ROOT / "models" / "final_selected_features.json"

SAMPLE_PATH = PROJECT_ROOT / "examples" / "sample_25_features.json"

DEFAULT_API_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8005",
)


# -----------------------------------------
# Page Configuration
# -----------------------------------------

st.set_page_config(
    page_title="Breast Cancer ML App",
    page_icon="🧬",
    layout="wide",
)


# -----------------------------------------
# Load Input Metadata
# -----------------------------------------


@st.cache_data
def load_input_data() -> tuple[list[str], dict[str, float]]:
    """Load feature names and sample values."""

    metadata = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))

    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))

    features = metadata["feature_names"]

    if not features or len(features) != len(set(features)):
        raise ValueError("Invalid selected-feature metadata.")

    if set(features) != set(sample):
        raise ValueError("Sample inputs do not match selected features.")

    return features, sample


try:
    selected_features, sample_values = load_input_data()

except (OSError, ValueError, KeyError, TypeError):
    st.error(
        "Cannot load the feature configuration. "
        "Run the training workflow and prepare "
        "the sample input file first."
    )
    st.stop()


# -----------------------------------------
# Sidebar
# -----------------------------------------

with st.sidebar:
    st.header("⚙️ API Configuration")

    api_url = (
        st.text_input(
            "FastAPI Base URL",
            value=DEFAULT_API_URL,
        )
        .strip()
        .rstrip("/")
    )

    st.divider()

    st.subheader("Model Information")

    st.write("Algorithm: Logistic Regression")

    st.write("Feature Selection: SelectKBest")

    st.write(f"Selected Features: {len(selected_features)}")

    st.write("Original Features: 30")

    st.divider()

    st.caption("This application connects to FastAPI " "through HTTP requests.")


# -----------------------------------------
# Main Page
# -----------------------------------------

st.title("🧬 Breast Cancer Classification")

st.markdown("""
    ### Machine Learning Prediction Application

    Enter the required feature measurements below.

    Click **Predict** to send the values to the
    FastAPI backend and display the model prediction.
    """)

st.warning(
    "Educational demonstration only. "
    "Predictions are not clinically validated "
    "and must not be used for medical decisions."
)

st.divider()


# -----------------------------------------
# Feature Input Form
# -----------------------------------------

st.subheader("📋 Input Features")

st.info(
    "The fields are pre-filled with measurements "
    "from a sample in the dataset. "
    "You can edit any value before predicting."
)

input_values = {}

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    for index, feature_name in enumerate(selected_features):

        target_column = col1 if index % 2 == 0 else col2

        with target_column:

            value = st.number_input(
                label=feature_name,
                min_value=0.0,
                value=float(sample_values[feature_name]),
                step=0.001,
                format="%.6f",
            )

            input_values[feature_name] = float(value)

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict",
        type="primary",
        use_container_width=True,
    )


# -----------------------------------------
# Send Prediction Request
# -----------------------------------------

if submitted:

    if not api_url.startswith(("http://", "https://")):
        st.error(
            "Please enter a valid FastAPI URL " "starting with http:// or https://."
        )
        st.stop()

    with st.spinner("Sending data to FastAPI..."):

        try:
            response = requests.post(
                f"{api_url}/predict",
                json=input_values,
                timeout=10,
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction request timed out. " "Please check the FastAPI service."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. "
                "Make sure the backend is running "
                "and the URL is correct."
            )

        except requests.exceptions.RequestException:
            st.error("An HTTP communication error occurred.")

        else:

            # Successful prediction
            if response.status_code == 200:

                try:
                    result = response.json()
                except ValueError:
                    st.error("The API returned invalid JSON.")
                    st.stop()

                predicted_class = result.get("predicted_class")

                class_label = result.get("class_label")

                if predicted_class not in (0, 1) or class_label not in (
                    "malignant",
                    "benign",
                ):
                    st.error("Unexpected prediction response format.")

                else:
                    st.divider()
                    st.subheader("📊 Prediction Result")

                    if class_label == "malignant":
                        st.warning("Model Prediction: Malignant")

                    else:
                        st.success("Model Prediction: Benign")

                    st.write(f"Predicted Class: {predicted_class}")

                    with st.expander("View API Response"):
                        st.json(result)

            # Pydantic validation error
            elif response.status_code == 422:

                st.error("Invalid input data. " "Please review the feature values.")

                with st.expander("Validation Details"):
                    try:
                        st.json(response.json())
                    except ValueError:
                        st.write("Validation details unavailable.")

            # Model unavailable
            elif response.status_code == 503:

                st.error(
                    "The prediction model is currently "
                    "unavailable. Check the FastAPI backend."
                )

            # Other API errors
            else:

                st.error(f"API returned HTTP {response.status_code}.")


# -----------------------------------------
# Footer
# -----------------------------------------

st.divider()

st.caption("Assignment 03 | FastAPI + Pydantic + " "Streamlit + Scikit-learn")
