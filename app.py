import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Breast Cancer Predictor /nGhabru Nkos Mi tujha PAthishi aahe✋",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# =========================================================
# TITLE
# =========================================================

st.title("🩺 Breast Cancer Prediction")

st.write(
    "An educational Machine Learning application "
    "using Logistic Regression."
)

st.info(
    "This application is for educational purposes only "
    "and is not a medical diagnostic tool."
)


# =========================================================
# FEATURE NAMES
# =========================================================

feature_names = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",

    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",

    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]


# =========================================================
# DEFAULT SAMPLE VALUES
# =========================================================

default_values = {
    "mean radius": 14.0,
    "mean texture": 20.0,
    "mean perimeter": 90.0,
    "mean area": 600.0,
    "mean smoothness": 0.10,
    "mean compactness": 0.10,
    "mean concavity": 0.08,
    "mean concave points": 0.05,
    "mean symmetry": 0.18,
    "mean fractal dimension": 0.06,

    "radius error": 0.4,
    "texture error": 1.0,
    "perimeter error": 2.5,
    "area error": 40.0,
    "smoothness error": 0.007,
    "compactness error": 0.02,
    "concavity error": 0.03,
    "concave points error": 0.01,
    "symmetry error": 0.02,
    "fractal dimension error": 0.003,

    "worst radius": 16.0,
    "worst texture": 25.0,
    "worst perimeter": 105.0,
    "worst area": 800.0,
    "worst smoothness": 0.14,
    "worst compactness": 0.25,
    "worst concavity": 0.25,
    "worst concave points": 0.12,
    "worst symmetry": 0.28,
    "worst fractal dimension": 0.08
}


# =========================================================
# INPUT SECTION
# =========================================================

st.header("🔢 Enter Tumor Measurements")

st.caption(
    "Enter values for the 30 features used by the trained model."
)


# Dictionary to store user inputs
inputs = {}


# =========================================================
# MEAN FEATURES
# =========================================================

st.subheader("Mean Measurements")

col1, col2 = st.columns(2)

mean_features = feature_names[0:10]

for i, feature in enumerate(mean_features):

    if i < 5:
        with col1:
            inputs[feature] = st.number_input(
                feature.title(),
                min_value=0.0,
                value=float(default_values[feature]),
                format="%.6f"
            )

    else:
        with col2:
            inputs[feature] = st.number_input(
                feature.title(),
                min_value=0.0,
                value=float(default_values[feature]),
                format="%.6f"
            )


# =========================================================
# ERROR FEATURES
# =========================================================

st.subheader("Error Measurements")

col1, col2 = st.columns(2)

error_features = feature_names[10:20]

for i, feature in enumerate(error_features):

    if i < 5:
        with col1:
            inputs[feature] = st.number_input(
                feature.title(),
                min_value=0.0,
                value=float(default_values[feature]),
                format="%.6f"
            )

    else:
        with col2:
            inputs[feature] = st.number_input(
                feature.title(),
                min_value=0.0,
                value=float(default_values[feature]),
                format="%.6f"
            )


# =========================================================
# WORST FEATURES
# =========================================================

st.subheader("Worst Measurements")

col1, col2 = st.columns(2)

worst_features = feature_names[20:30]

for i, feature in enumerate(worst_features):

    if i < 5:
        with col1:
            inputs[feature] = st.number_input(
                feature.title(),
                min_value=0.0,
                value=float(default_values[feature]),
                format="%.6f"
            )

    else:
        with col2:
            inputs[feature] = st.number_input(
                feature.title(),
                min_value=0.0,
                value=float(default_values[feature]),
                format="%.6f"
            )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔍 Predict",
    use_container_width=True
)


# =========================================================
# PREDICTION LOGIC
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # Convert user inputs into DataFrame
    # -----------------------------------------------------

    input_data = pd.DataFrame([inputs])

    # Make sure feature order is exactly the same
    # as the training data
    input_data = input_data[feature_names]


    # -----------------------------------------------------
    # Scale the input
    # -----------------------------------------------------

    input_scaled = scaler.transform(input_data)


    # -----------------------------------------------------
    # Make prediction
    # -----------------------------------------------------

    prediction = model.predict(input_scaled)


    # ----------------------------------------------------- 
    # Get probability
    # -----------------------------------------------------

    probability = model.predict_proba(input_scaled)


    # -----------------------------------------------------
    # Convert prediction to readable result
    # -----------------------------------------------------

    if prediction[0] == 0:

        result = "Malignant"

    else:

        result = "Benign"


    # -----------------------------------------------------
    # Extract probabilities
    # -----------------------------------------------------

    malignant_probability = probability[0][0] * 100

    benign_probability = probability[0][1] * 100


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.divider()

    st.header("📊 Prediction Result")


    if result == "Benign":

        st.success(
            f"### Prediction: {result}"
        )

    else:

        st.error(
            f"### Prediction: {result}"
        )


    # -----------------------------------------------------
    # Probability columns
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Malignant Probability",
            f"{malignant_probability:.2f}%"
        )


    with col2:

        st.metric(
            "Benign Probability",
            f"{benign_probability:.2f}%"
        )


    # -----------------------------------------------------
    # Probability bar
    # -----------------------------------------------------

    st.subheader("Model Probability")

    probability_data = pd.DataFrame(
        {
            "Class": [
                "Malignant",
                "Benign"
            ],

            "Probability": [
                malignant_probability,
                benign_probability
            ]
        }
    )

    st.bar_chart(
        probability_data.set_index("Class")
    )


    # -----------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------

    st.warning(
        "⚠️ This prediction is generated by a machine learning "
        "model for educational purposes. It must not be used "
        "for actual medical diagnosis or treatment decisions."
    )