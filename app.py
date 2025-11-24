import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Wine Quality Prediction", layout="centered")

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🍷 Wine Quality Prediction App")
st.write("Predict whether wine is **Good (Quality ≥ 7)** or **Not Good** using chemical properties.")

# All feature names (order must match your model training)
feature_names = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

# Default mean values (from dataset)
defaults = {
    "fixed acidity": 8.3,
    "volatile acidity": 0.52,
    "citric acid": 0.27,
    "residual sugar": 2.5,
    "chlorides": 0.087,
    "free sulfur dioxide": 15.87,
    "total sulfur dioxide": 46.47,
    "density": 0.9967,
    "pH": 3.31,
    "sulphates": 0.56,
    "alcohol": 10.0
}

# Sidebar input option
st.sidebar.header("Select Input Mode")
input_mode = st.sidebar.radio("Choose one:", ["Manual Input", "Upload CSV (Batch)"])

# --------------------------------
# Manual Prediction UI
# --------------------------------
if input_mode == "Manual Input":
    st.subheader("Enter Wine Chemical Properties")

    inputs = []
    for feature in feature_names:
        value = st.number_input(feature, value=float(defaults[feature]))
        inputs.append(value)

    if st.button("Predict"):
        X = np.array([inputs])
        pred = model.predict(X)[0]

        # Probability (if model supports it)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]
        else:
            prob = None

        if pred == 1:
            st.success("👍 The wine is **GOOD** (Quality ≥ 7)")
        else:
            st.error("👎 The wine is **NOT GOOD** (Quality < 7)")

        if prob is not None:
            st.info(f"Confidence: {prob:.2f}")

# --------------------------------
# CSV Upload Prediction
# --------------------------------
else:
    st.subheader("Upload CSV for Batch Prediction")

    st.write("CSV must contain these columns in order:")
    st.code(", ".join(feature_names))

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Ensure features exist
        if list(df.columns) != feature_names:
            st.warning("⚠ CSV columns do not match expected feature order.")
            st.write("Expected:", feature_names)
        else:
            predictions = model.predict(df.values)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(df.values)[:, 1]
                df["prob_good"] = prob

            df["prediction"] = predictions

            st.write("Predictions:")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "batch_predictions.csv", "text/csv")
