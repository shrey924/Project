import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "credit_card_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("💳 Credit Card Fraud Detection")

st.markdown("""
Upload a CSV file containing transaction data.

Required columns:
Time, V1, V2, ... , V28, Amount
""")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        # Remove target column if present
        if "Class" in df.columns:
            df = df.drop("Class", axis=1)

        expected_columns = [
            'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
            'V10','V11','V12','V13','V14','V15','V16','V17',
            'V18','V19','V20','V21','V22','V23','V24','V25',
            'V26','V27','V28','Amount'
        ]

        # Ensure correct column order
        df = df[expected_columns]

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        prediction = model.predict(df)

        result_df = df.copy()

        result_df["Prediction"] = prediction

        result_df["Prediction"] = result_df["Prediction"].map({
            0: "Legitimate",
            1: "Fraud"
        })

        st.subheader("Prediction Result")
        st.dataframe(result_df.head())

        csv = result_df.to_csv(index=False)

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")