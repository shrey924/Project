import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load Model
model = joblib.load("credit_card_model.pkl")

st.title("💳 Credit Card Fraud Detection")

st.markdown("""
Upload a CSV file containing transaction data.

Required columns:

Time, V1 ... V28, Amount
""")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        prediction = model.predict(df)

        df["Prediction"] = prediction

        df["Prediction"] = df["Prediction"].map({
            0: "Legitimate",
            1: "Fraud"
        })

        st.subheader("Prediction Result")
        st.dataframe(df.head())

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")