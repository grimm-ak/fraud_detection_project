import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Fraud Detection System", layout="centered")

# --- Load model and scaler ---
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('best_lgbm_clf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# --- Feature columns (must match training) ---
feature_columns = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiffOrg', 'balanceDiffDest',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

st.title("💸 Real-Time Fraud Detection System")
st.markdown("Enter transaction details to predict if it's fraudulent.")
st.header("Transaction Details")

col1, col2 = st.columns(2)
with col1:
    step = st.number_input("Step (hour)", min_value=1, value=1, key="step")
    amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f", key="amount")
    oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=10000.0, format="%.2f", key="obo")
    newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=9000.0, format="%.2f", key="nbo")
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=500.0, format="%.2f", key="obd")
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=1500.0, format="%.2f", key="nbd")

with col2:
    transaction_type = st.selectbox(
        "Transaction Type",
        ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'),
        key="type"
    )
    st.markdown("*(Other features V1-V28 are anonymized; we use engineered features.)*")

if st.button("Predict Fraud"):
    # Create input dataframe
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    input_df['step'] = step
    input_df['amount'] = amount
    input_df['oldbalanceOrg'] = oldbalanceOrg
    input_df['newbalanceOrig'] = newbalanceOrig
    input_df['oldbalanceDest'] = oldbalanceDest
    input_df['newbalanceDest'] = newbalanceDest
    input_df['balanceDiffOrg'] = oldbalanceOrg - newbalanceOrig
    input_df['balanceDiffDest'] = newbalanceDest - oldbalanceDest

    # One-hot encode transaction type
    input_df['type_CASH_OUT'] = transaction_type == 'CASH_OUT'
    input_df['type_DEBIT'] = transaction_type == 'DEBIT'
    input_df['type_PAYMENT'] = transaction_type == 'PAYMENT'
    input_df['type_TRANSFER'] = transaction_type == 'TRANSFER'
    # CASH_IN is baseline, no column set

    # Match training feature order
    input_data_processed = input_df[feature_columns]

    # Scale input
    scaled_array = scaler.transform(input_data_processed)
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns)

    # Predict
    prediction_proba = model.predict_proba(scaled_df)[:, 1][0]
    prediction_label = model.predict(scaled_df)[0]

    st.subheader("Prediction Result:")
    if prediction_label == 1:
        st.error("🔴 FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success("🟢 LEGITIMATE TRANSACTION.")

    st.write(f"**Fraud Probability:** {prediction_proba:.4f}")

   # SHAP Explanation using Waterfall Plot
st.subheader("Why this prediction? (Feature Contributions)")
explainer = shap.Explainer(model)
shap_values = explainer(scaled_df)

# Waterfall plot for single prediction
fig = shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

st.info("💡 Red pushes toward fraud, blue toward legitimate.")

