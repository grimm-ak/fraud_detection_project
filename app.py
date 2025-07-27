import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Required features (must match training exactly)
required_features = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiffOrg', 'balanceDiffDest',
    'isFlaggedFraud', 'type_CASH_OUT', 'type_TRANSFER'
]

st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💳 Fraud Transaction Detection")
st.markdown("Enter transaction details to predict if it's **Fraudulent or Legit**.")

# Input form
with st.form("fraud_form"):
    step = st.number_input("Step", min_value=1)
    amount = st.number_input("Transaction Amount", min_value=0.0)
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0)
    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0)
    isFlaggedFraud = st.selectbox("Is Flagged Fraud?", [0, 1])
    tx_type = st.selectbox("Transaction Type", ['TRANSFER', 'CASH_OUT'])

    submit = st.form_submit_button("🚀 Predict")

if submit:
    # One-hot encode transaction type
    type_TRANSFER = 1 if tx_type == 'TRANSFER' else 0
    type_CASH_OUT = 1 if tx_type == 'CASH_OUT' else 0

    # Compute engineered features
    balanceDiffOrg = oldbalanceOrg - newbalanceOrig
    balanceDiffDest = newbalanceDest - oldbalanceDest

    # Build input DataFrame
    input_data = pd.DataFrame([{
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'balanceDiffOrg': balanceDiffOrg,
        'balanceDiffDest': balanceDiffDest,
        'isFlaggedFraud': isFlaggedFraud,
        'type_CASH_OUT': type_CASH_OUT,
        'type_TRANSFER': type_TRANSFER
    }])[required_features]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability of fraud: {prob:.2f})")

    # SHAP Explanation
    st.markdown("#### 🔍 SHAP Explanation")
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_scaled)

        expected_value = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1]
        shap_val = shap_values[0].values if not isinstance(shap_values, list) else shap_values[1][0]

        shap.plots._waterfall.waterfall_legacy(
            expected_value,
            shap_val,
            feature_names=required_features
        )
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be displayed: {str(e)}")
