import streamlit as st
import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Define the correct feature names used during training (same order)
feature_names = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiffOrg', 'balanceDiffDest'
]

# Streamlit UI
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Fraud Detection System")
st.markdown("Enter transaction details below:")

# User input form
with st.form("input_form"):
    step = st.number_input("Step", min_value=1)
    amount = st.number_input("Transaction Amount")
    oldbalanceOrg = st.number_input("Old Balance (Origin)")
    newbalanceOrig = st.number_input("New Balance (Origin)")
    oldbalanceDest = st.number_input("Old Balance (Destination)")
    newbalanceDest = st.number_input("New Balance (Destination)")

    submitted = st.form_submit_button("Detect Fraud")

if submitted:
    # Derived features
    balanceDiffOrg = newbalanceOrig - oldbalanceOrg
    balanceDiffDest = newbalanceDest - oldbalanceDest

    # Prepare input with correct column names
    input_data = {
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'balanceDiffOrg': balanceDiffOrg,
        'balanceDiffDest': balanceDiffDest
    }

    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.error(f"⚠️ Likely Fraudulent Transaction! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction. (Probability of Fraud: {prob:.2f})")

    # SHAP Explanation
    st.markdown("#### 🔎 SHAP Explanation")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_scaled)

    # Plot SHAP
    shap.plots.waterfall(shap_values[0], max_display=10)
