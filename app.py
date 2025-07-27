import streamlit as st
import numpy as np
import pandas as pd
import shap
import joblib
import json
import lightgbm as lgb
import matplotlib.pyplot as plt

# Load model, scaler, and feature names
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Fraud Detection - LightGBM Model")
st.write("Enter transaction details to predict whether it's fraudulent.")

# Split UI into columns
col1, col2 = st.columns(2)

# User inputs for all 12 features
with col1:
    step = st.number_input("Step (Time step)", min_value=0, max_value=100000, value=1)
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, value=5000.0)
    newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, value=4000.0)
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=1000.0)
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=2000.0)

with col2:
    balanceDiffOrg = st.number_input("Balance Difference Origin", value=oldbalanceOrg - newbalanceOrig)
    balanceDiffDest = st.number_input("Balance Difference Destination", value=newbalanceDest - oldbalanceDest)
    type_CASH_OUT = st.checkbox("Transaction Type: CASH_OUT")
    type_DEBIT = st.checkbox("Transaction Type: DEBIT")
    type_PAYMENT = st.checkbox("Transaction Type: PAYMENT")
    type_TRANSFER = st.checkbox("Transaction Type: TRANSFER")

# Prepare input in correct order
input_data = pd.DataFrame([[
    step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest,
    newbalanceDest, balanceDiffOrg, balanceDiffDest,
    int(type_CASH_OUT), int(type_DEBIT),
    int(type_PAYMENT), int(type_TRANSFER)
]], columns=feature_names)

# Scale input
input_scaled = scaler.transform(input_data)

# Predict and display
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! Probability: {proba:.2%}")
    else:
        st.success(f"✅ Legitimate Transaction. Probability of fraud: {proba:.2%}")
    
    # SHAP explanation
    st.subheader("🔍 SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    
    # Plot SHAP waterfall
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[1], 
        shap_values[1][0], 
        feature_names=feature_names, 
        features=input_data.iloc[0]
    )
    st.pyplot(fig)
