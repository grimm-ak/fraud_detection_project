import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(layout="wide")
st.title("💳 Fraud Detection App with SHAP Explanation")

# User Inputs
st.markdown("#### 📝 Enter Transaction Details")
input_data = {
    "step": st.number_input("Step", value=1),
    "amount": st.number_input("Transaction Amount", value=1000.0),
    "oldbalanceOrg": st.number_input("Old Balance Origin", value=5000.0),
    "newbalanceOrig": st.number_input("New Balance Origin", value=4000.0),
    "oldbalanceDest": st.number_input("Old Balance Destination", value=10000.0),
    "newbalanceDest": st.number_input("New Balance Destination", value=11000.0),
    "type_CASH_OUT": st.selectbox("Type: CASH_OUT?", ["No", "Yes"]) == "Yes",
    "type_TRANSFER": st.selectbox("Type: TRANSFER?", ["No", "Yes"]) == "Yes",
}

if st.button("🔍 Predict Fraud"):

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert boolean to int
    input_df["type_CASH_OUT"] = input_df["type_CASH_OUT"].astype(int)
    input_df["type_TRANSFER"] = input_df["type_TRANSFER"].astype(int)

    # Add engineered features (very important!)
    input_df["balanceDiffOrg"] = input_df["oldbalanceOrg"] - input_df["newbalanceOrig"]
    input_df["balanceDiffDest"] = input_df["newbalanceDest"] - input_df["oldbalanceDest"]

    # Reorder columns to match training order
    final_columns = [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest", "type_CASH_OUT", "type_TRANSFER",
        "balanceDiffOrg", "balanceDiffDest"
    ]
    input_df = input_df[final_columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability of Fraud: {prob:.2f})")

    # SHAP Explanation
    st.markdown("#### 🔎 SHAP Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    # Display SHAP values as waterfall plot
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.initjs()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[1],
        shap_values[1][0],
        feature_names=input_df.columns
    )
    st.pyplot(bbox_inches='tight')
