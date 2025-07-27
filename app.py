import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load saved model and scaler
model = joblib.load('best_lgbm_clf_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the required features (same order as during training)
feature_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                   'diff_origin_balances', 'is_balance_zero', 'is_amount_high']

# Streamlit UI
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💳 Fraud Transaction Detection")
st.markdown("Enter transaction details to check for potential fraud.")

# Input fields
step = st.number_input("Step (Time Step)", min_value=1, max_value=744, value=1)
amount = st.number_input("Amount", min_value=0.0, step=1.0)
oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, step=1.0)
newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, step=1.0)

# Predict button
if st.button("Predict Fraud"):
    # Create input dataframe
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    input_df['step'] = step
    input_df['amount'] = amount
    input_df['oldbalanceOrg'] = oldbalanceOrg
    input_df['newbalanceOrig'] = newbalanceOrig
    input_df['diff_origin_balances'] = oldbalanceOrg - newbalanceOrig
    input_df['is_balance_zero'] = 1 if oldbalanceOrg == 0 and newbalanceOrig == 0 else 0
    input_df['is_amount_high'] = 1 if amount > 200000 else 0

    # Scale features
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]
    pred_label = "Fraud" if prediction == 1 else "Not Fraud"
    st.subheader(f"🔍 Prediction: **{pred_label}**")

    # SHAP Explanation
    st.markdown("#### 🔎 SHAP Explanation")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(scaled_input)
        shap.initjs()
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values.values[0])
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be displayed: {e}")
