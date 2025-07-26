import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load trained model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Fraud Detection App")
st.write("Enter transaction details to predict if it's fraudulent.")

# ----------------------------
# Define input fields manually
# ----------------------------
st.subheader("🔢 Transaction Features")

# Transaction type one-hot
type_TRANSFER = st.selectbox("Type: TRANSFER?", [0, 1])
type_CASH_OUT = st.selectbox("Type: CASH_OUT?", [0, 1])
type_DEBIT = st.selectbox("Type: DEBIT?", [0, 1])
type_PAYMENT = st.selectbox("Type: PAYMENT?", [0, 1])

# Numeric fields
amount = st.number_input("Amount", min_value=0.0, value=3900.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=4200.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=300.0)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=0.0)

# Derived features
balanceDiffOrg = newbalanceOrig - oldbalanceOrg
balanceDiffDest = newbalanceDest - oldbalanceDest

# Collect into a DataFrame
input_data = pd.DataFrame([{
    'type_CASH_OUT': type_CASH_OUT,
    'type_DEBIT': type_DEBIT,
    'type_PAYMENT': type_PAYMENT,
    'type_TRANSFER': type_TRANSFER,
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest,
    'balanceDiffOrg': balanceDiffOrg,
    'balanceDiffDest': balanceDiffDest
}])

st.markdown("### 🧾 Final Input")
st.dataframe(input_data)

# Scale and Predict
X_scaled = scaler.transform(input_data)
pred = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0][1]

st.markdown(f"### 🎯 Prediction: {'🚨 Fraud' if pred==1 else '✅ Not Fraud'}")
st.markdown(f"**Confidence: {proba:.2%}**")

# SHAP Explanation
st.subheader("📉 Feature Impact (SHAP)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(pd.DataFrame(X_scaled, columns=input_data.columns))

fig, ax = plt.subplots(figsize=(10, 4))
shap.bar_plot(shap_values[1][0], feature_names=input_data.columns, max_display=10)
st.pyplot(fig)

st.markdown("---")
st.caption("Model: LightGBM • Explainability: SHAP • UI: Streamlit")
