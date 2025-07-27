import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="💳 Fraud Detection", layout="wide")
st.title("💳 Real-Time Fraud Detection App")

# Sidebar - Model Information
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.markdown("**Model:** LightGBM Classifier")
st.sidebar.markdown("**Test AUC:** ~0.98")
st.sidebar.markdown("**Precision:** ~0.93")
st.sidebar.markdown("**Recall:** ~0.90")
st.sidebar.markdown("**Interpretability:** SHAP Waterfall Plot")

# Sidebar - Security Note
st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ **Note:** This is a demo project for educational purposes. Real-world fraud detection systems involve additional security, regulatory compliance, and complex pipelines.")

# Load model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Required features
required_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                     'oldbalanceDest', 'newbalanceDest', 'balanceDiffOrg',
                     'balanceDiffDest', 'type_CASH_OUT', 'type_DEBIT',
                     'type_PAYMENT', 'type_TRANSFER']

# UI inputs
st.markdown("#### 📝 Enter Transaction Details")

user_input = {}
col1, col2 = st.columns(2)
with col1:
    user_input['step'] = st.number_input("Step (Hour)", min_value=1, max_value=744, value=1)
    user_input['amount'] = st.number_input("Amount", min_value=0.0)
    user_input['oldbalanceOrg'] = st.number_input("Old Balance Origin", min_value=0.0)
    user_input['newbalanceOrig'] = st.number_input("New Balance Origin", min_value=0.0)
    user_input['balanceDiffOrg'] = st.number_input("Balance Diff Origin", min_value=-1e7, max_value=1e7)

with col2:
    user_input['oldbalanceDest'] = st.number_input("Old Balance Dest", min_value=0.0)
    user_input['newbalanceDest'] = st.number_input("New Balance Dest", min_value=0.0)
    user_input['balanceDiffDest'] = st.number_input("Balance Diff Dest", min_value=-1e7, max_value=1e7)
    tx_type = st.selectbox("Transaction Type", ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    for t in ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        user_input[f"type_{t}"] = 1 if tx_type == t else 0

# Reset button
if st.button("🔄 Reset Inputs"):
    st.experimental_rerun()

# Prediction
if st.button("🚨 Predict Fraud"):
    input_df = pd.DataFrame([user_input])
    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)[0]
    pred_proba = model.predict_proba(scaled_input)[0][1]

    st.markdown("#### 🧮 Fraud Prediction")
    st.write(f"**Prediction:** {'FRAUD ❗' if pred == 1 else 'Not Fraud ✅'}")
    st.write(f"**Confidence:** {pred_proba * 100:.2f}%")

    # Visual confidence bar
    st.markdown("#### 🎯 Model Confidence Score")
    st.progress(int(pred_proba * 100))

    # SHAP Explanation
    st.markdown("#### 🔎 SHAP Explanation")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_input)
        shap.initjs()
        fig, ax = plt.subplots(figsize=(10, 3))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value[1], shap_values[1][0], feature_names=input_df.columns.tolist(), max_display=12, show=False
        )
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be displayed: {str(e)}")
