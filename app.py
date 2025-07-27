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
        st.error("❌ Model or scaler file not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# --- SHAP Explainer ---
@st.cache_resource
def get_shap_explainer(_model):
    return shap.Explainer(_model)

explainer = get_shap_explainer(model)

# --- Feature columns (must match training) ---
feature_columns = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiffOrg', 'balanceDiffDest',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# --- UI Header ---
st.title("💸 Real-Time Fraud Detection System")
st.markdown("Enter transaction details to predict if it's fraudulent.")

# 🔐 Notes Section
with st.expander("ℹ️ About this Model"):
    st.markdown("""
    - **Model**: LightGBM Classifier trained on a synthetic dataset derived from real-world patterns.
    - **Input Features**: Transaction type, amounts, balances, and derived features.
    - **Confidence**: We show the probability of fraud to enhance transparency.
    - **User Experience Note**: Lightweight UI designed for instant feedback.
    - **Security Note**: No user data is stored or transmitted. All computation is local to your session.
    """)

st.divider()
st.header("📥 Transaction Details")

# --- Input UI ---
col1, col2 = st.columns(2)
with col1:
    step = st.number_input("Step (hour)", min_value=1, value=1)
    amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f")
    oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=10000.0, format="%.2f")
    newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=9000.0, format="%.2f")
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=500.0, format="%.2f")
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=1500.0, format="%.2f")

with col2:
    transaction_type = st.selectbox(
        "Transaction Type",
        ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
    )
    st.markdown("*(Note: We use derived features instead of anonymized V1–V28.)*")

# Spacer to bring SHAP plot into view without scroll
st.markdown("<div style='margin-top: 20px'></div>", unsafe_allow_html=True)

# --- Predict Button ---
if st.button("🔍 Predict Fraud"):
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

    # Scale input
    input_scaled = scaler.transform(input_df)
    scaled_df = pd.DataFrame(input_scaled, columns=feature_columns)

    # Prediction
    prediction_proba = model.predict_proba(scaled_df)[:, 1][0]
    prediction_label = model.predict(scaled_df)[0]

    st.subheader("🧾 Prediction Result")
    if prediction_label == 1:
        st.error("🔴 FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success("🟢 LEGITIMATE TRANSACTION.")

    st.markdown(f"**Model Confidence (Fraud Probability)**: `{prediction_proba:.4f}`")

    # --- SHAP Explanation ---
    st.subheader("📊 Why this prediction?")
    try:
        shap_values = explainer(scaled_df)
        plt.clf()
        shap.plots.waterfall(shap_values[0], show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        st.caption("🔎 Red increases fraud probability; blue reduces it.")
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be displayed: {e}")
