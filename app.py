import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection System", layout="centered")

# --- Load model and scaler ---
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('best_lgbm_clf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_model_and_scaler()

@st.cache_resource
def get_shap_explainer(_model):
    return shap.Explainer(_model)

explainer = get_shap_explainer(model)

feature_columns = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiffOrg', 'balanceDiffDest',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# Title
st.title("💸 Real-Time Fraud Detection System")
st.markdown("Enter transaction details below to check for fraud in real time.")

# Form inside expander to collapse after prediction
with st.expander("🧾 Transaction Details (Click to Expand/Collapse)", expanded=True):
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
        st.markdown("*(Note: We use derived features instead of anonymized V1–V28.)*")

# --- Predict ---
if st.button("Predict Fraud"):
    # Prepare input
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    input_df['step'] = step
    input_df['amount'] = amount
    input_df['oldbalanceOrg'] = oldbalanceOrg
    input_df['newbalanceOrig'] = newbalanceOrig
    input_df['oldbalanceDest'] = oldbalanceDest
    input_df['newbalanceDest'] = newbalanceDest
    input_df['balanceDiffOrg'] = oldbalanceOrg - newbalanceOrig
    input_df['balanceDiffDest'] = newbalanceDest - oldbalanceDest

    # One-hot encode type
    input_df['type_CASH_OUT'] = transaction_type == 'CASH_OUT'
    input_df['type_DEBIT'] = transaction_type == 'DEBIT'
    input_df['type_PAYMENT'] = transaction_type == 'PAYMENT'
    input_df['type_TRANSFER'] = transaction_type == 'TRANSFER'

    # Scale
    scaled_array = scaler.transform(input_df[feature_columns])
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns)

    # Predict
    prediction_proba = model.predict_proba(scaled_df)[:, 1][0]
    prediction_label = model.predict(scaled_df)[0]

    # --- Show result at top ---
    st.markdown("---")
    st.header("🔍 Prediction Result")
    if prediction_label == 1:
        st.error("🔴 FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success("🟢 LEGITIMATE TRANSACTION.")

    st.write(f"**Fraud Probability:** `{prediction_proba:.4f}`")

    # --- SHAP Plot ---
    st.subheader("Why this prediction? (Feature Contributions)")
    shap_values = explainer(scaled_df)
    plt.clf()
    shap.plots.waterfall(shap_values[0], show=False)
    fig = plt.gcf()
    st.pyplot(fig)

    st.info("💡 Red pushes the prediction toward fraud; blue pushes toward legitimate.")

    # --- Extra Info ---
    with st.expander("ℹ️ Model Info"):
        st.markdown("""
        - **Model**: LightGBM Classifier  
        - **Trained on**: 6M+ financial transactions  
        - **Includes**: Feature scaling, engineered features, SHAP interpretability  
        - **Transaction Types Used**: One-hot encoded  
        """)

    with st.expander("🔒 Security Note"):
        st.markdown("""
        - No user data is stored.  
        - Model runs locally or in secured cloud.  
        - This is a demo and not a substitute for financial institution systems.
        """)

    with st.expander("🎯 User Experience"):
        st.markdown("""
        - Simple two-column layout  
        - Fast feedback after click  
        - Explanation always shown below prediction
        """)

