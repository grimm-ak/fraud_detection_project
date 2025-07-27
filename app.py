import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

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

@st.cache_resource
def get_shap_explainer(_model):
    return shap.Explainer(_model)

explainer = get_shap_explainer(model)

# --- Feature columns ---
feature_columns = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiffOrg', 'balanceDiffDest',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# --- Title & Info Section ---
st.title("💸 Real-Time Fraud Detection System")

with st.expander("ℹ️ Model Info"):
    st.markdown("""
    - LightGBM classifier trained on 6M+ transactions  
    - Uses derived balance features and one-hot encoded types  
    - Calibrated using GridSearchCV with class balancing  
    """)

with st.expander("🔒 Security Note"):
    st.markdown("""
    - All predictions happen **locally**  
    - No transaction data is stored or sent externally  
    """)

with st.expander("🎯 User Experience"):
    st.markdown("""
    - Instant predictions with clear visual explanation  
    - Waterfall SHAP plot shows **why** behind every decision  
    """)

# --- Input UI ---
st.header("🧾 Transaction Details")
col1, col2 = st.columns(2)
with col1:
    step = st.number_input("Step (hour)", min_value=1, value=1)
    amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f")
    oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=10000.0, format="%.2f")
    newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=9000.0, format="%.2f")
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=500.0, format="%.2f")
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=1500.0, format="%.2f")
with col2:
    transaction_type = st.selectbox("Transaction Type", ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'))
    st.markdown("*(Note: Uses derived features instead of anonymized V1–V28)*")

# --- Predict Button ---
if st.button("🔍 Predict Fraud"):
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

    scaled = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled, columns=feature_columns)

    prediction_proba = model.predict_proba(scaled_df)[:, 1][0]
    prediction_label = model.predict(scaled_df)[0]

    st.subheader("🧠 Prediction Result")
    if prediction_label == 1:
        st.error("🔴 FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success("🟢 LEGITIMATE TRANSACTION.")
    st.write(f"**Fraud Probability:** `{prediction_proba:.4f}`")

    # --- SHAP Explanation ---
    st.subheader("🧩 Why this prediction?")
    shap_values = explainer(scaled_df)
    plt.clf()
    shap.plots.waterfall(shap_values[0], show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    st.caption("Red pushes toward fraud; blue toward legitimate.")

    # --- Save Prediction Option ---
    if st.checkbox("💾 Save this prediction"):
        save_dict = input_df.copy()
        save_dict['Prediction'] = prediction_label
        save_dict['Fraud Probability'] = round(prediction_proba, 4)

        save_path = "saved_predictions.csv"
        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path)
            updated_df = pd.concat([existing_df, save_dict], ignore_index=True)
        else:
            updated_df = save_dict

        updated_df.to_csv(save_path, index=False)
        st.success("✅ Prediction saved to `saved_predictions.csv`")
