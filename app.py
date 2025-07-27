import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Fraud Detection App", layout="centered")

# --- Load Model and Scaler ---
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# --- Feature columns (must match training) ---
feature_columns = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiffOrg', 'balanceDiffDest',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# --- Required features for batch prediction ---
required_features = feature_columns.copy()

# --- Title ---
st.title("💳 Fraud Detection System")

# --- Choose mode ---
mode = st.sidebar.radio("Choose Input Mode", ["Single Prediction", "Batch Prediction (CSV Upload)"])

if mode == "Single Prediction":
    st.header("🧾 Enter Transaction Details")

    step = st.number_input("Step (Time)", min_value=1, max_value=1000, value=1)
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    oldbalanceOrg = st.number_input("Original Balance (Sender)", min_value=0.0, value=1000.0)
    newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=900.0)
    oldbalanceDest = st.number_input("Original Balance (Receiver)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=100.0)

    transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

    # Manual one-hot encoding
    type_CASH_OUT = int(transaction_type == "CASH_OUT")
    type_DEBIT = int(transaction_type == "DEBIT")
    type_PAYMENT = int(transaction_type == "PAYMENT")
    type_TRANSFER = int(transaction_type == "TRANSFER")

    balanceDiffOrg = oldbalanceOrg - newbalanceOrig
    balanceDiffDest = newbalanceDest - oldbalanceDest

    input_data = pd.DataFrame([[
        step, amount, oldbalanceOrg, newbalanceOrig,
        oldbalanceDest, newbalanceDest,
        balanceDiffOrg, balanceDiffDest,
        type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
    ]], columns=feature_columns)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction")
    if prediction == 1:
        st.error(f"⚠️ This transaction is predicted to be FRAUDULENT with probability {probability:.2f}")
    else:
        st.success(f"✅ This transaction is predicted to be LEGITIMATE with probability {1 - probability:.2f}")

    # SHAP Explanation
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        st.markdown("#### 🔎 SHAP Explanation (Waterfall Plot)")
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value[1],
            shap_values[1][0],
            feature_names=feature_columns
        )
        st.pyplot(bbox_inches="tight")
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be displayed: {e}")

elif mode == "Batch Prediction (CSV Upload)":
    st.header("📁 Upload Transactions CSV")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        missing_cols = [col for col in required_features if col not in data.columns]
        if missing_cols:
            st.error(f"The following required columns are missing: {missing_cols}")
        else:
            input_scaled = scaler.transform(data[required_features])
            preds = model.predict(input_scaled)
            data["is_fraud_predicted"] = preds

            st.success("✅ Predictions completed")
            st.dataframe(data.head())

            csv_download = data.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results CSV", data=csv_download, file_name="fraud_predictions.csv", mime="text/csv")
