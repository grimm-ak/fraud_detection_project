import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Required features (in order)
required_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                     'newbalanceDest', 'balanceDiffOrg', 'balanceDiffDest', 'type_CASH_OUT',
                     'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

# Custom CSS to fix scroll/appearance
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Collapsible Model Info ---
with st.expander("ℹ️ Model Info"):
    st.markdown("""
    **LightGBM Classifier trained on 6M+ transactions**
    - Uses derived balance features and one-hot encoded transaction types  
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
    - SHAP waterfall plot shows the *why* behind every decision
    """)

st.title("💸 Fraud Detection AI")

st.markdown("Enter transaction details below to detect fraud.")

# --- Input UI ---
col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step (time)", value=1)
    amount = st.number_input("Transaction Amount", value=1000.0)
    oldbalanceOrg = st.number_input("Old Balance Origin", value=5000.0)
    newbalanceOrig = st.number_input("New Balance Origin", value=4000.0)
    oldbalanceDest = st.number_input("Old Balance Destination", value=1000.0)
    newbalanceDest = st.number_input("New Balance Destination", value=2000.0)

with col2:
    tx_type = st.selectbox("Transaction Type", ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])

# Derived features
balanceDiffOrg = oldbalanceOrg - newbalanceOrig
balanceDiffDest = newbalanceDest - oldbalanceDest

# One-hot encoding
type_CASH_OUT = 1 if tx_type == 'CASH_OUT' else 0
type_DEBIT = 1 if tx_type == 'DEBIT' else 0
type_PAYMENT = 1 if tx_type == 'PAYMENT' else 0
type_TRANSFER = 1 if tx_type == 'TRANSFER' else 0

# Create feature vector
input_data = np.array([[step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest,
                        newbalanceDest, balanceDiffOrg, balanceDiffDest, type_CASH_OUT,
                        type_DEBIT, type_PAYMENT, type_TRANSFER]])

# Scale features
scaled_input = scaler.transform(input_data)

# --- Predict Button ---
if st.button("🚀 Predict Fraud"):
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]  # probability of fraud

    st.markdown("### 🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {proba:.2f})")
    else:
        st.success(f"✅ Transaction is Legitimate (Confidence: {1 - proba:.2f})")

    
        # --- SHAP Explanation ---
    try:
        st.markdown("#### 🧠 Why this prediction? (SHAP Explanation)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_input)
        shap.initjs()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value[1],
            shap_values[1][0],
            feature_names=required_features,
            max_display=12,
            show=False,
            ax=ax  # ✅ Attach to axis
        )
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be displayed: {str(e)}")


    # --- Save Prediction ---
    try:
        result = {
            "Step": step,
            "Amount": amount,
            "OldBalanceOrg": oldbalanceOrg,
            "NewBalanceOrig": newbalanceOrig,
            "OldBalanceDest": oldbalanceDest,
            "NewBalanceDest": newbalanceDest,
            "Type": tx_type,
            "Prediction": "Fraud" if prediction == 1 else "Legit",
            "Confidence": round(proba if prediction == 1 else 1 - proba, 4)
        }

        result_df = pd.DataFrame([result])
        if os.path.exists("saved_predictions.csv"):
            result_df.to_csv("saved_predictions.csv", mode="a", header=False, index=False)
        else:
            result_df.to_csv("saved_predictions.csv", index=False)
    except:
        st.warning("Prediction could not be saved.")

# --- View Saved Predictions ---
st.subheader("📊 Saved Predictions History")

if os.path.exists("saved_predictions.csv"):
    saved_df = pd.read_csv("saved_predictions.csv")

    show_only_fraud = st.checkbox("Show only fraud predictions")
    if show_only_fraud:
        saved_df = saved_df[saved_df['Prediction'] == 'Fraud']

    st.dataframe(saved_df.tail(50), use_container_width=True)

    # Download
    csv_download = saved_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv_download, "saved_predictions.csv", "text/csv")

    # Clear saved data
    if st.button("🗑️ Clear All Saved Predictions"):
        os.remove("saved_predictions.csv")
        st.success("All saved predictions cleared. Please refresh.")
        st.stop()
else:
    st.info("No predictions saved yet.")
