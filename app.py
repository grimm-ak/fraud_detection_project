import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

# Load the model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Custom preset transactions
presets = {
    "None": {},
    "🔴 Typical Fraud (Cash Out)": {
        "step": 350,
        "amount": 9000.0,
        "oldbalanceOrg": 0.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 9000.0,
        "transaction_type": "CASH_OUT"
    },
    "🟢 Safe Payment": {
        "step": 120,
        "amount": 500.0,
        "oldbalanceOrg": 1500.0,
        "newbalanceOrig": 1000.0,
        "oldbalanceDest": 2000.0,
        "newbalanceDest": 2500.0,
        "transaction_type": "PAYMENT"
    },
    "🔁 Suspicious Transfer": {
        "step": 450,
        "amount": 10000.0,
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 11000.0,
        "transaction_type": "TRANSFER"
    }
}

st.set_page_config(layout="wide")
st.title("💳 Fraud Detection App")

with st.expander("ℹ️ Model Info"):
    st.markdown("""
    **LightGBM classifier trained on 6M+ transactions**  
    - Uses derived balance features and one-hot encoded types  
    - Calibrated using GridSearchCV with class balancing  
    """)

with st.expander("🔒 Security Note"):
    st.markdown("All predictions happen locally. No transaction data is stored or sent externally.")

with st.expander("🎯 User Experience"):
    st.markdown("""
    - Instant predictions with clear visual explanation  
    - Waterfall SHAP plot shows the _why_ behind every decision  
    """)

tab1, tab2 = st.tabs(["🔍 Predict", "📊 Feature Impact"])

with tab1:
    st.subheader("📥 Enter Transaction Details")

    selected_preset = st.selectbox("🎛️ Load a Test Transaction Preset", list(presets.keys()))
    preset = presets.get(selected_preset, {})

    step = st.number_input("Step (hour)", min_value=1, value=preset.get("step", 1))
    amount = st.number_input("Amount", min_value=0.0, value=preset.get("amount", 1000.0), format="%.2f")
    oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=preset.get("oldbalanceOrg", 10000.0), format="%.2f")
    newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=preset.get("newbalanceOrig", 9000.0), format="%.2f")
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=preset.get("oldbalanceDest", 500.0), format="%.2f")
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=preset.get("newbalanceDest", 1500.0), format="%.2f")

    transaction_type = st.selectbox(
        "Transaction Type",
        ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'],
        index=['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'].index(preset.get("transaction_type", "CASH_IN"))
    )

    if st.button("🚀 Predict"):
        type_encoded = [1 if transaction_type == t else 0 for t in ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']]
        input_data = [step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest] + type_encoded
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        pred = model.predict(input_scaled)[0]
        pred_proba = model.predict_proba(input_scaled)[0][pred]

        st.subheader("📢 Prediction Result")
        if pred == 1:
            st.markdown("🔴 **FRAUDULENT TRANSACTION.**")
        else:
            st.markdown("🟢 **LEGITIMATE TRANSACTION.**")

        if pred_proba >= 0.98:
            confidence_text = f"High Confidence ({pred_proba:.4f})"
            color = "🧠"
        elif pred_proba >= 0.90:
            confidence_text = f"Medium Confidence ({pred_proba:.4f})"
            color = "⚠️"
        else:
            confidence_text = f"Low Confidence ({pred_proba:.4f})"
            color = "❗"

        st.markdown(f"{color} **Model Confidence:** {confidence_text}")
        if pred_proba < 0.90:
            st.markdown("⚠️ Model has low certainty, so use this prediction carefully.")

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        st.session_state.shap_plot = shap_values
        st.session_state.input_scaled = input_scaled

with tab2:
    st.subheader("📊 SHAP Feature Impact")
    if "shap_plot" in st.session_state:
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(shap.Explanation(values=st.session_state.shap_plot[1][0],
                                             base_values=explainer.expected_value[1],
                                             data=st.session_state.input_scaled[0],
                                             feature_names=['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                                                            'oldbalanceDest', 'newbalanceDest', 
                                                            'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']),
                             max_display=12, show=False)
        st.pyplot(fig)
    else:
        st.info("Run a prediction first to see SHAP explanation.")
