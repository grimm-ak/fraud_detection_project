import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Fraud Detection App", layout="wide")

# ----- Sidebar Info Section -----
with st.sidebar.expander("ℹ️ Model Info", expanded=True):
    st.markdown("""
**LightGBM classifier trained on 6M+ transactions**
- Uses derived balance features and one-hot encoded types
- Calibrated using GridSearchCV with class balancing
    """)

with st.sidebar.expander("🔒 Security Note", expanded=True):
    st.markdown("""
**All predictions happen locally**
- No transaction data is stored or sent externally
    """)

with st.sidebar.expander("🎯 User Experience", expanded=True):
    st.markdown("""
**Instant predictions with visual explanation**
- Waterfall SHAP plot shows why the model made a decision
    """)

# ----- UI -----
st.title("💳 Real-Time Credit Card Fraud Detection")

st.markdown("Enter the transaction details below:")

# Input fields
col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step (Time Step)", min_value=1, value=1)
    amount = st.number_input("Amount ($)", min_value=0.0, value=50.0)
    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=1000.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=950.0)

with col2:
    type_ = st.selectbox("Transaction Type", ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'])
    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=0.0)

if st.button("🔍 Predict Fraud"):
    # Derived Features
    df = pd.DataFrame({
        'step': [step],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'errorBalanceOrig': [oldbalanceOrg - newbalanceOrig - amount],
        'errorBalanceDest': [newbalanceDest + amount - oldbalanceDest],
    })

    # One-hot encode 'type'
    for t in ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']:
        df[f"type_{t}"] = [1 if type_ == t else 0]

    # Reorder features to match training data
    feature_order = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                     'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig',
                     'errorBalanceDest', 'type_CASH_OUT', 'type_PAYMENT',
                     'type_CASH_IN', 'type_TRANSFER', 'type_DEBIT']

    df = df[feature_order]

    # Scale
    scaled_df = scaler.transform(df)

    # Prediction and Confidence
    prediction = model.predict(scaled_df)[0]
    proba = model.predict_proba(scaled_df)[0][1]  # Probability of fraud
    confidence = proba if prediction == 1 else 1 - proba

    # Confidence level
    if confidence >= 0.9:
        confidence_level = "High Confidence"
        confidence_msg = "✅ The model is highly confident this is a legitimate transaction." if prediction == 0 else "⚠️ The model is highly confident this transaction is fraudulent."
    elif confidence >= 0.6:
        confidence_level = "Medium Confidence"
        confidence_msg = "⚠️ The model is moderately confident. Please double-check this result."
    else:
        confidence_level = "Low Confidence"
        confidence_msg = "⚠️ The model has low certainty, so use this prediction carefully."

    # Output
    st.markdown("### 📢 Prediction Result")
    if prediction == 1:
        st.error("🔴 FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success("🟢 LEGITIMATE TRANSACTION.")

    st.markdown(f"🧠 **Model Confidence:** {confidence_level} (`{confidence:.4f}`)")
    st.info(confidence_msg)

    # SHAP
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_df)
        st.markdown("---")
        st.subheader("🔎 SHAP Explanation (Waterfall Plot)")
        fig, ax = plt.subplots(figsize=(10, 3))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value[1], shap_values[1][0], df.columns, max_display=14, show=False
        )
        st.pyplot(fig)
    except Exception as e:
        st.warning("⚠️ SHAP explanation could not be displayed: " + str(e))
