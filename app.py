import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💸 Fraud Transaction Detector")

st.write("Enter the transaction details below:")

# Input UI
col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step (Time Step)", min_value=1)
    amount = st.number_input("Transaction Amount", min_value=0.0)
    oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0)
    newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0)

with col2:
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0)
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0)
    transaction_type = st.selectbox("Transaction Type", ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])

if st.button("🔍 Predict Fraud"):
    # Build input DataFrame
    input_df = pd.DataFrame({
        'step': [step],
        'amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest],
        'type': [transaction_type]
    })

    # Create engineered features
    input_df['balanceDiffOrg'] = input_df['oldbalanceOrg'] - input_df['newbalanceOrig']
    input_df['balanceDiffDest'] = input_df['newbalanceDest'] - input_df['oldbalanceDest']

    # One-hot encode transaction type
    input_df['type_CASH_OUT'] = (input_df['type'] == 'CASH_OUT').astype(int)
    input_df['type_DEBIT'] = (input_df['type'] == 'DEBIT').astype(int)
    input_df['type_PAYMENT'] = (input_df['type'] == 'PAYMENT').astype(int)
    input_df['type_TRANSFER'] = (input_df['type'] == 'TRANSFER').astype(int)
    input_df.drop(columns=['type'], inplace=True)

    # Ensure column order
    feature_order = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                     'newbalanceDest', 'balanceDiffOrg', 'balanceDiffDest',
                     'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    input_df = input_df[feature_order]

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This transaction is predicted to be **FRAUDULENT** (probability: {pred_proba:.2%})")
    else:
        st.success(f"✅ This transaction is predicted to be **GENUINE** (probability: {1 - pred_proba:.2%})")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    st.markdown("#### 🔎 SHAP Explanation")

    shap.initjs()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][0], feature_names=input_df.columns)
    st.pyplot(bbox_inches='tight')
