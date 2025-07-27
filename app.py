import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('best_lgbm_clf_model.joblib')
scaler = joblib.load('scaler.joblib')

# Feature columns used during training
feature_names = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

st.set_page_config(layout="centered")
st.title("🚨 Fraud Detection App")
st.write("Enter the transaction details below:")

# Collect user inputs
step = st.number_input("Step (Hour)", min_value=1, max_value=744, value=1)
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=5000.0)
newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=4000.0)
oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=1000.0)
newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=2000.0)

tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"])

if st.button("Predict Fraud"):
    # Build input dataframe
    input_data = {
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'type': tx_type
    }

    input_df = pd.DataFrame([input_data])

    # One-hot encode 'type'
    type_dummies = pd.get_dummies(input_df['type'], prefix='type')
    input_df = pd.concat([input_df.drop('type', axis=1), type_dummies], axis=1)

    # Ensure all expected dummy columns exist
    for col in ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training features
    input_df = input_df[feature_names]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("### Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability of Fraud: {prediction_prob:.2f})")

    # SHAP Explanation
    st.markdown("#### 🔎 SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    shap.initjs()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[1], shap_values[1][0], feature_names=feature_names
    )
    st.pyplot(bbox_inches='tight')
