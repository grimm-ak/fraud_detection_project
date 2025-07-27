import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Load model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Required features (used in the model)
required_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
                     'errorBalanceOrig', 'errorBalanceDest', 'type_CASH_OUT', 'type_TRANSFER']

# Initialize SHAP
explainer = shap.TreeExplainer(model)

# App title
st.set_page_config(page_title="💳 Fraud Detector", layout="wide")
st.title("💳 Real-Time Credit Card Fraud Detection")

# Sidebar – collapsible info sections
with st.expander("ℹ️ Model Info", expanded=True):
    st.markdown("""
    **LightGBM classifier trained on 6M+ transactions**  
    • Uses derived balance features and one-hot encoded types  
    • Calibrated using GridSearchCV with class balancing  
    """)

with st.expander("🔒 Security Note", expanded=True):
    st.markdown("""
    **All predictions happen locally**  
    • No transaction data is stored or sent externally  
    """)

with st.expander("🎯 User Experience", expanded=True):
    st.markdown("""
    **Instant predictions with clear visual explanation**  
    • Waterfall SHAP bar plot shows *why* behind every decision  
    """)

# User input section
st.markdown("### 📝 Enter Transaction Details")

with st.form("input_form"):
    step = st.number_input("Step (Hour)", min_value=1, max_value=744, value=1)
    amount = st.number_input("Amount", min_value=0.0, value=100.0)
    oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, value=1000.0)
    newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, value=900.0)
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=0.0)
    type_input = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
    save_prediction = st.checkbox("💾 Save this prediction")

    submit = st.form_submit_button("🚀 Predict")

if submit:
    # Derived features
    errorBalanceOrig = oldbalanceOrg - newbalanceOrig - amount
    errorBalanceDest = newbalanceDest - oldbalanceDest - amount

    type_transfer = 1 if type_input == "TRANSFER" else 0
    type_cash_out = 1 if type_input == "CASH_OUT" else 0

    input_data = pd.DataFrame([{
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'errorBalanceOrig': errorBalanceOrig,
        'errorBalanceDest': errorBalanceDest,
        'type_CASH_OUT': type_cash_out,
        'type_TRANSFER': type_transfer
    }])[required_features]

    # Scale and predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0][1]

    # Output
    st.markdown("### 🧠 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {prediction_proba:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Confidence: {1 - prediction_proba:.2%})")

    # Save prediction if checked
    if save_prediction:
        pred_log = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **input_data.iloc[0].to_dict(),
            'prediction': int(prediction),
            'probability': float(prediction_proba)
        }
        log_df = pd.DataFrame([pred_log])
        log_exists = os.path.exists("prediction_logs.csv")
        log_df.to_csv("prediction_logs.csv", mode='a', header=not log_exists, index=False)
        st.info("📝 Prediction saved to `prediction_logs.csv`")

    # SHAP Explanation (safe version)
    st.markdown("### 🔍 SHAP Explanation")
    try:
        shap_values = explainer.shap_values(scaled_input)

        # Plot SHAP values with matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        shap_df = pd.DataFrame({
            'Feature': required_features,
            'SHAP Value': shap_values[1][0],
            'Value': scaled_input[0]
        })

        shap_df = shap_df.reindex(shap_df['SHAP Value'].abs().sort_values(ascending=False).index)
        colors = ['green' if val > 0 else 'red' for val in shap_df['SHAP Value']]
        ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
        ax.set_title("Feature Impact on Prediction", fontsize=14)
        ax.invert_yaxis()
        st.pyplot(fig)

    except Exception as e:
        st.info(f"⚠️ SHAP explanation could not be displayed: {e}")
