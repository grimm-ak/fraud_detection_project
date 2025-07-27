import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap

# Load the trained LightGBM model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Define all feature names expected by the model (replace with your actual list)
feature_names = [
    'feature1', 'feature2', 'feature3', 'feature4', 'feature5',
    # Add all your actual feature columns used during training here
]

# Streamlit app layout
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("🔍 Fraud Detection App")
st.markdown("Provide the transaction details below to predict fraud.")

# UI inputs
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Predict button
if st.button("Predict Fraud"):
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input
    input_scaled = scaler.transform(input_df)
    
    # Prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    # Display result
    st.markdown(f"### 🧾 Prediction: {'🚨 Fraudulent' if prediction == 1 else '✅ Not Fraudulent'}")
    st.markdown(f"**Probability of Fraud:** `{prediction_proba:.2f}`")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)

    st.markdown("#### 🔎 SHAP Explanation")
    fig = shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[1],
        shap_values[1][0],
        feature_names=feature_names,
        features=input_scaled[0]
    )
    st.pyplot(bbox_inches="tight")
