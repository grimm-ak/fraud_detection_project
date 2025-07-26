# app.py (Uncommented Prediction Logic)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap # Keep this import

# ... (Previous code for loading model, scaler, feature_columns, UI setup) ...

# --- Preprocess Input & Make Prediction ---
if st.button("Predict Fraud"): # <-- UNCOMMENT THIS LINE (and its corresponding tab)
    # 1. Create a DataFrame from inputs
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    input_df['step'] = step
    input_df['amount'] = amount
    input_df['oldbalanceOrg'] = oldbalanceOrg
    input_df['newbalanceOrig'] = newbalanceOrig
    input_df['oldbalanceDest'] = oldbalanceDest
    input_df['newbalanceDest'] = newbalanceDest
    
    # 2. Re-create engineered features
    input_df['balanceDiffOrg'] = input_df['oldbalanceOrg'] - input_df['newbalanceOrig']
    input_df['balanceDiffDest'] = input_df['newbalanceDest'] - input_df['oldbalanceDest']

    # 3. Apply One-Hot Encoding for 'type' (Corrected Logic)
    input_df['type_CASH_OUT'] = False
    input_df['type_DEBIT'] = False
    input_df['type_PAYMENT'] = False
    input_df['type_TRANSFER'] = False 

    if transaction_type == 'CASH_IN':
        pass # Do nothing for CASH_IN
    elif transaction_type == 'CASH_OUT':
        input_df['type_CASH_OUT'] = True
    elif transaction_type == 'DEBIT':
        input_df['type_DEBIT'] = True
    elif transaction_type == 'PAYMENT':
        input_df['type_PAYMENT'] = True
    elif transaction_type == 'TRANSFER':
        input_df['type_TRANSFER'] = True

    # 4. Ensure column order and presence match training data's 'feature_columns'
    input_data_processed = input_df[feature_columns]

    # 5. Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_data_processed)
    
    # 6. Make prediction
    prediction_proba = model.predict_proba(scaled_input)[:, 1][0]
    prediction_label = model.predict(scaled_input)[0]

    st.subheader("Prediction Result:")
    if prediction_label == 1:
        st.error(f"🔴 FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success(f"🟢 LEGITIMATE TRANSACTION.")
    
    st.write(f"**Fraud Probability:** {prediction_proba:.4f}")

    # --- Commented out SHAP Interpretation for now ---
    # st.subheader("Why this prediction? (Feature Contributions)")
    # explainer = shap.TreeExplainer(model)
    # shap_values_raw_single = explainer.shap_values(scaled_input)
    # ... (rest of SHAP code) ...