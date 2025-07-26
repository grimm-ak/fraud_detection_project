# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load your saved model and scaler
import shap # To display SHAP explanations

# --- Load the trained model and scaler ---
# These files must be in the same directory as app.py (or an accessible path)
try:
    model = joblib.load('best_lgbm_clf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    st.sidebar.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model or scaler: {e}")
    st.sidebar.info("Please ensure 'best_lgbm_clf_model.joblib' and 'scaler.joblib' are in the same directory as app.py.")
    st.stop() # Stop the app if model/scaler can't be loaded

# --- Define expected features (MUST match your training features in order and name) ---
# Replace with your actual 12 features in the correct order.
# You can get this list by running print(X.columns.tolist()) in your notebook after features selection.
feature_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                   'newbalanceDest', 'balanceDiffOrg', 'balanceDiffDest',
                   'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']


# --- Streamlit UI ---
st.set_page_config(page_title="Fraud Detection System", layout="centered")
st.title("💸 Real-Time Fraud Detection System")
st.markdown("Enter transaction details to predict if it's fraudulent.")

# --- Input Fields ---
st.header("Transaction Details")

col1, col2 = st.columns(2)
with col1:
    step = st.number_input("Step (Time step in hours)", min_value=1, value=1)
    amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f")
    oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=10000.0, format="%.2f")
    newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=9000.0, format="%.2f")
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=500.0, format="%.2f")
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=1500.0, format="%.2f")

with col2:
    # Transaction Type Selection
    transaction_type = st.selectbox(
        "Transaction Type",
        ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
    )
    st.markdown("*(Other features V1-V28 are anonymized and not typically exposed as direct inputs. We use derived ones.)*")


# --- Preprocess Input & Make Prediction ---
if st.button("Predict Fraud"):
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

    # 3. Apply One-Hot Encoding for 'type' 

    input_df['type_CASH_OUT'] = False
    input_df['type_DEBIT'] = False
    input_df['type_PAYMENT'] = False
    input_df['type_TRANSFER'] = True 

    # Set the selected type to True
    if transaction_type == 'CASH_OUT':
        input_df['type_CASH_OUT'] = True
    elif transaction_type == 'DEBIT':
        input_df['type_DEBIT'] = True
    elif transaction_type == 'PAYMENT':
        input_df['type_PAYMENT'] = True
    elif transaction_type == 'TRANSFER':
        input_df['type_TRANSFER'] = True
    # If CASH_IN, all others remain False (as CASH_IN was drop_first=True)

    # 4. Ensure column order and presence match training data's 'feature_columns'

    # The simplest way is to reindex.
    input_data_processed = input_df[feature_columns]

    # 5. Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_data_processed)