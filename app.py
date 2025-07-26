# app.py (Corrected st.set_page_config placement)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Fraud Detection System", layout="centered") # <--- MOVED HERE

# --- Load the trained model and scaler ---
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('best_lgbm_clf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found. Please ensure they are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.info("Please ensure 'best_lgbm_clf_model.joblib' and 'scaler.joblib' are valid joblib files.")
        st.stop()

model, scaler = load_model_and_scaler()

# --- Debug Checkpoint 1 (Now after config and load) ---
st.write("Checkpoint 1: Model and Scaler loaded. UI should appear.")

# --- Streamlit UI (Rest of the main UI, now comes after config and load) ---
st.title("💸 Real-Time Fraud Detection System")
st.markdown("Enter transaction details to predict if it's fraudulent.")

# --- Debug Checkpoint 2 ---
st.write("Checkpoint 2: Title and markdown rendered. Input header should appear.")

st.header("Transaction Details")

# --- Debug Checkpoint 3 ---
st.write("Checkpoint 3: Transaction Details header rendered. Columns should appear.")

col1, col2 = st.columns(2)
with col1:
    step = st.number_input("Step (Time step in hours)", min_value=1, value=1)
    amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f")
    oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=10000.0, format="%.2f")
    newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=9000.0, format="%.2f")
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=500.0, format="%.2f")
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=1500.0, format="%.2f")

with col2:
    transaction_type = st.selectbox(
        "Transaction Type",
        ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
    )
    st.markdown("*(Other features V1-V28 are anonymized and not typically exposed as direct inputs. We use derived ones.)*")

# --- Debug Checkpoint 4 ---
st.write("Checkpoint 4: All input fields rendered. Predict button should appear.")

st.button("Predict Fraud") # Keep this line for the button, but remove the 'if' condition initially

# --- Prediction and SHAP logic remains commented out for this test ---
# if st.button("Predict Fraud"):
#     # ... (prediction and SHAP logic) ...