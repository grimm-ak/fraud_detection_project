import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Fraud Detection System", layout="centered")

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

# Define expected features (This should be here)
feature_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                   'newbalanceDest', 'balanceDiffOrg', 'balanceDiffDest',
                   'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

# --- NO st.write OR OTHER st.* CALLS BEFORE HERE ---
# (Remove the debug Checkpoint 1, 2, 3, 4 st.write lines if they are still there).
# (Remove the st.button("Predict Fraud") outside the if condition if it's there).

st.title("💸 Real-Time Fraud Detection System")
st.markdown("Enter transaction details to predict if it's fraudulent.")

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
    transaction_type = st.selectbox(
        "Transaction Type",
        ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
    )
    st.markdown("*(Other features V1-V28 are anonymized and not typically exposed as direct inputs. We use derived ones.)*")

if st.button("Predict Fraud"):
    # ... (full prediction and SHAP logic) ...
    pass # Placeholder for full logic