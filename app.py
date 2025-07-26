# app.py (Corrected Model Loading Section)

import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load your saved model and scaler
import shap # To display SHAP explanations

# --- Load the trained model and scaler ---
# Use st.cache_resource to cache the model and scaler.
# This decorator runs the function only once and stores the result in cache,
# and it properly handles the initialization of Streamlit's 'st' object.
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('best_lgbm_clf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        # We'll remove st.sidebar messages from here and put general success/error messages later if needed
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found. Please ensure they are in the correct directory.")
        st.stop() # Stop the app if files are missing
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.info("Please ensure 'best_lgbm_clf_model.joblib' and 'scaler.joblib' are valid joblib files.")
        st.stop()

# Call the cached function to load the model and scaler
model, scaler = load_model_and_scaler()

# --- Streamlit UI (This part is still at the top level, but doesn't cause problem) ---
st.set_page_config(page_title="Fraud Detection System", layout="centered")
st.title("💸 Real-Time Fraud Detection System")
st.markdown("Enter transaction details to predict if it's fraudulent.")

# ... (rest of your app.py code below this, including feature_columns, input fields, prediction logic)