import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load your saved model and scaler
import shap # To display SHAP explanations
import matplotlib.pyplot as plt # For SHAP plotting

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Fraud Detection System", layout="centered")

# --- Model Performance Metrics (FILL THESE IN FROM YOUR TRAINING NOTEBOOK) ---
MODEL_PERFORMANCE_METRICS = {
    "ROC AUC": 0.9999,  # Example: Fill with your model's test ROC AUC
    "Precision": 0.9920, # Example: Fill with your model's test Precision
    "Recall": 0.9973,   # Example: Fill with your model's test Recall
    "F1-Score": 0.9946  # Example: Fill with your model's test F1-Score
}

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

# --- Define expected features (MUST match your training features in order and name) ---
feature_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                   'newbalanceDest', 'balanceDiffOrg', 'balanceDiffDest',
                   'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

# --- Streamlit UI: Main Content ---
st.title("💸 Real-Time Fraud Detection System")
st.markdown("Enter transaction details to predict if it's fraudulent.")
st.header("Transaction Details")

# --- Input Fields with Session State for Reset ---
# Initialize session state for inputs if not already present
if 'step_val' not in st.session_state:
    st.session_state.step_val = 1
if 'amount_val' not in st.session_state:
    st.session_state.amount_val = 1000.0
if 'obo_val' not in st.session_state:
    st.session_state.obo_val = 10000.0
if 'nbo_val' not in st.session_state:
    st.session_state.nbo_val = 9000.0
if 'obd_val' not in st.session_state:
    st.session_state.obd_val = 500.0
if 'nbd_val' not in st.session_state:
    st.session_state.nbd_val = 1500.0
if 'type_idx' not in st.session_state:
    st.session_state.type_idx = 0 # Index of 'CASH_IN'

col1, col2 = st.columns(2)
with col1:
    step = st.number_input("Step (hour)", min_value=1, key="step_input", value=st.session_state.step_val)
    amount = st.number_input("Amount", min_value=0.0, format="%.2f", key="amount_input", value=st.session_state.amount_val)
    oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, format="%.2f", key="obo_input", value=st.session_state.obo_val)
    newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, format="%.2f", key="nbo_input", value=st.session_state.nbo_val)
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, format="%.2f", key="obd_input", value=st.session_state.obd_val)
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, format="%.2f", key="nbd_input", value=st.session_state.nbd_val)

with col2:
    transaction_options = ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER')
    transaction_type = st.selectbox(
        "Transaction Type",
        transaction_options,
        key="type_input",
        index=st.session_state.type_idx
    )
    st.markdown("*(Note: We use derived features instead of anonymized V1–V28.)*")

# --- Reset Button Functionality ---
def reset_inputs():
    st.session_state.step_val = 1
    st.session_state.amount_val = 1000.0
    st.session_state.obo_val = 10000.0
    st.session_state.nbo_val = 9000.0
    st.session_state.obd_val = 500.0
    st.session_state.nbd_val = 1500.0
    st.session_state.type_idx = 0 # Reset to 'CASH_IN'
    st.rerun() # <--- FIXED: Changed from st.experimental_rerun() to st.rerun()

if st.button("Reset Inputs", on_click=reset_inputs):
    pass # Function handles the reset

# --- Predict Button ---
if st.button("Predict Fraud", key="predict_button"):
    # 1. Create input dataframe
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    input_df['step'] = step
    input_df['amount'] = amount
    input_df['oldbalanceOrg'] = oldbalanceOrg
    input_df['newbalanceOrig'] = newbalanceOrig
    input_df['oldbalanceDest'] = oldbalanceDest
    input_df['newbalanceDest'] = newbalanceDest
    
    # 2. Re-create engineered features
    input_df['balanceDiffOrg'] = oldbalanceOrg - newbalanceOrig
    input_df['balanceDiffDest'] = newbalanceDest - oldbalanceDest

    # 3. One-hot encode transaction type
    input_df['type_CASH_OUT'] = False
    input_df['type_DEBIT'] = False
    input_df['type_PAYMENT'] = False
    input_df['type_TRANSFER'] = False 

    if transaction_type == 'CASH_IN':
        pass
    elif transaction_type == 'CASH_OUT':
        input_df['type_CASH_OUT'] = True
    elif transaction_type == 'DEBIT':
        input_df['type_DEBIT'] = True
    elif transaction_type == 'PAYMENT':
        input_df['type_PAYMENT'] = True
    elif transaction_type == 'TRANSFER':
        input_df['type_TRANSFER'] = True

    # 4. Match training feature order
    input_data_processed = input_df[feature_columns]

    # 5. Scale input
    scaled_array = scaler.transform(input_data_processed)
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns)

    # 6. Predict
    prediction_proba = model.predict_proba(scaled_df)[:, 1][0]
    prediction_label = model.predict(scaled_df)[0]

    st.subheader("Prediction Result:")
    if prediction_label == 1:
        st.error(f"🔴 FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success(f"🟢 LEGITIMATE TRANSACTION.")

    st.write(f"**Fraud Probability:** `{prediction_proba:.4f}`")

    # --- Model Confidence Visual ---
    st.markdown("### Model Confidence:")
    confidence_level = prediction_proba * 100
    
    st.progress(int(confidence_level))
    
    if prediction_label == 1:
        st.write(f"Confidence in Fraud Detection: **{confidence_level:.2f}%**")
        st.markdown(f'<p style="color:red;font-weight:bold;">This prediction is likely to be FRAUD.</p>', unsafe_allow_html=True)
    else:
        st.write(f"Confidence in Legitimate Transaction: **{100 - confidence_level:.2f}%**")
        st.markdown(f'<p style="color:green;font-weight:bold;">This prediction is likely to be LEGITIMATE.</p>', unsafe_allow_html=True)


    # --- SHAP Explanation ---
    st.subheader("Why this prediction? (Feature Contributions)")
    
    explainer = shap.Explainer(model)
    
    shap_values_raw_single = explainer(scaled_df) # Use explainer(data) for newer SHAP
    if isinstance(shap_values_raw_single, list) and len(shap_values_raw_single) > 1:
        shap_values_for_plot_single = shap_values_raw_single[1]
        expected_value_for_plot_single = explainer.expected_value[1]
    else:
        shap_values_for_plot_single = shap_values_raw_single
        expected_value_for_plot_single = explainer.expected_value
    
    if shap_values_for_plot_single.ndim > 1:
        shap_values_for_plot_single = shap_values_for_plot_single[0]

    st.write("The plot below shows how each feature pushed the prediction (blue for negative impact, red for positive impact).")
    
    html_plot = shap.force_plot(expected_value_for_plot_single, shap_values_for_plot_single, scaled_df, plot_cmap='RdBu').html()
    st.components.v1.html(html_plot, height=300, scrolling=True)

    st.info("💡 A higher red bar means that feature value pushed the prediction towards FRAUD. A higher blue bar means it pushed it towards LEGITIMATE.")


# --- Streamlit UI: Sidebar for Model Info ---
st.sidebar.header("Model Performance Metrics (Test Set)")
st.sidebar.markdown(
    f"""
    - **ROC AUC:** `{MODEL_PERFORMANCE_METRICS['ROC AUC']:.4f}`
    - **Precision:** `{MODEL_PERFORMANCE_METRICS['Precision']:.4f}`
    - **Recall:** `{MODEL_PERFORMANCE_METRICS['Recall']:.4f}`
    - **F1-Score:** `{MODEL_PERFORMANCE_METRICS['F1-Score']:.4f}`
    """
)
st.sidebar.markdown(
    """
    *These metrics reflect the model's performance on a balanced test set during training.*
    """
)

# --- Security Note Disclaimer ---
st.markdown("---") # Separator
st.caption("""
**Disclaimer:** This is a demonstration for educational and portfolio purposes only. 
Real-world fraud detection systems are vastly more complex, involving extensive domain expertise, 
sophisticated rule engines, anomaly detection, graph analytics, continuous monitoring, and robust security measures. 
This demo should NOT be used for actual financial decisions.
""")