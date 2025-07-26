import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load your saved model and scaler
import shap # To display SHAP explanations

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

# Define expected features (MUST match your training features in order and name)
feature_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                   'newbalanceDest', 'balanceDiffOrg', 'balanceDiffDest',
                   'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']


st.title("💸 Real-Time Fraud Detection System")
st.markdown("Enter transaction details to predict if it's fraudulent.")

st.header("Transaction Details")

col1, col2 = st.columns(2)
with col1:
    step = st.number_input("Step (Time step in hours)", min_value=1, value=1, key="input_step") # Added key
    amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f", key="input_amount") # Added key
    oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=10000.0, format="%.2f", key="input_oldbalanceorg") # Added key
    newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=9000.0, format="%.2f", key="input_newbalanceorig") # Added key
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=500.0, format="%.2f", key="input_oldbalancedest") # Added key
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=1500.0, format="%.2f", key="input_newbalancedest") # Added key

with col2:
    transaction_type = st.selectbox(
        "Transaction Type",
        ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'),
        key="input_transaction_type" # Added key
    )
    st.markdown("*(Other features V1-V28 are anonymized and not typically exposed as direct inputs. We use derived ones.)*")

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

    # 3. Apply One-Hot Encoding for 'type' (Corrected Logic)
    # Initialize all type_ columns to False first
    input_df['type_CASH_OUT'] = False
    input_df['type_DEBIT'] = False
    input_df['type_PAYMENT'] = False
    input_df['type_TRANSFER'] = False 

    # Set only the selected type to True
    if transaction_type == 'CASH_IN':
        pass # Do nothing for CASH_IN, as its dummy is the baseline
    elif transaction_type == 'CASH_OUT':
        input_df['type_CASH_OUT'] = True
    elif transaction_type == 'DEBIT':
        input_df['type_DEBIT'] = True
    elif transaction_type == 'PAYMENT':
        input_df['type_PAYMENT'] = True
    elif transaction_type == 'TRANSFER':
        input_df['type_TRANSFER'] = True

    # 4. Ensure column order and presence match training data's 'feature_columns'
    # This creates the DataFrame in the correct order for scaling and prediction
    input_data_processed = input_df[feature_columns]

    # 5. Scale the input data using the loaded scaler
    # scaler.transform expects a 2D array, which input_data_processed (DataFrame) is.
    # The output is a NumPy array, which is typical for model input.
    scaled_input_array = scaler.transform(input_data_processed)
    
    # --- CRITICAL FIX FOR SHAP: Pass a DataFrame with feature names to the model and SHAP ---
    # Convert the scaled NumPy array back to a DataFrame with feature names
    # This resolves the 'X does not have valid feature names' warning and potentially helps SHAP plotting.
    scaled_input_df_for_model = pd.DataFrame(scaled_input_array, columns=feature_columns)

    # 6. Make prediction
    prediction_proba = model.predict_proba(scaled_input_df_for_model)[:, 1][0] # Use DataFrame here
    prediction_label = model.predict(scaled_input_df_for_model)[0] # Use DataFrame here

    st.subheader("Prediction Result:")
    if prediction_label == 1:
        st.error(f"🔴 FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success(f"🟢 LEGITIMATE TRANSACTION.")
    
    st.write(f"**Fraud Probability:** {prediction_proba:.4f}")

    # --- SHAP Interpretation for the single prediction ---
    st.subheader("Why this prediction? (Feature Contributions)")
    
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values using the DataFrame with feature names
    shap_values_raw_single = explainer.shap_values(scaled_input_df_for_model) 
    
    if isinstance(shap_values_raw_single, list) and len(shap_values_raw_single) > 1:
        shap_values_for_plot_single = shap_values_raw_single[1] 
        expected_value_for_plot_single = explainer.expected_value[1] 
    else:
        shap_values_for_plot_single = shap_values_raw_single
        expected_value_for_plot_single = explainer.expected_value
    
    # Ensure shap_values_for_plot_single is 1D for force_plot
    if shap_values_for_plot_single.ndim > 1:
        shap_values_for_plot_single = shap_values_for_plot_single[0] 

    # For force_plot, the base_values (scaled_input) should ideally have feature names as well.
    # The single_input_df already has names and is based on scaled_input_array.
    # We already have scaled_input_df_for_model which is perfect for this.
    
    st.write("The plot below shows how each feature pushed the prediction (blue for negative impact, red for positive impact).")
    
    html_plot = shap.force_plot(expected_value_for_plot_single, shap_values_for_plot_single, scaled_input_df_for_model, plot_cmap='RdBu').html() # Use scaled_input_df_for_model here
    st.components.v1.html(html_plot, height=300, scrolling=True)

    st.info("💡 A higher red bar means that feature value pushed the prediction towards FRAUD. A higher blue bar means it pushed it towards LEGITIMATE.")