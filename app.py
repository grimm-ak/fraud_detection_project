import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load saved model and scaler
model = joblib.load('best_lgbm_clf_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load required feature list
required_features = joblib.load('required_features.joblib')

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("🚨 Fraud Detection App")

# User Experience Note
st.info("🔐 This app uses a trained machine learning model to predict fraud based on financial transaction inputs. All predictions are processed locally and securely.")

st.markdown("### 🔧 Enter transaction details:")

# Input form
step = st.number_input("Step", min_value=1, max_value=100000, value=1)
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, value=5000.0)
newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, value=4000.0)

if st.button("Predict Fraud"):
    # Build input in correct order with default 0s, then replace filled fields
    input_df = pd.DataFrame(0, index=[0], columns=required_features)
    input_df['step'] = step
    input_df['amount'] = amount
    input_df['oldbalanceOrg'] = oldbalanceOrg
    input_df['newbalanceOrig'] = newbalanceOrig

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.error(f"⚠️ The transaction is predicted to be **FRAUDULENT**.")
    else:
        st.success(f"✅ The transaction is predicted to be **LEGITIMATE**.")

    st.markdown(f"**Model Confidence:** `{round(probability*100, 2)}%` that this is fraud.")

    # SHAP explanation (optional)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        st.markdown("#### 🔎 SHAP Explanation")
        shap.initjs()
        fig = shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][0], feature_names=required_features, show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation could not be displayed: {e}")

# Model Info Section
with st.expander("📦 Model Information"):
    st.markdown("""
    - **Model**: LightGBM Classifier  
    - **Training Data**: ~6 million transactions  
    - **Features**: Custom engineered (step, amount, balances, etc.)  
    - **Balanced**: Using class weights for fraud detection  
    - **Interpretability**: SHAP-based  
    """)

# Security Note
with st.expander("🔐 Security Note"):
    st.markdown("""
    - This app does **not collect or store** any personal data.  
    - Model is loaded and run locally via Streamlit.  
    - Predictions are for demo/educational purposes only.  
    """)
