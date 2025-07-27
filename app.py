import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Streamlit Page Config ---
st.set_page_config(page_title="💸 Fraud Detection System", layout="centered")

# --- Load Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('best_lgbm_clf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model or scaler file not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# --- SHAP Explainer ---
@st.cache_resource
def get_shap_explainer(_model):
    return shap.Explainer(_model)

explainer = get_shap_explainer(model)

# --- Features as per training ---
feature_columns = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiffOrg', 'balanceDiffDest',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# --- App Title & Info Cards ---
st.title("💸 Real-Time Fraud Detection System")

# --- Info Sections ---
st.markdown("""
### ℹ️ Model Info  
- LightGBM classifier trained on 6M+ transactions  
- Uses derived balance features and one-hot encoded types  
- Calibrated using GridSearchCV with class balancing  

### 🔒 Security Note  
- All predictions happen locally  
- No transaction data is stored or sent externally  

### 🎯 User Experience  
- Instant predictions with clear visual explanation  
- Waterfall SHAP plot shows the **why** behind every decision  
""")

# --- Tabs for Input and Output ---
tab1, tab2 = st.tabs(["🔍 Predict Fraud", "📊 SHAP Summary"])

with tab1:
    st.header("📝 Enter Transaction Details")

    col1, col2 = st.columns(2)
    with col1:
        step = st.number_input("Step (hour)", min_value=1, value=1, key="step")
        amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f", key="amount")
        oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=10000.0, format="%.2f", key="obo")
        newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=9000.0, format="%.2f", key="nbo")
        oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=500.0, format="%.2f", key="obd")
        newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=1500.0, format="%.2f", key="nbd")

    with col2:
        transaction_type = st.selectbox(
            "Transaction Type",
            ('CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'),
            key="type"
        )
        st.markdown("*(Note: Using derived features instead of anonymized V1–V28.)*")

    if st.button("🚨 Predict Fraud", use_container_width=True):
        # --- Prepare Input ---
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        input_df['step'] = step
        input_df['amount'] = amount
        input_df['oldbalanceOrg'] = oldbalanceOrg
        input_df['newbalanceOrig'] = newbalanceOrig
        input_df['oldbalanceDest'] = oldbalanceDest
        input_df['newbalanceDest'] = newbalanceDest
        input_df['balanceDiffOrg'] = oldbalanceOrg - newbalanceOrig
        input_df['balanceDiffDest'] = newbalanceDest - oldbalanceDest

        input_df['type_CASH_OUT'] = transaction_type == 'CASH_OUT'
        input_df['type_DEBIT'] = transaction_type == 'DEBIT'
        input_df['type_PAYMENT'] = transaction_type == 'PAYMENT'
        input_df['type_TRANSFER'] = transaction_type == 'TRANSFER'
        # CASH_IN is default (baseline)

        # --- Process Input ---
        input_processed = input_df[feature_columns]
        scaled_array = scaler.transform(input_processed)
        scaled_df = pd.DataFrame(scaled_array, columns=feature_columns)

        # --- Predict ---
        prediction_proba = model.predict_proba(scaled_df)[:, 1][0]
        prediction_label = model.predict(scaled_df)[0]

        st.subheader("📢 Prediction Result")
        if prediction_label == 1:
            st.error("🔴 FRAUDULENT TRANSACTION DETECTED!")
        else:
            st.success("🟢 LEGITIMATE TRANSACTION.")

        st.write(f"**Fraud Probability:** `{prediction_proba:.4f}`")

        # --- SHAP Explanation: Waterfall Plot ---
        st.subheader("🔍 Why this prediction?")
        shap_values = explainer(scaled_df)

        plt.clf()
        shap.plots.waterfall(shap_values[0], show=False)
        fig = plt.gcf()
        st.pyplot(fig)

        st.info("💡 Red features push toward fraud; blue toward legitimate.")

with tab2:
    st.header("📊 SHAP Feature Importance Summary")
    # Use dummy input to show summary plot of all features
    try:
        background_data = pd.DataFrame(np.random.rand(100, len(feature_columns)), columns=feature_columns)
        background_data = pd.DataFrame(scaler.transform(background_data), columns=feature_columns)
        shap_vals = explainer(background_data)

        st.markdown("Top features contributing to fraud detection decisions:")
        plt.clf()
        shap.plots.bar(shap_vals, show=False)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"Couldn't load summary plot: {e}")
