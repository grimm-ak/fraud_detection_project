import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Streamlit Config ---
st.set_page_config(page_title="💸 Fraud Detection", layout="centered")

# --- Load Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('best_lgbm_clf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_model_and_scaler()

@st.cache_resource
def get_shap_explainer(_model):
    return shap.Explainer(_model)

explainer = get_shap_explainer(model)

feature_columns = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiffOrg', 'balanceDiffDest',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# --- App Title ---
st.title("💸 Real-Time Fraud Detection System")

# --- Collapsible Info Section ---
with st.expander("ℹ️ Show Model Info / Security Notes"):
    st.markdown("""
    ### 🔍 Model Info  
    - **LightGBM classifier** trained on 6M+ transactions  
    - Includes derived balance features + one-hot encoded types  
    - Calibrated with **GridSearchCV** and **class balancing**

    ### 🔐 Privacy & Security  
    - All processing is **local** to this session  
    - **No data is stored or sent externally**

    ### 🧠 UX Features  
    - Fast predictions + visual reasoning  
    - **SHAP waterfall** explains key features influencing the decision  
    """)

# --- Presets ---
presets = {
    
    "🏧 High-Value Cash Out (Suspicious)": {
        'step': 120,
        'amount': 980000.00,
        'oldbalanceOrg': 1000000.0,
        'newbalanceOrig': 20000.0,
        'oldbalanceDest': 0.0,
        'newbalanceDest': 980000.0,
        'transaction_type': 'CASH_OUT'
    },
    "📥 Typical Payment": {
        'step': 320,
        'amount': 120.00,
        'oldbalanceOrg': 2000.0,
        'newbalanceOrig': 1880.0,
        'oldbalanceDest': 1000.0,
        'newbalanceDest': 1120.0,
        'transaction_type': 'PAYMENT'
    }
}

# --- Tabs Layout ---
tab1, tab2 = st.tabs(["🚨 Predict Fraud", "📈 Feature Impact Stats"])

with tab1:
    st.header("📝 Enter Transaction Details")

    preset_choice = st.selectbox("📦 Choose a Preset Transaction or Enter Manually", list(presets.keys()))
    preset = presets[preset_choice]

    col1, col2 = st.columns(2)
    with col1:
        step = st.number_input("Step (hour)", min_value=1, value=preset.get("step", 1))
        amount = st.number_input("Amount", min_value=0.0, value=preset.get("amount", 1000.0), format="%.2f")
        oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=preset.get("oldbalanceOrg", 10000.0), format="%.2f")
        newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=preset.get("newbalanceOrig", 9000.0), format="%.2f")
        oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=preset.get("oldbalanceDest", 500.0), format="%.2f")
        newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=preset.get("newbalanceDest", 1500.0), format="%.2f")

    with col2:
        transaction_type = st.selectbox("Transaction Type", ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'],
                                        index=['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'].index(preset.get("transaction_type", "CASH_IN")))
        st.caption("*(Derived features used instead of anonymized V1–V28)*")

    st.markdown("---")

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

        # --- Scale Input ---
        scaled_input = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_input, columns=feature_columns)

        # --- Prediction ---
        prediction = model.predict(scaled_df)[0]
        proba = model.predict_proba(scaled_df)[0][1]

        confidence = proba if prediction == 1 else (1 - proba)
        confidence_label = (
            "🟢 High Confidence" if confidence >= 0.90 else
            "🟡 Medium Confidence" if confidence >= 0.70 else
            "🔴 Low Confidence"
        )

        # --- Result ---
        st.subheader("📢 Prediction Result")
        if prediction == 1:
            st.error("🔴 **FRAUDULENT TRANSACTION DETECTED!**")
        else:
            st.success("🟢 **LEGITIMATE TRANSACTION.**")

        st.markdown(f"**🧠 Model Confidence:** {confidence_label} (`{confidence * 100:.2f}%`)")

        if "Low" in confidence_label:
            st.warning("⚠️ The model is unsure. Please verify this result manually.")

        # --- SHAP Explanation ---
        with st.expander("🔍 Why this prediction? (SHAP Waterfall)", expanded=True):
            shap_values = explainer(scaled_df)

            plt.clf()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(plt.gcf())
            st.caption("🔹 Blue pushes toward Legit | 🔺 Red pushes toward Fraud")

            st.session_state['last_shap'] = shap_values[0]
            st.session_state['last_features'] = scaled_df.iloc[0]

with tab2:
    st.header("📊 Feature Impact Summary")

    if 'last_shap' in st.session_state:
        shap_values = st.session_state['last_shap'].values
        features = st.session_state['last_features']

        feature_impact_df = pd.DataFrame({
            'Feature': features.index,
            'Value': features.values,
            'SHAP Impact': shap_values
        }).sort_values(by='SHAP Impact', key=abs, ascending=False)

        st.markdown("#### 🔍 Top 5 Influential Features")
        st.dataframe(feature_impact_df.head(5), use_container_width=True)

        most_positive = feature_impact_df.loc[feature_impact_df['SHAP Impact'].idxmax()]
        most_negative = feature_impact_df.loc[feature_impact_df['SHAP Impact'].idxmin()]

        st.markdown(f"🟥 **Most Fraud-Inducing:** `{most_positive['Feature']}` → SHAP = `{most_positive['SHAP Impact']:.3f}`")
        st.markdown(f"🟦 **Most Fraud-Reducing:** `{most_negative['Feature']}` → SHAP = `{most_negative['SHAP Impact']:.3f}`")
    else:
        st.info("⚠️ Run a prediction to view feature impact insights.")
