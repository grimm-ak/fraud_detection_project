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
with st.expander("ℹ️ Click to Show/Hide Model Info, Security & UX"):
    st.markdown("""
    ### 📌 Model Info  
    - LightGBM classifier trained on 6M+ transactions  
    - Uses derived balance features and one-hot encoded types  
    - Calibrated using GridSearchCV with class balancing  

    ### 🔐 Security Note  
    - All predictions happen locally  
    - No transaction data is stored or sent externally  

    ### 🎯 User Experience  
    - Instant predictions with clear visual explanation  
    - SHAP waterfall plot shows **why** the model made its decision  
    """)

# --- Tabs Layout ---
tab1, tab2 = st.tabs(["🔍 Predict Fraud", "📈 Feature Impact Stats"])

with tab1:
    st.header("📝 Enter Transaction Details")

    col1, col2 = st.columns(2)
    with col1:
        step = st.number_input("Step (hour)", min_value=1, value=1)
        amount = st.number_input("Amount", min_value=0.0, value=1000.0, format="%.2f")
        oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=10000.0, format="%.2f")
        newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=9000.0, format="%.2f")
        oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=500.0, format="%.2f")
        newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=1500.0, format="%.2f")

    with col2:
        transaction_type = st.selectbox("Transaction Type", ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
        st.markdown("*(Derived features used instead of V1–V28 anonymized ones)*")

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

        # --- Result Section ---
        st.subheader("📢 Prediction Result")
    if prediction == 1:
        st.markdown("## 📢 Prediction Result")
        st.error("🔴 **Warning: FRAUDULENT TRANSACTION Detected!**")
        st.markdown(
            f"""
            🧠 **Model Confidence:** `{proba:.2%}`

            ⚠️ Our model believes with high confidence that this transaction is **likely to be fraudulent**.

            Immediate review or action is recommended.
            """
        )
    else:
        st.markdown("## 📢 Prediction Result")
        st.success("🟢 **This transaction appears to be LEGITIMATE.**")
        st.markdown(
            f"""
            🧠 **Model Confidence:** `{(1 - proba):.2%}`

            ✅ Based on the transaction details you provided, our model is very confident that this is a **normal, safe transaction**.

            No suspicious patterns were detected.
            """
        )

        # --- SHAP Waterfall Plot ---
        with st.expander("🔍 Why this prediction? (SHAP Waterfall)", expanded=True):
            shap_values = explainer(scaled_df)

            plt.clf()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(plt.gcf())
            st.caption("Red pushes toward fraud | Blue pushes toward legit")

            # Store for Tab 2
            st.session_state['last_shap'] = shap_values[0]
            st.session_state['last_features'] = scaled_df.iloc[0]

with tab2:
    st.header("📈 Feature Impact on Prediction")

    if 'last_shap' in st.session_state:
        shap_values = st.session_state['last_shap'].values
        features = st.session_state['last_features']

        feature_impact_df = pd.DataFrame({
            'Feature': features.index,
            'Value': features.values,
            'SHAP Impact': shap_values
        }).sort_values(by='SHAP Impact', key=abs, ascending=False)

        st.markdown("#### 🔍 Top 5 Key Features")
        st.dataframe(feature_impact_df.head(5), use_container_width=True)

        most_positive = feature_impact_df.loc[feature_impact_df['SHAP Impact'].idxmax()]
        most_negative = feature_impact_df.loc[feature_impact_df['SHAP Impact'].idxmin()]

        st.markdown(f"🟥 **Most Fraud-Inducing:** `{most_positive['Feature']}` → SHAP = `{most_positive['SHAP Impact']:.3f}`")
        st.markdown(f"🟦 **Most Fraud-Reducing:** `{most_negative['Feature']}` → SHAP = `{most_negative['SHAP Impact']:.3f}`")
    else:
        st.info("⚠️ Make a prediction first to view feature impact.")
