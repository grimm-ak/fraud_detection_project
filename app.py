import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lightgbm

# Set page config
st.set_page_config(page_title="Fraud Detection App", layout="centered")

# Load model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Title
st.title("🚨 Fraud Detection - Streamlit App")
st.markdown("Upload transaction data or try our sample dataset to detect fraud.")

# Load sample data
@st.cache_data
def load_sample_data():
    df = pd.read_csv("sample_test_data.csv")
    return df

sample_df = load_sample_data()

# Show sample preview
st.subheader("🔍 Sample Data Preview")
st.dataframe(sample_df.head(10))

# --- Evaluation on Sample Data ---
if st.button("📊 Run Evaluation on Sample"):
    try:
        X_sample = sample_df.drop("is_fraud", axis=1)
        y_sample = sample_df["is_fraud"]
        X_scaled = scaler.transform(X_sample)
        y_pred = model.predict(X_scaled)

        results_df = pd.DataFrame({
            "Actual": y_sample,
            "Predicted": y_pred.astype(int)
        })

        st.success("✅ Prediction Complete")
        st.dataframe(results_df.head(10))

        # SHAP explainability for first sample
        st.subheader("📈 SHAP Explanation (1st row)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled[:1])

        st.set_option("deprecation.showPyplotGlobalUse", False)
        shap.initjs()
        st.pyplot(shap.force_plot(explainer.expected_value[1], shap_values[1], X_sample.iloc[0], matplotlib=True))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# --- Optional Upload ---
st.subheader("📂 Or Upload Your Own CSV File")

uploaded_file = st.file_uploader("Upload transaction data CSV", type=["csv"])

if uploaded_file:
    try:
        user_df = pd.read_csv(uploaded_file)
        st.dataframe(user_df.head(10))

        if "is_fraud" in user_df.columns:
            X_user = user_df.drop("is_fraud", axis=1)
        else:
            X_user = user_df

        X_scaled = scaler.transform(X_user)
        y_pred = model.predict(X_scaled)

        st.subheader("✅ Predictions on Uploaded Data")
        st.write(pd.DataFrame({"Predicted": y_pred.astype(int)}).head(10))

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
