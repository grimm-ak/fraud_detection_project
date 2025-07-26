import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Fraud Detection App", layout="centered")

# Load model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Load sample data
@st.cache_data
def load_sample_data():
    return pd.read_csv("sample_test_data.csv")

sample_df = load_sample_data()

# Title
st.title("💳 Fraud Detection System")
st.write("Select a transaction to check if it's fraudulent and explain why.")

# Dropdown for row selection
selected_index = st.selectbox("Select Transaction Index", range(len(sample_df)))

# Get selected row
selected_row = sample_df.iloc[[selected_index]]
st.write("### 🔍 Selected Transaction Preview")
st.dataframe(selected_row)

# Scale it
X_scaled = scaler.transform(selected_row)

# Predict
pred = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0][1]

st.markdown(f"## 🧠 Prediction: {'Fraud ❗' if pred else 'Not Fraud ✅'}")
st.markdown(f"**Probability of Fraud:** `{proba:.2%}`")

# SHAP Explanation
st.markdown("### 🔎 Why this prediction? (SHAP Explanation)")

# SHAP init
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(selected_row)

# Plot SHAP
plt.figure(figsize=(10, 3))
shap.waterfall_plot(shap.Explanation(values=shap_values[1][0], 
                                     base_values=explainer.expected_value[1],
                                     data=selected_row.iloc[0]))
st.pyplot(plt.gcf())
