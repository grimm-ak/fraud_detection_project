import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load the model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Set Streamlit page config
st.set_page_config(page_title="Fraud Detection App", layout="wide")

st.title("💸 Fraud Detection App using LightGBM")
st.write("This app predicts whether a transaction is fraudulent or not based on selected inputs.")

# Load a sample row with correct columns (used during model training)
# Replace this dict with any real example row from your training set
sample_data = pd.DataFrame([{
    'type_CASH_OUT': 0, 'type_DEBIT': 0, 'type_PAYMENT': 0, 'type_TRANSFER': 1,
    'amount': 3900.0, 'oldbalanceOrg': 4200.0, 'newbalanceOrig': 300.0,
    'oldbalanceDest': 0.0, 'newbalanceDest': 0.0,
    'balanceDiffOrg': -3900.0, 'balanceDiffDest': 0.0
}])

# Let user adjust input values
st.subheader("🔧 Input Features")
selected_row = sample_data.copy()
for col in selected_row.columns:
    if 'type_' in col:
        selected_row[col] = st.selectbox(f"{col}", [0, 1], index=int(selected_row[col][0]))
    else:
        selected_row[col] = st.number_input(f"{col}", value=float(selected_row[col][0]))

# Show the updated row
st.markdown("### Final Input Preview")
st.dataframe(selected_row)

# Scale and predict
X_scaled = scaler.transform(selected_row)
pred = model.predict(X_scaled)[0]
proba = model.predict_proba(X_scaled)[0][1]

st.markdown(f"### ✅ Prediction: {'Fraud' if pred==1 else 'Not Fraud'} with probability **{proba:.2%}**")

# SHAP Explainer and Plot
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(pd.DataFrame(X_scaled, columns=selected_row.columns))

st.markdown("### 🔍 Feature Impact (SHAP Values)")
fig, ax = plt.subplots(figsize=(10, 3))
shap.bar_plot(shap_values[1][0], feature_names=selected_row.columns, max_display=10)
st.pyplot(fig)

st.markdown("---")
st.markdown("Made with ❤️ using LightGBM, SHAP, and Streamlit")
