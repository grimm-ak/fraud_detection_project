import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# --- Page Config ---
st.set_page_config(page_title="Fraud Detection System", layout="centered")

# --- Load model & scaler ---
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('best_lgbm_clf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler not found.")
        st.stop()

model, scaler = load_model_and_scaler()

# --- Load test data for evaluation ---
@st.cache_data
def load_test_data():
    try:
        df = pd.read_csv("test_data.csv")  # Ensure this file exists
        X = df.drop("label", axis=1)
        y = df["label"]
        X_scaled = scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns), y
    except Exception as e:
        return None, None

X_test, y_test = load_test_data()

# --- Feature Columns ---
feature_columns = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balanceDiffOrg', 'balanceDiffDest',
    'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
]

# --- App Title ---
st.title("🧠 Real-Time Fraud Detection System")
st.markdown("Enter transaction details to predict if it's fraudulent.")

# --- Input UI ---
st.header("Transaction Details")
col1, col2 = st.columns(2)
with col1:
    step = st.number_input("Step (hour)", min_value=1, value=1)
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input("Old Balance Originator", min_value=0.0, value=10000.0)
    newbalanceOrig = st.number_input("New Balance Originator", min_value=0.0, value=9000.0)
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=500.0)
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=1500.0)
with col2:
    transaction_type = st.selectbox("Transaction Type", ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
    st.markdown("*(Note: We use derived features instead of anonymized V1–V28.)*")

# --- Prediction ---
if st.button("Predict Fraud"):
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
    input_df['step'] = step
    input_df['amount'] = amount
    input_df['oldbalanceOrg'] = oldbalanceOrg
    input_df['newbalanceOrig'] = newbalanceOrig
    input_df['oldbalanceDest'] = oldbalanceDest
    input_df['newbalanceDest'] = newbalanceDest
    input_df['balanceDiffOrg'] = oldbalanceOrg - newbalanceOrig
    input_df['balanceDiffDest'] = newbalanceDest - oldbalanceDest

    # One-hot encoding for transaction type
    input_df['type_CASH_OUT'] = transaction_type == 'CASH_OUT'
    input_df['type_DEBIT'] = transaction_type == 'DEBIT'
    input_df['type_PAYMENT'] = transaction_type == 'PAYMENT'
    input_df['type_TRANSFER'] = transaction_type == 'TRANSFER'

    input_data_processed = input_df[feature_columns]
    scaled_array = scaler.transform(input_data_processed)
    scaled_df = pd.DataFrame(scaled_array, columns=feature_columns)

    prediction_proba = model.predict_proba(scaled_df)[:, 1][0]
    prediction_label = model.predict(scaled_df)[0]

    st.subheader("Prediction Result:")
    if prediction_label == 1:
        st.error("🔴 FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success("🟢 LEGITIMATE TRANSACTION.")
    st.write(f"**Fraud Probability:** `{prediction_proba:.4f}`")

    # SHAP Waterfall Explanation
    st.subheader("Why this prediction? (Feature Contributions)")
    explainer = shap.Explainer(model)
    shap_values = explainer(scaled_df)

    plt.clf()
    shap.plots.waterfall(shap_values[0], show=False)
    fig = plt.gcf()
    st.pyplot(fig)

    st.info("💡 Red pushes toward fraud; blue pushes toward legitimate.")

# --- Evaluation Metrics ---
if X_test is not None and y_test is not None:
    st.subheader("📊 Model Evaluation")

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("**Classification Report:**")
    st.dataframe({
        "Accuracy": [round(report['accuracy'], 2)],
        "Precision": [round(report['weighted avg']['precision'], 2)],
        "Recall": [round(report['weighted avg']['recall'], 2)],
        "F1-Score": [round(report['weighted avg']['f1-score'], 2)]
    })

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax_cm, cmap=plt.cm.Blues, colorbar=False)
    st.pyplot(fig_cm)

    # ROC Curve
    st.markdown("### ROC Curve")
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic (ROC)')
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.info("""
    - **Confusion Matrix** compares actual vs predicted.
    - **ROC Curve** shows model's trade-off between sensitivity & specificity.
    - **F1-Score** balances precision & recall, critical for fraud detection.
    """)
else:
    st.warning("⚠️ Test data not found. Evaluation skipped.")
