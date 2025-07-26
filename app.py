import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Fraud Detection using LightGBM")

# Load the trained model and scaler
model = joblib.load("best_lgbm_clf_model.joblib")
scaler = joblib.load("scaler.joblib")

# File upload section
uploaded_file = st.file_uploader("📂 Upload Transaction CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Uploaded Data")
    st.dataframe(df.head(10))

    # Preprocessing
    st.subheader("⚙️ Data Preprocessing")
    df["balanceDiffOrg"] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df["balanceDiffDest"] = df['newbalanceDest'] - df['oldbalanceDest']
    df_encoded = pd.get_dummies(df, columns=["type"], drop_first=True)

    # Drop unused/leaky features
    drop_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    drop_cols = [col for col in drop_cols if col in df_encoded.columns]
    X = df_encoded.drop(drop_cols + ['isFraud'], axis=1, errors='ignore')
    
    # Target
    y = df_encoded['isFraud'] if 'isFraud' in df_encoded.columns else None

    # Feature scaling
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"❌ Error during scaling: {e}")
        st.stop()

    # Predictions
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    df['Fraud Prediction'] = y_pred
    df['Fraud Probability'] = y_proba

    st.subheader("🔍 Prediction Results")
    st.dataframe(df[['Fraud Prediction', 'Fraud Probability'] + list(X.columns)].head(10))

    # Downloadable predictions
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Prediction CSV", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

    # Evaluation (if ground truth exists)
    if y is not None:
        st.subheader("📊 Model Evaluation")

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_proba)

        st.markdown(f"- **Accuracy**: `{accuracy:.4f}`")
        st.markdown(f"- **Precision**: `{precision:.4f}`")
        st.markdown(f"- **Recall**: `{recall:.4f}`")
        st.markdown(f"- **F1 Score**: `{f1:.4f}`")
        st.markdown(f"- **ROC AUC Score**: `{roc_auc:.4f}`")

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        # Classification report
        st.subheader("📄 Classification Report")
        st.text(classification_report(y, y_pred))

else:
    st.info("👆 Please upload a CSV file to get started.")
