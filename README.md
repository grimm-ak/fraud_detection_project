# Real-Time Fraud Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-000?style=for-the-badge&logo=lightgbm&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-4C5C92?style=for-the-badge&logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Business Problem & Impact](#2-business-problem--impact)
- [3. Dataset](#3-dataset)
- [4. Methodology](#4-methodology)
- [5. Model Performance](#5-model-performance)
- [6. Model Interpretability (XAI)](#6-model-interpretability-xai)
- [7. MLOps & Scalability Considerations](#7-mlops--scalability-considerations)
- [8. How to Run Locally](#8-how-to-run-locally)
- [9. Live Demo](#9-live-demo)
- [10. Future Work](#10-future-work)

---

### 1. Project Overview
This project focuses on building and deploying a robust machine learning system for real-time fraud detection in financial transactions. It demonstrates an end-to-end data science workflow from data preprocessing and advanced model training to interpretability and considerations for production deployment.

### 2. Business Problem & Impact
Financial fraud is a significant challenge for banks and e-commerce platforms, leading to substantial financial losses and erosion of customer trust. This project aims to accurately identify fraudulent transactions, enabling organizations to:
- **Minimize Financial Losses:** By detecting fraud quickly and effectively.
- **Improve Customer Experience:** By reducing false positives (legitimate transactions flagged as fraud) and ensuring genuine transactions are processed smoothly.
- **Enhance Security:** By strengthening the overall fraud prevention capabilities.
- **Optimize Resource Allocation:** By focusing fraud investigation efforts on high-probability cases.

### 3. Dataset
The dataset used is the [**Fraud Detection Dataset by Aman Ali Siddiqui**](https://www.kaggle.com/datasets/amanalibox/fraud-detection-dataset) from Kaggle, containing anonymized financial transaction data.
- **Size:** Approximately 6.3 million transactions.
- **Key Features:** Includes transaction amount, time step, original and new account balances, and various anonymized features (V1-V28).
- **Class Imbalance:** The dataset is highly imbalanced, with fraudulent transactions constituting only ~0.1% of the total, presenting a realistic challenge for model development.
- **Download:** Due to its large size (over 100MB), the raw dataset is not hosted directly in this repository. Please download it from the [Kaggle dataset page](https://www.kaggle.com/datasets/amanalibox/fraud-detection-dataset).

### 4. Methodology
The project follows a structured machine learning pipeline:
1.  **Data Loading & Initial Exploration:** Loaded transactional data and performed initial checks on shape, columns, and basic statistics.
2.  **Feature Engineering (Basic):**
    - Created `balanceDiffOrg` (change in originating account balance) and `balanceDiffDest` (change in destination account balance) as derived features.
    - Utilized existing numerical features (`step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`).
    - Handled categorical `type` feature via One-Hot Encoding.
3.  **Missing Value Imputation:** Missing numerical values were imputed using the median of their respective columns from the training data to ensure data completeness.
4.  **Class Imbalance Handling:** To address the extreme class imbalance (approx. 965:1 non-fraud to fraud), `RandomUnderSampler` was applied to balance the dataset, ensuring the model learns effectively from both classes.
5.  **Data Splitting:** Data was split into training and testing sets (e.g., 80/20 ratio), with stratification to maintain class balance in both sets.
6.  **Feature Scaling:** Numerical features were scaled using `StandardScaler` to standardize their range, aiding model convergence and performance.
7.  **Model Training (LightGBM with Hyperparameter Tuning):**
    - A **LightGBM Classifier** was chosen for its high performance and efficiency with tabular data.
    - **`GridSearchCV`** was employed for comprehensive hyperparameter tuning, optimizing parameters such as `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `reg_alpha`, and `reg_lambda` to achieve the best `ROC_AUC` score.
8.  **Model Evaluation:** Performance was rigorously evaluated on the unseen test set using key metrics for imbalanced classification.

### 5. Model Performance
The optimized LightGBM Classifier achieved outstanding performance on the balanced test set:
- **Accuracy:** `[Your Test Set Accuracy, e.g., 0.9946]`
- **Precision:** `[Your Test Set Precision, e.g., 0.9920]`
- **Recall:** `[Your Test Set Recall, e.g., 0.9973]`
- **F1-Score:** `[Your Test Set F1-Score, e.g., 0.9946]`
- **ROC AUC:** `[Your Test Set ROC AUC, e.g., 0.9999]`

This performance demonstrates the model's high capability in accurately identifying fraudulent transactions while minimizing false positives and false negatives.

### 6. Model Interpretability (XAI)
To provide transparency and build trust, **SHAP (SHapley Additive exPlanations)** was integrated to interpret the model's decisions:
- **Global Feature Importance:** Visualizations (Bar and Beeswarm plots, which are generated in the accompanying Jupyter Notebook) illustrate which features contribute most to the model's overall fraud predictions.
- **Individual Prediction Explanation:** An interactive [SHAP force plot](<You can include a static image of a sample force plot here, or simply describe its functionality, as it's interactive in the live demo.>) is generated in the live demo to explain *why* a specific transaction was flagged as fraudulent, detailing the contribution of each feature value.

### 7. MLOps & Scalability Considerations
This project is designed with production readiness in mind:
- **Containerization:** The application is **containerized using Docker**, ensuring a consistent and isolated environment for deployment across various platforms.
- **Deployment Strategy:** The Streamlit application (see [Live Demo](#9-live-demo) below) demonstrates real-time inference capability.
    - *For live deployment, an automated pinging mechanism via GitHub Actions is implemented to prevent app inactivity shutdown on free tiers.*
    - *For enterprise-grade deployment, the Dockerized application can be seamlessly deployed to cloud platforms such as **Google Cloud Run** or **AWS EC2**, leveraging their scalability and robust infrastructure.*
- **Real-time Data Ingestion (Conceptual):** For truly real-time fraud detection at scale, data would be ingested via streaming platforms like **Apache Kafka** (e.g., from payment gateways), processed, and fed to the model.
- **Model Monitoring (Conceptual):** In a production environment, continuous monitoring would be established for model performance, data drift, and concept drift, with alerts triggered for degradation.
- **Model & Data Versioning (Conceptual):** Tools like MLflow for experiment tracking and DVC (Data Version Control) would be used to manage different versions of models and datasets, ensuring reproducibility and auditability.

### 8. How to Run Locally
To run this project on your local machine:
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/grimm-ak/fraud_detection_project.git](https://github.com/grimm-ak/fraud_detection_project.git) # Replace with your actual repo URL
    cd fraud_detection_project
    ```
2.  **Download Dataset:** Download `AIML Dataset.csv` from Kaggle ([https://www.kaggle.com/datasets/amanalibox/fraud-detection-dataset](https://www.kaggle.com/datasets/amanalibox/fraud-detection-dataset)) and place it in a `data/` directory (create this folder if it doesn't exist within your cloned repository).
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Fraud_Detection_Notebook.ipynb # Replace with your actual notebook name
    ```
    *(Run all cells in the notebook to ensure model and scaler are saved to .joblib files.)*
6.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

### 9. Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_ACTUAL_STREAMLIT_CLOUD_URL_HERE)
*(**CRITICAL:** Replace `YOUR_ACTUAL_STREAMLIT_CLOUD_URL_HERE` with the actual URL of your deployed Streamlit app. This is the link recruiters will click!)*

### 10. Future Work
- Explore more advanced feature engineering techniques (e.g., transaction velocity, spending habits over time, graph-based features if data allows).
- Implement more sophisticated sampling methods (e.g., SMOTE, ADASYN) or cost-sensitive learning.
- Experiment with other ensemble models (CatBoost) or Deep Learning models (e.g., Autoencoders for anomaly detection).
- Full MLOps pipeline integration with CI/CD for continuous model deployment and retraining.
- Enhance the user interface with more interactive elements or visual feedback.

---

**Your Tasks:**

1.  **Copy and paste this entire template** into your `README.md` file.
2.  **Fill in all the placeholders** with your specific metrics and URLs.
    * `[Your Test Set Accuracy, Precision, Recall, F1-Score, ROC AUC]`
    * `YOUR_ACTUAL_STREAMLIT_CLOUD_URL_HERE`
    * `https://github.com/grimm-ak/fraud_detection_project.git` (confirm your exact repo URL)
    * `Fraud_Detection_Notebook.ipynb` (confirm your actual notebook name)
3.  **Save the `README.md` file.**
4.  **Commit and push the `README.md` to your GitHub repository.**