# app.py (Corrected SHAP interpretation section)

# ... (Previous code including loading model, scaler, UI, etc.) ...

# --- Preprocess Input & Make Prediction ---
if st.button("Predict Fraud"):
    # ... (Your input processing and prediction logic) ...

    st.subheader("Prediction Result:")
    if prediction_label == 1:
        st.error(f"🔴 FRAUDULENT TRANSACTION DETECTED!")
    else:
        st.success(f"🟢 LEGITIMATE TRANSACTION.")
    
    st.write(f"**Fraud Probability:** {prediction_proba:.4f}")

    # --- SHAP Interpretation for the single prediction ---
    st.subheader("Why this prediction? (Feature Contributions)")
    
    explainer = shap.TreeExplainer(model)
    
    # --- CRITICAL FIX FOR SHAP IndexError ---
    shap_values_raw_single = explainer.shap_values(scaled_input) 

    # Check if shap_values_raw_single is a list (typical for binary classification)
    if isinstance(shap_values_raw_single, list) and len(shap_values_raw_single) > 1:
        # If it's a list with multiple elements, assume index 1 is for the positive class
        shap_values_for_plot_single = shap_values_raw_single[1] 
        expected_value_for_plot_single = explainer.expected_value[1] 
    else:
        # If it's not a list, or list has only one element, assume it's directly the SHAP values for the output
        # This happens if the model output is interpreted as a single-dimensional prediction (e.g., probability)
        shap_values_for_plot_single = shap_values_raw_single
        expected_value_for_plot_single = explainer.expected_value # No indexing needed here
    
    # Ensure shap_values_for_plot_single is 1D for force_plot (it should be 1, n_features)
    # The .force_plot expects (n_features,) for a single instance.
    if shap_values_for_plot_single.ndim > 1:
        shap_values_for_plot_single = shap_values_for_plot_single[0] # Take the first (and only) instance's values


    single_input_df = pd.DataFrame(scaled_input, columns=feature_columns)

    st.write("The plot below shows how each feature pushed the prediction (blue for negative impact, red for positive impact).")
    shap.initjs()
    html = f"<head>{shap.getjs()}</head><body>{shap.force_plot(expected_value_for_plot_single, shap_values_for_plot_single, single_input_df).html()}</body>"
    st.components.v1.html(html, height=300, scrolling=True)

    st.info("💡 A higher red bar means that feature value pushed the prediction towards FRAUD. A higher blue bar means it pushed it towards LEGITIMATE.")