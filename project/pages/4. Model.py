import streamlit as st
import pandas as pd
import numpy as np 

# --- Dummy Model Results (Kept for function completeness) ---
def get_dummy_results(name):
    """Generates placeholder metrics for demonstration."""
    if "Cirrhosis" in name:
        return {
            "Accuracy": 0.85,
            "F1 Score": 0.82,
            "Best Model": "Random Forest",
            "Target Column": "Target_Cirrhosis"
        }
    elif "Hepatitis" in name:
        return {
            "Accuracy": 0.92,
            "F1 Score": 0.91,
            "Best Model": "Support Vector Machine (SVM)",
            "Target Column": "Target_Hepatitis"
        }
    elif "Indian Liver Patient" in name:
        return {
            "Accuracy": 0.75,
            "F1 Score": 0.74,
            "Best Model": "Logistic Regression",
            "Target Column": "Is_Patient"
        }
    return {}

def get_dummy_prediction(model_name, features):
    """
    Placeholder function to simulate prediction based on input features, 
    incorporating medical trends and updated input ranges.
    """
    
    # ----------------------------------------------------
    # Define the new LOGIC RANGES based on the user's input/screenshot
    # These ranges are used for normalization inside the function.
    # ----------------------------------------------------
    
    # Age: 18 to 90
    MIN_AGE, MAX_AGE = 18, 90
    
    # Bilirubin: 0.1 to 20.0 (High value increases risk) - Weight: +0.2
    MIN_BILIRUBIN, MAX_BILIRUBIN = 0.1, 20.0
    
    # Albumin: 1.0 to 5.0 (Low value increases risk) - Weight: -0.2
    MIN_ALBUMIN, MAX_ALBUMIN = 1.0, 5.0
    
    # SGOT: 10 to 4925 (High value increases risk) - Weight: +0.1
    MIN_SGOT, MAX_SGOT = 10, 4925
    
    # Alk_Phos: 3 to 2110 (High value slightly increases risk) - Weight: +0.05
    MIN_ALK_PHOS, MAX_ALK_PHOS = 3, 2110
    
    # ----------------------------------------------------
    # Normalization (Scaling feature values to a 0-1 range)
    # ----------------------------------------------------
    
    def safe_normalize(val, min_val, max_val):
        """Safely scales a value between 0 and 1."""
        if max_val <= min_val: # Avoid division by zero/invalid range
            return 0.0
        return np.clip((val - min_val) / (max_val - min_val), 0.0, 1.0)
    
    norm_bilirubin = safe_normalize(features['Bilirubin'], MIN_BILIRUBIN, MAX_BILIRUBIN)
    norm_sgot = safe_normalize(features['SGOT'], MIN_SGOT, MAX_SGOT)
    norm_age = safe_normalize(features['Age'], MIN_AGE, MAX_AGE)
    norm_alk_phos = safe_normalize(features['Alk_Phos'], MIN_ALK_PHOS, MAX_ALK_PHOS)

    # Albumin is inverted: lower value = higher risk (1 - normalized_albumin)
    norm_albumin = safe_normalize(features['Albumin'], MIN_ALBUMIN, MAX_ALBUMIN)
    norm_albumin_inverted = 1.0 - norm_albumin
    
    # Gender (Placeholder: Male tends to have slightly higher risk)
    gender_risk = 0.05 if features['Gender'] == 'Male' else 0.0

    # ----------------------------------------------------
    # Calculate base risk score
    # ----------------------------------------------------
    base_risk_score = (
        0.30 + # Intercept/Base probability
        (norm_bilirubin * 0.20) +          # Bilirubin: High = High Risk
        (norm_albumin_inverted * 0.20) +   # Albumin: Low = High Risk (inverted)
        (norm_sgot * 0.10) +               # SGOT: High = High Risk
        (norm_age * 0.05) +                # Age: High = High Risk
        (norm_alk_phos * 0.05) +           # Alk Phos: High = High Risk
        gender_risk
    )
    
    # Introduce model-specific variation
    np.random.seed(hash(model_name) % 1000) 
    variation = np.random.uniform(-0.05, 0.05) 
    
    # Final probability (clamped between 0.05 and 0.95)
    probability = np.clip(base_risk_score + variation, 0.05, 0.95)
    
    return probability

def display_model():
    """Displays only the Interactive Prediction section."""
    
    st.title("💡 Model Prediction Tool")
    st.markdown("Use this tool to simulate a patient's results and see the predicted risk of a target condition across different models.")
    
    # --- 1. Interactive Prediction Tool ---
    
    st.header("🔮 Patient Input and Prediction")

    # Define all input ranges based on the screenshot provided
    # Note: st.number_input does not allow '<' or '-' in the label, so the range indicators are moved to the help text.
    
    st.subheader("Patient Input Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Age Slider (Range 18-90)
        age = st.slider("Age", 18, 90, 49) # Default changed to 49 as per screenshot
        
        # Gender Selectbox
        gender = st.selectbox("Gender", ["Male", "Female"], index=1) # Default changed to Female
        
        # Bilirubin Input (Range 0.1-20.0)
        bilirubin = st.number_input(
            "Total Bilirubin (mg/dL)<0.1 - 20.0>", 
            min_value=0.1, max_value=20.0, value=4.0, step=0.1, format="%.1f",
            help="Range: 0.1 - 20.0"
        ) # Default changed to 4.0

    with col2:
        # Alk phosphate Input (Range 3-2110)
        alk_phos = st.number_input(
            "Alkaline Phosphatase (IU/L)<3 - 2110>", 
            min_value=3, max_value=2110, value=800, step=10,
            help="Range: 3 - 2110"
        ) # Range and default changed
        
        # SGOT Input (Range 10-4925)
        sgot = st.number_input(
            "SGOT/AST (U/L)<10 - 4925>", 
            min_value=10, max_value=4925, value=3000, step=10,
            help="Range: 10 - 4925"
        ) # Range and default changed
        
    with col3:
        # Albumin Input (Range 1.0-5.0)
        albumin = st.number_input(
            "Albumin (g/dL)<1.0 - 5.0>", 
            min_value=1.0, max_value=5.0, value=4.3, step=0.1, format="%.1f",
            help="Range: 1.0 - 5.0"
        ) # Range and default changed
        
    st.markdown("---")
    
    # Compile features for prediction
    input_features = {
        "Age": age,
        "Gender": gender,
        "Bilirubin": bilirubin,
        "Alk_Phos": alk_phos,
        "SGOT": sgot,
        "Albumin": albumin
    }

    # Display different models' results
    model_comparisons = [
        "Random Forest", 
        "Support Vector Machine (SVM)", 
        "Logistic Regression"
    ]

    st.subheader("Predicted Condition Risk (%)")
    
    # Display the results in columns
    pred_cols = st.columns(len(model_comparisons))
    
    for idx, model in enumerate(model_comparisons):
        prob = get_dummy_prediction(model, input_features)
        
        # Determine risk level for visual feedback
        if prob > 0.75:
            risk_level = "High Risk"
            color = "red"
        elif prob > 0.5:
            risk_level = "Moderate Risk"
            color = "orange"
        else:
            risk_level = "Low Risk"
            color = "green"
        
        with pred_cols[idx]:
            # Display the result using st.metric for a clean look
            pred_cols[idx].metric(model, f"{prob*100:.1f}%")
            st.markdown(f"**<p style='color:{color}; font-size:14px; text-align: center;'>{risk_level}</p>**", unsafe_allow_html=True)
            
    st.markdown("---")
    
# --- RUN THE APP ---
if __name__ == '__main__':
    display_model()