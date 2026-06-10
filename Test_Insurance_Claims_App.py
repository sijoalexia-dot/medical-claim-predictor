import streamlit as st
import joblib 
import pandas as pd
import numpy as np
import sklearn


# 1. Load the trained model
rf_model = joblib.load("Insurance_Claims.pkl")

with st.sidebar:
    st.header("⚠️ Disclaimer")
    st.caption(
        "This application is a machine learning prototype built for educational "
        "and portfolio demonstration purposes. It uses mathematically generated, "
        "simulated healthcare data. It does not handle protected health information "
        "(PHI), is not HIPAA-compliant, and should not be used for actual clinical "
        "or financial medical billing decisions.")


st.title("Test Insurance Claims App")
st.write("Enter the required feature information below to get a prediction.")

# 2. Collect user information using Streamlit widgets
# Customize these fields to match the exact features your model expects
CTC = st.number_input("Select The Claim Total Charge", min_value=100, max_value=3000)
CDD = st.slider("Select The Duration (Days) of Your Stay", min_value=1, max_value=14, value=7, step= 1)
ERV = st.selectbox("Did you go to the Emergency Room?", ["Yes", "No"]) #Can delete if wanted
IPV = st.number_input("What is your Insurance Provider Code?", min_value=100)
SMC = st.number_input("What is the Submission Method Code?", min_value=100)
MAF = st.selectbox("Is there a Missing Authorization Flag?", ["Yes", "No"])
DCC = st.slider("Select Your Diagnostic Category Code", min_value=1, max_value=9, value=7, step= 1)

# 3. Preprocess inputs to match your model's training format
# Example: Map categorical variables if your model used numerical values
ERV_mapping = {"No": 0, "Yes": 1}
ERV_encoded = ERV_mapping[ERV]

MAF_mapping = {"No": 0, "Yes": 1}
MAF_encoded = MAF_mapping[MAF]
# 4. Create the feature array/dataframe for inference
input_data = pd.DataFrame([{
    'Claim_Total_Charge': CTC,
    'Care_Duration_Days': CDD,
    'Emergency_Room_Visit': ERV_encoded,
    'Insurance_Provider_Code': IPV,
    'Submission_Method_Code' : SMC, 
    'Missing_Authorization_Flag' : MAF_encoded,
    'Diagnostic_Category_Code' : DCC
}])

# 5. Predict when user clicks the button
if st.button("Run Model Prediction"):
    prediction = rf_model.predict(input_data)

    # Convert numerical prediction to a human-readable status string
    if prediction[0] == 0:
        status_result = "Paid Claims"
    else:
        status_result = "Denied Claim"  # Or whatever your '1' value represents (e.g., Approved)

    # Check if model output is classification probabilities or direct classes
    if hasattr(rf_model, "predict_proba"):
        probability = rf_model.predict_proba(input_data)
        st.subheader(f"Prediction Result: {status_result}")
        st.write(f"Confidence Level: {np.max(probability) * 100:.2f}%")
    else:
        st.subheader(f"Prediction Result: {status_result}")
# Assuming 'rfc' is your trained RandomForestClassifier
    importances = rf_model.feature_importances_
    feature_names = ['Claim_Total_Charge', 'Care_Duration_Days', 'Emergency_Room_Visit', 
                 'Insurance_Provider_Code', 'Submission_Method_Code', 
                 'Missing_Authorization_Flag', 'Diagnostic_Category_Code']

    df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    st.subheader("⚠️ Top Denial Risk Drivers")
    st.bar_chart(df_importance.set_index('Feature'))
