import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load the trained model
model = joblib.load("mental_health_model.pkl")

# Load feature names to ensure input format matches training
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

# Streamlit App Title
st.title("🧠 Mental Health Prediction System")
st.write("Answer the following questions to assess your mental health risk.")

age = st.number_input("Enter your age:", min_value=10, max_value=100, step=1, format="%d")
gender = st.selectbox("Select your gender:", ["Male", "Female", "Other"], index=None, placeholder="Select...")
family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"], index=None, placeholder="Select...")
work_interfere = st.selectbox("How often does mental health interfere with work?", ["Never", "Rarely", "Sometimes", "Often"], index=None, placeholder="Select...")
tech_company = st.selectbox("Do you work in a tech company?", ["Yes", "No"], index=None, placeholder="Select...")
self_employed = st.selectbox("Are you self-employed?", ["Yes", "No", "Unknown"], index=None, placeholder="Select...")

# Prevent user errors
if None in [gender, family_history, work_interfere, tech_company, self_employed]:
    st.warning("⚠️ Please fill in all fields before predicting.")
else:
    # ✅ Convert inputs into a DataFrame
    user_data = pd.DataFrame([{col: 0 for col in feature_names}])  # Ensure all expected features are present

    # Fill in user inputs
    user_data["Age"] = age
    if f"Gender_{gender}" in feature_names:
        user_data[f"Gender_{gender}"] = 1
    if f"family_history_{family_history}" in feature_names:
        user_data[f"family_history_{family_history}"] = 1
    if f"work_interfere_{work_interfere}" in feature_names:
        user_data[f"work_interfere_{work_interfere}"] = 1
    if f"tech_company_{tech_company}" in feature_names:
        user_data[f"tech_company_{tech_company}"] = 1
    if f"self_employed_{self_employed}" in feature_names:
        user_data[f"self_employed_{self_employed}"] = 1

    # Ensure user_data matches trained model feature order
    user_data = user_data.reindex(columns=feature_names, fill_value=0)

    # ✅ Prediction Button
    if st.button("Predict"):
        prediction = model.predict(user_data)[0]
        confidence = model.predict_proba(user_data)[0][1]  # Confidence score for "Needs Attention"

        # Display result
        if prediction == 1:
            st.error(f"⚠️ **Prediction:** Needs Attention (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ **Prediction:** No Significant Risk (Confidence: {confidence:.2f})")
