import joblib
import json
import pandas as pd

# Load the trained model
model = joblib.load("mental_health_model.pkl")

# Load feature names
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

def get_user_input():
    """Get user input and convert it into a dataframe for prediction."""
    
    age = int(input("Enter your age: "))
    gender = input("Enter your gender (Male/Female/Other): ").strip()
    family_history = input("Do you have a family history of mental illness? (Yes/No): ").strip()
    work_interfere = input("How often does mental health interfere with work? (Never/Rarely/Sometimes/Often): ").strip()
    tech_company = input("Do you work in a tech company? (Yes/No): ").strip()
    self_employed = input("Are you self-employed? (Yes/No/Unknown): ").strip()
    
    # Convert categorical inputs into one-hot encoding
    user_data = {
        "Age": [age],
        f"Gender_{gender}": [1],
        f"family_history_{family_history}": [1],
        f"work_interfere_{work_interfere}": [1],
        f"tech_company_{tech_company}": [1],
        f"self_employed_{self_employed}": [1],
    }
    
    # Convert to DataFrame
    user_df = pd.DataFrame(user_data)

    # Ensure all feature columns exist
    user_df = user_df.reindex(columns=feature_names, fill_value=0)

    return user_df

# Get input from user
user_df = get_user_input()

# Make prediction
prediction = model.predict(user_df)
probability = model.predict_proba(user_df)[:, 1]

# Show results
print("Mental Health Prediction System")
print(f"Prediction: {'Needs Attention' if prediction[0] else 'No Major Concern'}")
print(f"Confidence: {probability[0]:.2f}")
if probability >= 0.75:
    print("The model is quite confident in its prediction.")
elif probability >= 0.5:
    print("The model suggests this but with moderate confidence.")
else:
    print("The model's confidence is low. Consider providing more details.")