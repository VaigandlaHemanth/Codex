import streamlit as st
import pandas as pd
import joblib

# Load the saved model, encoders, and scaler
model = joblib.load("health_risk_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

def preprocess_input(data):
    """
    Preprocesses user input for prediction.
    Args:
        data (dict): User input data.
    Returns:
        pd.DataFrame: Preprocessed data ready for model prediction.
    """
    data = pd.DataFrame([data])
    categorical_features = ['gender', 'activity_level', 'smoking_status', 'alcohol_intake']
    numeric_features = ['age', 'weight_kg', 'height_cm', 'heart_rate', 'calories']

    # Process categorical features
    for col in categorical_features:
        le = label_encoders[col]
        if data[col][0] not in le.classes_:
            st.warning(f"Value '{data[col][0]}' for '{col}' not in training data. Defaulting to '{le.classes_[0]}'.")
            data[col] = le.classes_[0]
        data[col] = le.transform([data[col][0]])[0]  # Encode the categorical value

    # Scale numeric features
    data[numeric_features] = scaler.transform(data[numeric_features])

    # Ensure the entire DataFrame is numeric
    return data.astype(float)




# Streamlit UI
st.title("Smart Health Monitoring System")

# Input form
user_data = {
    "age": st.number_input("Age", min_value=0, max_value=120, value=25),
    "gender": st.selectbox("Gender", options=["Male", "Female"]),
    "weight_kg": st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=70.0),
    "height_cm": st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=170.0),
    "heart_rate": st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=200.0, value=70.0),
    "calories": st.number_input("Calories Burned", min_value=0.0, max_value=5000.0, value=2000.0),
    "activity_level": st.selectbox("Activity Level", options=["Low", "Moderate", "High"]),
    "smoking_status": st.selectbox("Smoking Status", options=["Non-Smoker", "Occasional", "Regular"]),
    "alcohol_intake": st.selectbox("Alcohol Intake", options=["None", "Occasional", "Regular"]),
}

if st.button("Predict Health Risk"):
    try:
        # Preprocess user data
        processed_data = preprocess_input(user_data)

        # Predict risk
        risk_prediction = model.predict(processed_data.values)[0]
        predicted_risk_label = label_encoders['risk'].inverse_transform([risk_prediction])[0]

        st.success(f"The predicted health risk is: {predicted_risk_label}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
