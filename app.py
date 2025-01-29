import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained SVM model and StandardScaler
model_path = "svm_trained_model.pkl"
scaler_path = "scaler.pkl"  # Ensure this file is in the same directory

try:
    model = joblib.load(model_path)  # Load trained model
    scaler = joblib.load(scaler_path)  # Load saved scaler
    st.success("✅ Model and Scaler Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")

# Function to encode gender (Male = 1, Female = 0) based on training encoding
def encode_gender(gender):
    label_encoder = LabelEncoder()
    label_encoder.fit(["Female", "Male"])  # Fit the encoder to map "Female" -> 0, "Male" -> 1
    return label_encoder.transform([gender])[0]

# Function to make predictions
def predict_svm(gender, age, salary):
    try:
        # Scale only the age and salary (not gender)
        scaled_features = scaler.transform([[age, salary]])
        
        # Combine the encoded gender with the scaled age and salary
        input_features = np.array([gender] + scaled_features[0].tolist())  # Add gender to the scaled features

        prediction = model.predict([input_features])  # Predict using SVM
        return prediction[0]  # Return the result
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
def main():
    st.title("🔮 SVM Model Prediction App")
    st.write("Enter the required inputs to get a prediction:")

    # User input fields
    gender = st.selectbox("Gender", ["Female", "Male"])  # Dropdown for gender
    age = st.number_input("Age", min_value=1, max_value=100, value=30)  
    salary = st.number_input("Estimated Salary", min_value=0, value=50000)

    # Convert gender to numerical value (using label encoding)
    gender_encoded = encode_gender(gender)

    # Predict when button is clicked
    if st.button("🔍 Predict"):
        result = predict_svm(gender_encoded, age, salary)
        st.success(f"🔎 Predicted Output: {result}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
