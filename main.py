import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🧑‍⚕️", layout="wide")

# Load the trained model
try:
    model = load_model('diabetes_prediction_model_v1.h5')
except Exception as e:
    st.error(f"❌ Error loading the model: {e}")
    st.stop()

# Load the fitted scaler
try:
    scaler = joblib.load("scaler_diabetes.pkl")
except Exception as e:
    st.error(f"❌ Error loading the scaler: {e}")
    st.stop()

def predict_diabetes(bmi, hba1c, glucose):
    """Makes a prediction using the loaded model and scaler."""
    input_data = pd.DataFrame({
        'hbA1c_level': [hba1c],
        'blood_glucose_level': [glucose]
    })

    # Scale the input using pre-fitted scaler
    scaled_input = scaler.transform(input_data)

    # Reshape for LSTM: (samples, time_steps, features)
    reshaped_input = np.array(scaled_input).reshape((1, 1, 3))

    # Make prediction
    prediction = model.predict(reshaped_input)[0][0]
    return prediction

# Streamlit UI
st.title("🩺 Diabetes Prediction App")
st.write("Enter your health metrics below to get a diabetes risk prediction:")

hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, step=0.1, value=5.5)
glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, step=1.0, value=100.0)

if st.button("🔍 Predict"):
    prediction = predict_diabetes(hba1c, glucose)
    st.subheader("Prediction Result:")

    if prediction >= 0.5:
        st.warning(f"⚠️ High risk of diabetes.\nProbability: **{prediction:.2f}**")
    else:
        st.success(f"✅ Low risk of diabetes.\nProbability: **{prediction:.2f}**")

    st.caption("🧠 Note: This model is an AI-based predictor and should not replace professional medical advice.")
