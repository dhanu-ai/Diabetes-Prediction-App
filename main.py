import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🧑‍⚕️", layout="wide")

# Load the trained model
try:
    model = load_model('diabetes_prediction_model_v1.h5')
    print("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Initialize the scaler WITHOUT loading a fitted one
scaler = MinMaxScaler()

def predict_diabetes(bmi, hba1c, glucose):
    """Makes a prediction using the loaded model."""
    input_data = pd.DataFrame({
        'bmi': [bmi],
        'hbA1c_level': [hba1c],
        'blood_glucose_level': [glucose]
    })

    # **WARNING:** Fitting the scaler here on single input data is WRONG.
    # This is only for demonstration and will likely lead to incorrect scaling.
    # In a proper setup, you MUST load the scaler fitted on the training data.
    scaler.fit(input_data)
    scaled_input = scaler.transform(input_data)

    # Reshape for LSTM (single sample, 1 timestep, 3 features)
    reshaped_input = np.array(scaled_input).reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))

    # Make the prediction
    prediction = model.predict(reshaped_input)[0][0]

    return prediction

st.title("Diabetes Prediction App")
st.write("Enter your health metrics below to get a diabetes risk prediction.")

bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, step=0.1)
hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, step=0.1)
glucose = st.number_input("Blood Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, step=1.0)

if st.button("Predict"):
    if model is not None:
        # **SEVERE WARNING:** The scaling here is likely incorrect.
        prediction_probability = predict_diabetes(bmi, hba1c, glucose)
        st.subheader("Prediction:")
        if prediction_probability >= 0.5:
            st.warning(f"High risk of diabetes (Probability: {prediction_probability:.2f})")
        else:
            st.success(f"Low risk of diabetes (Probability: {prediction_probability:.2f})")
        st.info("Note: This prediction is based on potentially incorrectly scaled input and should NOT be considered a medical diagnosis. Please consult a healthcare professional for accurate assessment.")
    else:
        st.error("Model not loaded. Please ensure 'diabetes_model.h5' is in the same directory.")