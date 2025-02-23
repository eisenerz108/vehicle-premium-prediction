import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI with styling
st.set_page_config(page_title="Car Insurance Premium Predictor", layout="centered")
st.markdown("## 🚗 Car Insurance Premium Predictor")
st.markdown("### Enter details below to estimate your insurance premium")

# Input fields with sliders and number inputs
driverAge = st.slider("Driver Age", min_value=18, max_value=100, value=30, step=1)
driverExperience = st.slider("Driver Experience (in years)", min_value=0, max_value=80, value=5, step=1)
previousAccidents = st.number_input("Number of Previous Accidents", min_value=0, max_value=10, value=0, step=1)
annualMileage = st.slider("Annual Mileage (in km)", min_value=0, max_value=500000, value=15000, step=1000)
carManufacturingYear = st.number_input("Car Manufacturing Year", min_value=1950, max_value=2025, value=2015, step=1)
carAge = st.slider("Car Age (in years)", min_value=0, max_value=75, value=5, step=1)

# Prediction button with better UI
if st.button("🚀 Predict Premium", help="Click to estimate your insurance premium"):
    input_data = np.array([driverAge, driverExperience, previousAccidents, annualMileage, carAge]).reshape(1, -1)
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Premium: {prediction[0]:,.2f}")
