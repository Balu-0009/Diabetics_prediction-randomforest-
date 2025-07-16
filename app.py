import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

st.title("Diabetes Prediction App")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 0, 120, 30)

# Predict
if st.button("Predict"):
    input_data = [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]]
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"The person is: {result}")
