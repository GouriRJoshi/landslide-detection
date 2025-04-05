import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('landslide_model.pkl')

# Title
st.title("Landslide Detection App")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    avg_rainfall = st.sidebar.slider('Average Rainfall (mm)', 0.0, 1000.0, 200.0)
    slope_angle = st.sidebar.slider('Slope Angle (degrees)', 0.0, 90.0, 30.0)
    soil_moisture = st.sidebar.slider('Soil Moisture (%)', 0.0, 100.0, 50.0)
    elevation = st.sidebar.slider('Elevation (m)', 0, 5000, 1000)
    vegetation = st.sidebar.slider('Vegetation Cover (%)', 0.0, 100.0, 50.0)

    data = {
        'avg_rainfall': avg_rainfall,
        'slope_angle': slope_angle,
        'soil_moisture': soil_moisture,
        'elevation': elevation,
        'vegetation': vegetation
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Display input features
st.subheader('User Input Parameters')
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# Output results
st.subheader('Prediction')
st.write('🌋 Landslide Likely' if prediction == 1 else '✅ No Landslide Risk')

st.subheader('Prediction Probability')
st.write(f"No Landslide: {prediction_proba[0]*100:.2f}%")
st.write(f"Landslide: {prediction_proba[1]*100:.2f}%")
