import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("landslide_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset just to get feature names
df = pd.read_csv("landslide_detection_dataset.csv")
features = df.drop("Landslide", axis=1).columns  # adjust based on your actual label column

# App Title
st.title("🌍 Landslide Detection App")
st.markdown("Predict landslides using machine learning with 98%+ accuracy.")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    input_data = {}
    for feature in features:
        input_data[feature] = st.sidebar.slider(f"{feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
    return pd.DataFrame([input_data])

input_df = user_input_features()

# Main panel
st.subheader("User Input Parameters")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

# Output
st.subheader("Prediction")
st.write("🟠 Landslide Detected" if prediction == 1 else "🟢 No Landslide")

st.subheader("Prediction Probability")
st.write(f"{prediction_proba * 100:.2f}% chance of Landslide")
