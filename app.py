import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("crop_model.pkl")

st.title("🌾 Jharkhand Crop Recommendation System")

st.write("Enter details below to get the best crop recommendation:")

# Numeric features
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

# Extra categorical features (edit according to your dataset)
soil_type = st.selectbox("Soil Type", ["Loamy", "Sandy", "Clay"])
district = st.selectbox("District", ["Ranchi", "Dhanbad", "Hazaribagh", "Other"])
season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
irrigation = st.selectbox("Irrigation Available", ["Yes", "No"])
organic = st.selectbox("Organic Matter Present", ["Yes", "No"])

# Convert categorical inputs to numerical (dummy encoding or simple binary)
soil_map = {"Loamy": [1, 0, 0], "Sandy": [0, 1, 0], "Clay": [0, 0, 1]}
district_map = {"Ranchi": [1, 0, 0, 0], "Dhanbad": [0, 1, 0, 0],
                "Hazaribagh": [0, 0, 1, 0], "Other": [0, 0, 0, 1]}

# You can customize these mappings to fit your training dataset’s one-hot encoding
season_map = {"Kharif": 1, "Rabi": 2, "Zaid": 3}
irrigation_map = {"Yes": 1, "No": 0}
organic_map = {"Yes": 1, "No": 0}

# Combine features in the exact order your model expects
input_data = [
    N, P, K, temperature, humidity, ph, rainfall,
    *soil_map[soil_type][:1],  # use correct number of one-hot columns
    irrigation_map[irrigation],
    organic_map[organic],
    season_map[season]
]

# Convert to 2D array
data = np.array(input_data).reshape(1, -1)

if st.button("Predict Crop 🌱"):
    prediction = model.predict(data)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")

# import streamlit as st
# import pandas as pd
# import pickle

# # Load your trained model (replace with your model filename)
# with open("crop_model.pkl", "rb") as f:
#     model = pickle.load(f)

# st.title("🌱 AI Crop Recommendation System")

# st.write("Enter your soil and environmental conditions to get crop recommendations.")

# # ---------------- Inputs ----------------
# ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)

# nitrogen = st.number_input("Nitrogen content (kg/ha)", min_value=0.0, max_value=200.0, step=1.0)
# phosphorus = st.number_input("Phosphorus content (kg/ha)", min_value=0.0, max_value=200.0, step=1.0)
# potassium = st.number_input("Potassium content (kg/ha)", min_value=0.0, max_value=200.0, step=1.0)

# organic_carbon = st.number_input("Organic Carbon (%)", min_value=0.0, max_value=10.0, step=0.1)

# rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, step=1.0)
# temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
# humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)

# irrigation = st.selectbox("Irrigation Available?", ["Yes", "No"])
# season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])

# landholding = st.number_input("Landholding size (ha)", min_value=0.0, max_value=100.0, step=0.1)

# # ---------------- Data Preparation ----------------
# input_data = pd.DataFrame([{
#     "Soil_pH": ph,
#     "Nitrogen_N": nitrogen,
#     "Phosphorus_P": phosphorus,
#     "Potassium_K": potassium,
#     "Organic_Carbon": organic_carbon,
#     "Rainfall_mm": rainfall,
#     "Temperature_C": temperature,
#     "Humidity_%": humidity,
#     "Irrigation_enc": 1 if irrigation == "Yes" else 0,
#     "Season_enc": {"Kharif": 0, "Rabi": 1, "Zaid": 2}[season],
#     "Landholding_ha": landholding,
#     "Soil_Fertility": nitrogen + phosphorus + potassium
# }])
# import joblib
# scaler = joblib.load("scalar.pkl")
# input_scaled = scaler.transform(input_data)

# # ---------------- Prediction ----------------
# if st.button("Recommended Crop"):
#     # prediction = model.predict(input_data)[0]
#     prediction = model.predict(input_scaled)[0]
#     st.success(f"🌾Recommended Crop: **{prediction}**")
