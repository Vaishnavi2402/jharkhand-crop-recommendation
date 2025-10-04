import streamlit as st
import pandas as pd
import pickle
import joblib

# ---------------- Load trained model & scaler ----------------
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

# prediction = model.predict(data)
st.title("🌱 AI Crop Recommendation System")
st.write("Enter your soil and environmental conditions to get crop recommendations.")

# ---------------- Inputs ----------------
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
nitrogen = st.number_input("Nitrogen content (kg/ha)", min_value=0.0, max_value=200.0, step=1.0)
phosphorus = st.number_input("Phosphorus content (kg/ha)", min_value=0.0, max_value=200.0, step=1.0)
potassium = st.number_input("Potassium content (kg/ha)", min_value=0.0, max_value=200.0, step=1.0)
organic_carbon = st.number_input("Organic Carbon (%)", min_value=0.0, max_value=10.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0)

irrigation = st.selectbox("Irrigation Available?", ["Yes", "No"])
season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
landholding = st.number_input("Landholding size (ha)", min_value=0.0, max_value=100.0, step=0.1)

# ---------------- Prepare input data ----------------
input_data = pd.DataFrame([{
    "Soil_pH": ph,
    "Nitrogen_N": nitrogen,
    "Phosphorus_P": phosphorus,
    "Potassium_K": potassium,
    "Organic_Carbon": organic_carbon,
    "Rainfall_mm": rainfall,
    "Temperature_C": temperature,
    "Humidity_%": humidity,
    "Irrigation_enc": 1 if irrigation == "Yes" else 0,
    "Season_enc": {"Kharif": 0, "Rabi": 1, "Zaid": 2}[season],
    "Landholding_ha": landholding,
    "Soil_Fertility": nitrogen + phosphorus + potassium  # derived feature
}])

# ---------------- Scale input ----------------
input_scaled = scaler.transform(input_data)

# ---------------- Predict ----------------
if st.button("Recommended Crop"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🌾 Recommended Crop: **{prediction}**")


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
