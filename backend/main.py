import os
import joblib
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------
# FastAPI App Setup
# ----------------------------
app = FastAPI(title="Carbon Credits Predictor")

# Enable CORS for any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request Model
# ----------------------------
class SolarRequest(BaseModel):
    lat: float
    lon: float
    area: float
    eff: float
    days: int

# ----------------------------
# Load ML Model Safely with Joblib
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")  # place model.pkl in same folder as main.py

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Model Error: {e}")
    model = None

# ----------------------------
# Helper Functions
# ----------------------------
def get_irradiance(lat, lon):
    try:
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}"
            f"&start=20240101&end=20240101&format=JSON"
        )
        response = requests.get(url, timeout=5)
        return list(response.json()['properties']['parameter']['ALLSKY_SFC_SW_DWN'].values())[0]
    except Exception:
        return 5.25  # fallback average irradiance

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/api/calculate")
async def calculate_credits(data: SolarRequest):
    # 1️⃣ Fetch Irradiance
    irradiance = get_irradiance(data.lat, data.lon)
    
    # 2️⃣ Predict Grid Emission Factor (Z) in kg/kWh
    if model:
        features = pd.DataFrame([{
            'Latitude': data.lat,
            'Longitude': data.lon,
            'State Name': 'Default',
            'Region': 'Default',
            'Tech_Clean': 'Subcritical',
            'Age': 15,
            'Coal_Intensity': 0.8,
            'Import_Ratio': 0.1,
            'Load_Factor': 0.7
        }])
        z_factor = model.predict(features)[0]
    else:
        z_factor = 0.82  # fallback average

    # 3️⃣ Solar Generation (kWh)
    daily_kwh = irradiance * data.area * data.eff * 0.75
    total_kwh = daily_kwh * data.days
    
    # 4️⃣ Carbon Credit Conversion
    co2_avoided_kg = total_kwh * z_factor
    carbon_credits = co2_avoided_kg / 1000  # metric tonnes

    return {
        "total_yield_kwh": round(total_kwh, 2),
        "co2_avoided_kg": round(co2_avoided_kg, 2),
        "carbon_credits": round(carbon_credits, 6),
        "z_factor": round(z_factor, 3),
        "days": data.days
    }

# ----------------------------
# Run Locally
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)