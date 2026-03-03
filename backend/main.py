import pickle
import pandas as pd
import requests
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React build
app.mount("/", StaticFiles(directory="dist", html=True), name="static")

class SolarRequest(BaseModel):
    lat: float
    lon: float
    area: float
    eff: float
    days: int

# Load model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Model Error: {e}")
    model = None

def get_irradiance(lat, lon):
    try:
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN&community=RE&longitude={lon}&latitude={lat}&start=20240101&end=20240101&format=JSON"
        response = requests.get(url, timeout=5)
        return list(response.json()['properties']['parameter']['ALLSKY_SFC_SW_DWN'].values())[0]
    except:
        return 5.25


@app.post("/api/calculate")
async def calculate_credits(data: SolarRequest):
    irradiance = get_irradiance(data.lat, data.lon)

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
        z_factor = 0.82

    daily_kwh = irradiance * data.area * data.eff * 0.75
    total_kwh = daily_kwh * data.days

    co2_avoided_kg = total_kwh * z_factor
    carbon_credits = co2_avoided_kg / 1000

    return {
        "total_yield_kwh": round(total_kwh, 2),
        "co2_avoided_kg": round(co2_avoided_kg, 2),
        "carbon_credits": round(carbon_credits, 6),
        "z_factor": round(z_factor, 3),
        "days": data.days
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)