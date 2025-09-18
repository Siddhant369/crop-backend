from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model + encoders
model = joblib.load("crop_model.pkl")
district_encoder = joblib.load("district_encoder.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")

# Load dataset for stats
df = pd.read_csv("crop_dataset.csv")
feature_stats = {
    "Nitrogen": {"mean": df["Nitrogen"].mean()},
    "Phosphorus": {"mean": df["Phosphorus"].mean()},
    "Potassium": {"mean": df["Potassium"].mean()},
    "pH": {"mean": df["pH"].mean()},
    "Rainfall": {"mean": df["Rainfall"].mean()},
    "Temperature": {"mean": df["Temperature"].mean()},
}

app = FastAPI()

# ✅ Pydantic model for input validation
class CropInput(BaseModel):
    district: str = ""
    soil_color: str = ""
    Nitrogen: float = feature_stats["Nitrogen"]["mean"]
    Phosphorus: float = feature_stats["Phosphorus"]["mean"]
    Potassium: float = feature_stats["Potassium"]["mean"]
    pH: float = feature_stats["pH"]["mean"]
    Rainfall: float = feature_stats["Rainfall"]["mean"]
    Temperature: float = feature_stats["Temperature"]["mean"]

@app.post("/predict")
def predict_crop(data: CropInput):
    # Encode categorical safely
    try:
        district_enc = district_encoder.transform([data.district])[0]
    except:
        district_enc = 0
    try:
        soil_enc = soil_encoder.transform([data.soil_color])[0]
    except:
        soil_enc = 0

    features = [
        district_enc,
        soil_enc,
        data.Nitrogen,
        data.Phosphorus,
        data.Potassium,
        data.pH,
        data.Rainfall,
        data.Temperature,
    ]

    pred = model.predict([features])
    crop_name = crop_encoder.inverse_transform(pred)[0]

    return {"predicted_crop": crop_name}

