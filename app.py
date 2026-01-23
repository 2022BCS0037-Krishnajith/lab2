from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
import os

app = FastAPI(
    title="Wine Quality Inference API",
    version="0.1.0"
)

# Load model
model = joblib.load("output/model.pkl")

RESULTS_PATH = "output/result.json"

# ---------- Input Schema ----------
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# ---------- Prediction ----------
@app.post("/predict")
def predict(data: WineInput):
    x = [[
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]]
    y = model.predict(x)[0]
    return {
    "name": "Krishnajith",
    "roll_no": "2022BCS0037",
    "wine_quality": int(round(y))
    }


# ---------- Metrics ----------
@app.get("/metrics")
def get_metrics():
    if not os.path.exists(RESULTS_PATH):
        raise HTTPException(status_code=404, detail="result.json not found")

    with open(RESULTS_PATH, "r") as f:
        return json.load(f)