
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("xgb_boxoffice.pkl")

class MovieInput(BaseModel):
    domestic_lifetime_gross: float
    domestic_percentage: float
    foreign_lifetime_gross: float
    foreign_percentage: float
    year: int

@app.get("/")
def home():
    return {"status": "API is live"}

@app.post("/predict")
def predict(data: MovieInput):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]

    return {"prediction": round(float(pred), 2)}
