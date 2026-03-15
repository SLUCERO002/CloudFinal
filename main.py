import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

FEATURES = [
    "actual_distance_miles",
    "actual_duration_hours",
    "idle_time_hours",
    "dispatch_dayofweek",
    "dispatch_month",
]
LABEL = "fuel_gallons_used"

df = pd.read_csv("training_data.csv")

X = df[FEATURES]
y = df[LABEL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestRegressor(n_estimators=150, random_state=42))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = float(mean_absolute_error(y_test, preds))

class TripFeatures(BaseModel):
    actual_distance_miles: float
    actual_duration_hours: float
    idle_time_hours: float
    dispatch_dayofweek: int
    dispatch_month: int

@app.get("/")
def home():
    return {"service": "CloudFinal ML Microservice", "status": "running", "model_mae_gallons": mae}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(features: TripFeatures):
    X_in = [[
        features.actual_distance_miles,
        features.actual_duration_hours,
        features.idle_time_hours,
        features.dispatch_dayofweek,
        features.dispatch_month,
    ]]
    y_hat = float(model.predict(X_in)[0])
    return {"predicted_fuel_gallons_used": y_hat}
