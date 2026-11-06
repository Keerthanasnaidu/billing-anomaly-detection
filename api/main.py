
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Locate the project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "random_forest.pkl"
FEATURES_PATH = PROJECT_DIR / "data" / "X_train.csv"

# Load the trained model and expected feature names
model = joblib.load(MODEL_PATH)
feature_columns = pd.read_csv(FEATURES_PATH, nrows=1).columns.tolist()

# Create the API
app = FastAPI(
    title="Billing Anomaly Detection API",
    description="Predicts whether a utility bill is normal or anomalous.",
    version="1.0.0",
)


class BillingRecord(BaseModel):
    features: dict[str, float]


@app.get("/")
def home():
    return {
        "message": "Billing Anomaly Detection API is running!",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "random_forest",
    }


@app.post("/predict")
def predict(record: BillingRecord):
    supplied_features = record.features

    # Check for missing or unexpected features
    missing_features = [
        column
        for column in feature_columns
        if column not in supplied_features
    ]

    extra_features = [
        column
        for column in supplied_features
        if column not in feature_columns
    ]

    if missing_features or extra_features:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing_features,
                "unexpected_features": extra_features,
            },
        )

    # Arrange features in the exact order expected by the model
    input_data = pd.DataFrame(
        [[supplied_features[column] for column in feature_columns]],
        columns=feature_columns,
    )

    # Predict whether the bill is anomalous
    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    anomaly_probability = float(probabilities[1])

    return {
        "prediction": prediction,
        "result": "Anomaly" if prediction == 1 else "Normal",
        "anomaly_probability": round(anomaly_probability, 4),
    }

