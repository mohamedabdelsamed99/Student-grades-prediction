# app.py
from fastapi import FastAPI
import joblib, json
import pandas as pd

# --- Load Artifacts ---
model = joblib.load("artifacts/model.joblib")
scaler = joblib.load("artifacts/scaler.joblib")

with open("artifacts/feature_columns.json", "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)

with open("artifacts/numeric_cols.json", "r", encoding="utf-8") as f:
    NUMERIC_COLS = json.load(f)

with open("artifacts/risk_labels.json", "r", encoding="utf-8") as f:
    RISK_LABELS = json.load(f)

# --- FastAPI app ---
app = FastAPI(title="Student Risk Prediction API (Minimal)")

def align_features(row: dict):
    """Ensure incoming data matches training feature order."""
    df = pd.DataFrame([row])
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0
    df = df[FEATURE_COLS]
    # scale numeric cols
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df

@app.post("/predict")
def predict(features: dict):
    X = align_features(features)
    preds = model.predict(X)
    proba = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
    idx = int(preds[0])
    return {
        "predicted_class_index": idx,
        "predicted_label": RISK_LABELS.get(str(idx), str(idx)),
        "probabilities": proba
    }
