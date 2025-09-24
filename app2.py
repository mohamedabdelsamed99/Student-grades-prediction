# app2.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib, json, pandas as pd

# --- Load Artifacts ---
model = joblib.load("artifacts/model.joblib")
scaler = joblib.load("artifacts/scaler.joblib")

with open("artifacts/feature_columns.json", "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)

with open("artifacts/numeric_cols.json", "r", encoding="utf-8") as f:
    NUMERIC_COLS = json.load(f)

# Risk labels mapping
RISK_LABELS = {
    "0": "High Risk",
    "1": "Medium Risk",
    "2": "Low Risk / Safe"
}

# --- Load test data for accuracy (if available) ---
try:
    test_df = pd.read_csv("artifacts/test_data.csv")  # لازم يكون فيه target
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"]
    test_accuracy = model.score(X_test, y_test)
except Exception as e:
    test_accuracy = None
    print("⚠️ No test data found, accuracy will not be shown")

# --- FastAPI app ---
app = FastAPI(title="Student Risk Prediction API (Raw Input)")

# --- Schema for raw student input ---
class StudentRawIn(BaseModel):
    school: Optional[str] = "GP"
    sex: Optional[str] = "F"
    age: Optional[int] = 16
    address: Optional[str] = "U"
    family_size: Optional[str] = "GT3"
    parent_status: Optional[str] = "T"
    mother_education: Optional[int] = 2
    father_education: Optional[int] = 2
    mother_job: Optional[str] = "other"
    father_job: Optional[str] = "other"
    school_reason: Optional[str] = "reputation"
    guardian_type: Optional[str] = "mother"
    travel_time: Optional[int] = 1
    study_time: Optional[int] = 2
    past_failures: Optional[int] = 0
    school_support: Optional[int] = 0
    family_support: Optional[int] = 0
    paid_classes: Optional[int] = 0
    extra_activities: Optional[int] = 0
    wants_higher_ed: Optional[int] = 1
    internet: Optional[int] = 1
    romantic: Optional[int] = 0
    family_relationship: Optional[int] = 4
    free_time: Optional[int] = 3
    social_outings: Optional[int] = 3
    workday_alcohol: Optional[int] = 1
    weekend_alcohol: Optional[int] = 2
    health_status: Optional[int] = 4
    absences: Optional[int] = 3
    grade_period1: Optional[int] = 12
    grade_period2: Optional[int] = 13

# --- Mapping for binary categorical values ---
BINARY_MAP = {
    'school': {'GP': 0, 'MS': 1},
    'sex': {'F': 0, 'M': 1},
    'address': {'U': 0, 'R': 1},
    'family_size': {'LE3': 0, 'GT3': 1},
    'parent_status': {'T': 0, 'A': 1}
}

MULTI_CLASS_COLS = ['mother_job', 'father_job', 'school_reason', 'guardian_type']

def preprocess_raw(s: StudentRawIn) -> dict:
    row = pd.DataFrame([s.dict()])

    # apply binary maps
    for col, mapping in BINARY_MAP.items():
        row[col] = row[col].map(mapping).astype(int)

    # engineered features
    row["social_index"] = row["social_outings"] + row["extra_activities"] + row["romantic"]
    row["avg_alcohol"] = (row["workday_alcohol"] + row["weekend_alcohol"]) / 2.0
    row["free_social_ratio"] = row["free_time"] / (row["social_outings"] + 1)

    # one-hot encode multi-class cols
    row = pd.get_dummies(row, columns=MULTI_CLASS_COLS, drop_first=True)

    # ensure all training cols exist
    for c in FEATURE_COLS:
        if c not in row.columns:
            row[c] = 0

    # final order
    row = row[FEATURE_COLS]

    # scale numeric cols
    for c in NUMERIC_COLS:
        row[c] = pd.to_numeric(row[c], errors="coerce").fillna(0)
    row[NUMERIC_COLS] = scaler.transform(row[NUMERIC_COLS])

    return row.iloc[0].to_dict()

@app.post("/predict_raw")
def predict_raw(payload: StudentRawIn):
    processed = preprocess_raw(payload)
    X = pd.DataFrame([processed])[FEATURE_COLS]
    preds = model.predict(X)
    proba = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
    idx = int(preds[0])
    return {
        "predicted_class_index": idx,
        "predicted_label": RISK_LABELS.get(str(idx), f"Class {idx}"),
        "probabilities": proba,
        "model_test_accuracy": test_accuracy
    }
