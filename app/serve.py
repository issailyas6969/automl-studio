"""Prediction API. Run: uvicorn app.serve:app --port 8000"""
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = Path("model")
model = joblib.load(MODEL_DIR / "model.joblib")
meta = json.loads((MODEL_DIR / "meta.json").read_text())

app = FastAPI(title="AutoML Prediction API")


class Batch(BaseModel):
    rows: List[Dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok", "model": meta["best_model"], "task": meta["task"]}


@app.get("/schema")
def schema():
    return {"features": meta["features"], "example": meta["example"], "metrics": meta["metrics"]}


@app.post("/predict")
def predict(batch: Batch):
    df = pd.DataFrame(batch.rows)
    missing = set(meta["features"]) - set(df.columns)
    if missing:
        raise HTTPException(422, f"Missing features: {sorted(missing)}")
    try:
        df = df[list(meta["features"])]
        out = {"predictions": model.predict(df).tolist()}
        if meta["task"] == "classification" and hasattr(model, "predict_proba"):
            out["probabilities"] = model.predict_proba(df).round(4).tolist()
            out["classes"] = [str(c) for c in model.classes_]
        return out
    except Exception as e:
        raise HTTPException(400, str(e))
