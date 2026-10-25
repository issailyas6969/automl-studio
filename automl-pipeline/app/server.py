"""AutoML Studio: web UI + REST API.  Run: uvicorn app.server:app --port 8000"""
import io, json, threading, uuid
from pathlib import Path
from typing import Any, Dict, List

import joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from . import engine

ROOT = Path("workspace"); (ROOT / "uploads").mkdir(parents=True, exist_ok=True)
JOBS, CACHE = {}, {}
app = FastAPI(title="AutoML Studio")


@app.get("/")
def index(): return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/upload")
async def upload(request: Request, name: str = "data.csv"):
    raw = await request.body()
    try: df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception as e: raise HTTPException(400, f"Could not read file as CSV: {e}")
    if df.shape[1] < 2 or len(df) < 2: raise HTTPException(400, "Need at least 2 columns and 2 rows.")
    uid = uuid.uuid4().hex[:8]; df.to_csv(ROOT / "uploads" / f"{uid}.csv", index=False)
    cols = [{"name": str(c), "dtype": str(df[c].dtype), "missing": round(float(df[c].isna().mean()), 3),
             "unique": int(df[c].nunique())} for c in df.columns]
    return {"id": uid, "name": name, "rows": len(df), "cols": cols, "suggested_target": str(df.columns[-1]),
            "preview": json.loads(df.head(6).to_json(orient="records")), "models": engine.available()}


class TrainReq(BaseModel):
    id: str; target: str; models: List[str]; n_iter: int = 6


@app.post("/api/train")
def train(r: TrainReq):
    if any(j["status"] == "running" for j in JOBS.values()): raise HTTPException(409, "A training job is already running.")
    path = ROOT / "uploads" / f"{r.id}.csv"
    if not path.exists(): raise HTTPException(404, "Dataset not found; upload again.")
    jid = uuid.uuid4().hex[:8]
    job = JOBS[jid] = {"status": "running", "log": [], "board": [], "progress": 0, "result": None, "error": None}

    def cb(kind, p):
        if kind == "log": job["log"].append(p)
        else: job[kind] = p

    def work():
        try:
            job["result"] = engine.run_training(pd.read_csv(path), r.target, r.models, max(2, min(r.n_iter, 30)),
                                                ROOT / "models" / jid, cb)
            job["status"] = "done"
        except Exception as e:
            job["status"], job["error"] = "error", str(e)
    threading.Thread(target=work, daemon=True).start()
    return {"job": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS: raise HTTPException(404, "Unknown job")
    return JOBS[jid]


def load(jid):
    if jid not in CACHE:
        d = ROOT / "models" / jid
        if not (d / "model.joblib").exists(): raise HTTPException(404, "Model not found")
        CACHE[jid] = (joblib.load(d / "model.joblib"), json.loads((d / "meta.json").read_text()))
    return CACHE[jid]


def prep(df, meta):
    miss = set(meta["features"]) - set(df.columns)
    if miss: raise HTTPException(422, f"Missing columns: {sorted(miss)}")
    df = df[list(meta["features"])].replace({None: np.nan})
    for c, t in meta["features"].items():
        if "int" in t or "float" in t: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


class Rows(BaseModel):
    rows: List[Dict[str, Any]]


@app.post("/api/predict/{jid}")
def predict(jid: str, b: Rows):
    model, meta = load(jid); df = prep(pd.DataFrame(b.rows), meta)
    out = {"predictions": np.asarray(model.predict(df)).tolist()}
    if meta["task"] == "classification" and hasattr(model, "predict_proba"):
        out["probabilities"] = model.predict_proba(df).round(4).tolist(); out["classes"] = [str(c) for c in model.classes_]
    return out


@app.post("/api/predict_csv/{jid}")
async def predict_csv(jid: str, request: Request):
    model, meta = load(jid)
    try: raw = pd.read_csv(io.BytesIO(await request.body()), sep=None, engine="python")
    except Exception as e: raise HTTPException(400, f"Bad CSV: {e}")
    raw["prediction"] = np.asarray(model.predict(prep(raw.copy(), meta)))
    return Response(raw.to_csv(index=False), media_type="text/csv")


@app.get("/api/download/{jid}")
def download(jid: str):
    load(jid); return FileResponse(ROOT / "models" / jid / "model.joblib", filename="model.joblib")
