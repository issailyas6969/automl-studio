"""Auto-ML pipeline: validate -> preprocess -> train/tune -> evaluate -> register -> save.

Usage:
    python -m app.train --data data/train.csv [--target label]
"""
import argparse, json, warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
MODEL_DIR = Path("model")


# ---------- 1. Validate & infer task ----------
def validate(df: pd.DataFrame, target: str) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in columns: {list(df.columns)}")
    if df[target].isna().any():
        raise ValueError("Target column contains missing values; clean it first.")
    if len(df) < 30:
        raise ValueError("Need at least 30 rows to train and evaluate.")
    return {
        "rows": len(df), "cols": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_pct": df.isna().mean().round(3).to_dict(),
        "constant_cols": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
    }


def infer_task(y: pd.Series) -> str:
    if y.dtype == object or y.dtype.name == "category" or y.dtype == bool:
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20:
        return "classification"
    return "regression"


# ---------- 2. Preprocess ----------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


# ---------- 3. Candidate models + search spaces ----------
def candidates(task: str):
    if task == "classification":
        return {
            "logreg": (LogisticRegression(max_iter=1000), {"m__C": [0.01, 0.1, 1, 10]}),
            "random_forest": (RandomForestClassifier(random_state=42),
                              {"m__n_estimators": [100, 200, 300], "m__max_depth": [None, 5, 10, 20]}),
            "gboost": (GradientBoostingClassifier(random_state=42),
                       {"m__n_estimators": [100, 200], "m__learning_rate": [0.03, 0.1, 0.2],
                        "m__max_depth": [2, 3, 4]}),
        }
    return {
        "ridge": (Ridge(), {"m__alpha": [0.1, 1, 10, 100]}),
        "random_forest": (RandomForestRegressor(random_state=42),
                          {"m__n_estimators": [100, 200, 300], "m__max_depth": [None, 5, 10, 20]}),
        "gboost": (GradientBoostingRegressor(random_state=42),
                   {"m__n_estimators": [100, 200], "m__learning_rate": [0.03, 0.1, 0.2],
                    "m__max_depth": [2, 3, 4]}),
    }


def evaluate(task, model, X_te, y_te) -> dict:
    pred = model.predict(X_te)
    if task == "classification":
        return {"accuracy": accuracy_score(y_te, pred),
                "f1": f1_score(y_te, pred, average="weighted")}
    return {"rmse": float(np.sqrt(mean_squared_error(y_te, pred))), "r2": r2_score(y_te, pred)}


def main(args):
    df = pd.read_csv(args.data)
    target = args.target or df.columns[-1]          # default: last column
    report = validate(df, target)
    df = df.drop_duplicates()
    df = df.drop(columns=[c for c in report["constant_cols"] if c != target])

    X, y = df.drop(columns=[target]), df[target]
    task = infer_task(y)
    print(f"Task: {task} | target: {target} | rows: {len(df)} | features: {X.shape[1]}")

    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    primary = "f1" if task == "classification" else "rmse"
    scoring = "f1_weighted" if task == "classification" else "neg_root_mean_squared_error"

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    best = {"score": None, "name": None, "pipe": None, "metrics": None}
    for name, (est, space) in candidates(task).items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("prep", build_preprocessor(X_tr)), ("m", est)])
            n_iter = min(args.n_iter, int(np.prod([len(v) for v in space.values()])))
            search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=3,
                                        scoring=scoring, random_state=42, n_jobs=-1)
            search.fit(X_tr, y_tr)
            metrics = evaluate(task, search.best_estimator_, X_te, y_te)
            mlflow.log_params(search.best_params_)
            mlflow.log_param("task", task)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.sklearn.log_model(search.best_estimator_, "model", serialization_format="cloudpickle")
            print(f"  {name:14s} { {k: round(float(v), 4) for k, v in metrics.items()} }")
            score = metrics[primary]
            better = best["score"] is None or (score > best["score"] if task == "classification" else score < best["score"])
            if better:
                best.update(score=score, name=name, pipe=search.best_estimator_, metrics=metrics)

    # ---------- 4. Save winner + metadata ----------
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["pipe"], MODEL_DIR / "model.joblib")
    meta = {
        "task": task, "target": target, "best_model": best["name"],
        "metrics": {k: float(v) for k, v in best["metrics"].items()},
        "features": {c: str(X[c].dtype) for c in X.columns},
        "example": X.iloc[0].replace({np.nan: None}).to_dict(),
        "classes": [str(c) for c in sorted(y.unique())] if task == "classification" else None,
        "validation": report,
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nBest: {best['name']} -> {meta['metrics']}\nSaved to {MODEL_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", default=None, help="Target column (default: last column)")
    p.add_argument("--experiment", default="automl")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--n-iter", type=int, default=6)
    main(p.parse_args())
