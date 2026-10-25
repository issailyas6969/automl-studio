"""AutoML engine: validate -> preprocess -> tune many models -> rank -> save."""
import json, time, warnings
from pathlib import Path
import joblib, mlflow, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
MAX_ROWS = 50000
LABELS = {"linear": "Linear / Logistic", "knn": "k-Nearest Neighbors", "tree": "Decision Tree",
          "random_forest": "Random Forest", "extra_trees": "Extra Trees", "gboost": "Gradient Boosting",
          "hist_gboost": "Hist Gradient Boosting", "svm": "Support Vector Machine",
          "xgboost": "XGBoost", "lightgbm": "LightGBM"}
DEFAULT_OFF = {"svm", "knn", "tree", "gboost"}


class Bundle:
    """Pipeline + label encoder; behaves like a normal sklearn model."""
    def __init__(self, pipe, le=None): self.pipe, self.le = pipe, le
    def predict(self, X):
        p = self.pipe.predict(X)
        return self.le.inverse_transform(np.asarray(p).astype(int)) if self.le is not None else p
    def predict_proba(self, X): return self.pipe.predict_proba(X)
    @property
    def classes_(self): return self.le.classes_


def registry(task):
    c = task == "classification"
    f = lambda a, b: a if c else b
    R = {
        "linear": (f(LogisticRegression(max_iter=2000), Ridge()),
                   {"m__C": [.01, .1, 1, 10, 100]} if c else {"m__alpha": [.1, 1, 10, 100]}),
        "knn": (f(KNeighborsClassifier(), KNeighborsRegressor()),
                {"m__n_neighbors": [3, 5, 9, 15], "m__weights": ["uniform", "distance"]}),
        "tree": (f(DecisionTreeClassifier(random_state=42), DecisionTreeRegressor(random_state=42)),
                 {"m__max_depth": [3, 5, 8, None], "m__min_samples_leaf": [1, 5, 20]}),
        "random_forest": (f(RandomForestClassifier(random_state=42), RandomForestRegressor(random_state=42)),
                          {"m__n_estimators": [100, 200, 300], "m__max_depth": [None, 6, 12, 20]}),
        "extra_trees": (f(ExtraTreesClassifier(random_state=42), ExtraTreesRegressor(random_state=42)),
                        {"m__n_estimators": [100, 200, 300], "m__max_depth": [None, 6, 12, 20]}),
        "gboost": (f(GradientBoostingClassifier(random_state=42), GradientBoostingRegressor(random_state=42)),
                   {"m__n_estimators": [100, 200], "m__learning_rate": [.03, .1, .2], "m__max_depth": [2, 3, 4]}),
        "hist_gboost": (f(HistGradientBoostingClassifier(random_state=42), HistGradientBoostingRegressor(random_state=42)),
                        {"m__learning_rate": [.03, .1, .2], "m__max_depth": [None, 3, 6], "m__max_iter": [100, 200]}),
        "svm": (f(SVC(probability=True), SVR()), {"m__C": [.1, 1, 10], "m__kernel": ["rbf", "linear"]}),
    }
    space = {"m__n_estimators": [100, 200, 300], "m__learning_rate": [.03, .1, .2], "m__max_depth": [3, 5, 7]}
    try:
        from xgboost import XGBClassifier, XGBRegressor
        R["xgboost"] = (f(XGBClassifier(eval_metric="logloss", verbosity=0, random_state=42), XGBRegressor(verbosity=0, random_state=42)), space)
    except Exception: pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        R["lightgbm"] = (f(LGBMClassifier(verbose=-1, random_state=42), LGBMRegressor(verbose=-1, random_state=42)), space)
    except Exception: pass
    return R


def available():
    return [{"key": k, "label": LABELS[k], "default": k not in DEFAULT_OFF} for k in registry("classification")]


def infer_task(y):
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        if y.nunique() > 50: raise ValueError("Target has too many distinct text values; pick a different target.")
        return "classification"
    return "classification" if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20 else "regression"


def preprocessor(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    oh = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30, sparse_output=False)
    return ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", oh)]), cat)], sparse_threshold=0)


def score(task, model, X, y):
    p = model.predict(X)
    if task == "classification":
        return {"accuracy": accuracy_score(y, p), "f1": f1_score(y, p, average="weighted")}
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))), "r2": r2_score(y, p)}


def run_training(df, target, keys, n_iter, out_dir, cb=lambda k, p: None):
    log = lambda m: cb("log", m)
    if target not in df.columns: raise ValueError(f"Target '{target}' not found.")
    df = df.dropna(subset=[target]).drop_duplicates()
    if len(df) < 30: raise ValueError("Need at least 30 usable rows.")
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42); log(f"Large dataset: sampled {MAX_ROWS:,} rows.")
    y, X = df[target], df.drop(columns=[target])
    X = X.apply(lambda s: s.astype(object) if s.dtype == bool else s)
    dropped = [c for c in X.columns if X[c].nunique(dropna=False) <= 1 or
               (not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 0.9 * len(X))]
    X = X.drop(columns=dropped)
    if X.shape[1] == 0: raise ValueError("No usable feature columns left.")
    task, le = infer_task(y), None
    log(f"Task: {task} | rows: {len(X):,} | features: {X.shape[1]}" + (f" | ignored: {dropped}" if dropped else ""))
    cv = 3
    if task == "classification":
        le = LabelEncoder(); y = pd.Series(le.fit_transform(y), index=y.index)
        mn = int(y.value_counts().min())
        if len(le.classes_) < 2 or mn < 2: raise ValueError("Each class needs at least 2 rows.")
        cv = max(2, min(3, mn))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                              stratify=y if task == "classification" else None)
    reg = registry(task); keys = [k for k in keys if k in reg]
    if not keys: raise ValueError("Select at least one available model.")
    primary = "f1" if task == "classification" else "rmse"
    scoring = "f1_weighted" if task == "classification" else "neg_root_mean_squared_error"
    mlflow.set_tracking_uri("sqlite:///mlflow.db"); mlflow.set_experiment("automl-studio")
    board, fitted = [], {}
    for i, k in enumerate(keys):
        est, space = reg[k]; t0 = time.time()
        if k in ("svm", "knn") and len(X_tr) > 10000:
            log(f"{LABELS[k]}: skipped (too slow above 10k rows)")
        else:
            log(f"Training {LABELS[k]}...")
            try:
                with mlflow.start_run(run_name=k):
                    pipe = Pipeline([("prep", preprocessor(X_tr)), ("m", est)])
                    n = min(n_iter, int(np.prod([len(v) for v in space.values()])))
                    s = RandomizedSearchCV(pipe, space, n_iter=n, cv=cv, scoring=scoring,
                                           random_state=42, n_jobs=-1).fit(X_tr, y_tr)
                    m = score(task, s.best_estimator_, X_te, y_te)
                    mlflow.log_params(s.best_params_); mlflow.log_metrics({a: float(b) for a, b in m.items()})
                fitted[k] = s.best_estimator_
                board.append({"key": k, "label": LABELS[k], "cv": round(abs(float(s.best_score_)), 4),
                              "metrics": {a: round(float(b), 4) for a, b in m.items()}, "secs": round(time.time() - t0, 1)})
                board.sort(key=lambda b: b["metrics"][primary], reverse=task == "classification")
                cb("board", list(board))
            except Exception as e:
                log(f"{LABELS[k]} failed: {str(e)[:150]}")
        cb("progress", (i + 1) / len(keys))
    if not board: raise ValueError("All models failed - see the log.")
    best = board[0]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(Bundle(fitted[best["key"]], le), out / "model.joblib")
    meta = {"task": task, "target": target, "best_model": best["label"], "metrics": best["metrics"], "board": board,
            "features": {c: str(X[c].dtype) for c in X.columns}, "dropped": dropped,
            "example": json.loads(X.iloc[[0]].to_json(orient="records"))[0],
            "choices": {c: sorted(map(str, X[c].dropna().unique()))[:30] for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])},
            "classes": [c.item() if hasattr(c, "item") else c for c in le.classes_] if le is not None else None}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"Done. Best: {best['label']} {best['metrics']}")
    return meta
