# AutoML Pipeline (MVP)

CSV in -> validate -> preprocess -> train/tune 3 models -> pick best -> MLflow log -> FastAPI + Docker.

## Run
    pip install -r requirements.txt
    python -m app.train --data data/your.csv --target your_target_column
    mlflow ui --backend-store-uri sqlite:///mlflow.db      # compare runs at :5000
    uvicorn app.serve:app --port 8000                       # docs at :8000/docs

## Docker (after training, so ./model exists)
    docker build -t automl-api . && docker run -p 8000:8000 automl-api

## Predict
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
      -d '{"rows":[{...copy "example" from GET /schema...}]}'

## Next steps
Streamlit upload UI, Evidently drift reports, Great Expectations checks, XGBoost/LightGBM,
Airflow/ZenML scheduling, retrain gate (only promote if new model beats the current one).

## Web UI (AutoML Studio)
    uvicorn app.server:app --port 8000     # open http://localhost:8000
Upload CSV -> choose target + models (linear, kNN, tree, RF, ExtraTrees, GBM, HistGBM, SVM, XGBoost, LightGBM)
-> live leaderboard -> predict via form, batch CSV, or REST (`/api/predict/{job}`). One training job runs at a time.
