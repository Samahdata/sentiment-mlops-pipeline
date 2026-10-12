# Dataset/Model Versioning & Drift Dashboards

Full MLflow + Grafana monitoring stack for the sentiment analysis project.

## What this does

- Builds **two dataset versions**:
  - `v1_original`: a balanced random sample of IMDB reviews (50/50 pos/neg)
  - `v2_drifted`: a **drifted** sample — imbalanced (80% negative / 20%
    positive) and biased toward shorter reviews, simulating real-world
    data drift over time
- Trains a **Logistic Regression + TF-IDF** model on each dataset version
- Logs both datasets and both model versions to **MLflow** (Data Registry +
  Model Registry), including metrics (accuracy, F1) for comparison
- Writes dataset stats, model metrics, and drift metrics to **Postgres**
- Visualizes everything in a pre-built **Grafana** dashboard
- Provides a **Streamlit app** with a dataset/model version selector, a live
  prediction interface, metrics charts, and a v1-vs-v2 comparison view

## Architecture

```
docker-compose.yml
├── postgres   → shared metrics store (dataset_versions, model_versions, drift_metrics tables)
├── mlflow     → MLflow tracking server (UI at :5000), backed by Postgres
├── app        → Streamlit app (UI at :8501) — also runs versioning.py to populate everything
└── grafana    → Dashboard (UI at :3000, login admin/admin) reading from Postgres
```

## How to run

1. Make sure Docker Desktop is running.
2. From this folder, start everything:
   ```
   docker compose up --build -d
   ```
   Wait about a minute for all services to become healthy.
3. Run the versioning script **once** to create both dataset versions, train
   both models, and populate MLflow + Postgres + Grafana:
   ```
   docker compose exec app python versioning.py
   ```
   This takes a few minutes (downloads IMDB, trains 2 models).
4. Open:
   - **MLflow UI**: http://localhost:5000 — see both runs under the
     `sentiment-analysis-versioning` experiment, and both versions of the
     `sentiment-classifier` registered model.
   - **Grafana**: http://localhost:3000 (admin/admin) — dashboard
     "Sentiment Model Versioning & Drift Dashboard" is auto-loaded.
   - **Streamlit app**: http://localhost:8501 — select a dataset/model
     version, try predictions, and view comparison charts.

## Files

- `docker-compose.yml` — orchestrates all 4 services
- `mlflow_server/` — Dockerfile for the MLflow tracking server
- `mlflow_app/versioning.py` — builds dataset v1/v2, trains + logs both models
- `mlflow_app/app.py` — Streamlit app (selectors, prediction, charts)
- `postgres/init.sql` — schema for dataset/model/drift metrics tables
- `grafana/provisioning/` — auto-configured datasource + dashboard JSON
