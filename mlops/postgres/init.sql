-- Tables used by the Streamlit app to record dataset/model info,
-- and by Grafana to build dashboards from.
--
-- IMPORTANT: these live in a separate "app" schema, NOT "public", because
-- MLflow's own internal tracking/registry tables (created automatically by
-- the MLflow server) already include tables named "model_versions" etc. in
-- the "public" schema. Using a separate schema avoids any name collisions.

CREATE SCHEMA IF NOT EXISTS app;

CREATE TABLE IF NOT EXISTS app.dataset_versions (
    id SERIAL PRIMARY KEY,
    version_name TEXT NOT NULL,          -- e.g. 'v1_original', 'v2_drifted'
    num_samples INTEGER NOT NULL,
    positive_ratio REAL NOT NULL,        -- proportion of positive labels (0-1)
    avg_review_length REAL NOT NULL,     -- avg. number of words per review
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS app.model_versions (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,            -- e.g. 'sentiment-classifier'
    version_name TEXT NOT NULL,          -- e.g. 'v1_original', 'v2_drifted'
    trained_on_dataset TEXT NOT NULL,    -- FK-ish reference to app.dataset_versions.version_name
    accuracy REAL NOT NULL,
    f1_score REAL NOT NULL,
    mlflow_run_id TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS app.drift_metrics (
    id SERIAL PRIMARY KEY,
    dataset_version TEXT NOT NULL,
    metric_name TEXT NOT NULL,           -- e.g. 'positive_ratio', 'avg_review_length'
    metric_value REAL NOT NULL,
    baseline_value REAL NOT NULL,        -- the v1 value, for comparison
    drift_pct REAL NOT NULL,             -- percentage change vs baseline
    recorded_at TIMESTAMP DEFAULT NOW()
);
