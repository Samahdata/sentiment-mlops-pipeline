"""
Dataset & model versioning with MLflow + Postgres
Creates two dataset versions (v1 = balanced original, v2 = drifted:
class-imbalanced + shorter reviews), trains a model on each, logs both
datasets and models to MLflow, and writes dataset/model stats to Postgres
so Grafana can visualize them.

Run this INSIDE the `app` container (it has mlflow + psycopg2 installed),
e.g.:
    docker compose exec app python versioning.py
"""

import os
import re
import string
import pickle
import inspect
import logging

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import psycopg2

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("sentiment-analysis-versioning")

PG_CONFIG = dict(
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    dbname=os.environ.get("POSTGRES_DB", "mlflow"),
    user=os.environ.get("POSTGRES_USER", "mlflow"),
    password=os.environ.get("POSTGRES_PASSWORD", "mlflow"),
)

# ---------------------------------------------------------------------------
# Preprocessing (same logic as the training notebook, kept self-contained here)
# ---------------------------------------------------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> str:
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess(text: str) -> str:
    return tokenize_and_lemmatize(clean_text(text))


# ---------------------------------------------------------------------------
# Step 1 — Build dataset versions
# ---------------------------------------------------------------------------
def build_dataset_versions():
    logger.info("Downloading IMDB dataset...")
    ds = load_dataset("stanfordnlp/imdb")
    df = ds["train"].to_pandas()

    # v1: balanced random sample (the "original" distribution)
    neg_v1 = df[df["label"] == 0].sample(n=2500, random_state=42)
    pos_v1 = df[df["label"] == 1].sample(n=2500, random_state=42)
    v1 = pd.concat([neg_v1, pos_v1]).sample(frac=1, random_state=42).reset_index(drop=True)

    # v2: DRIFTED sample — class-imbalanced (80% negative / 20% positive)
    # AND biased toward shorter reviews, to simulate real-world drift
    # (e.g. a shift toward more negative, terser user feedback over time).
    df["word_count"] = df["text"].str.split().str.len()
    short_reviews = df[df["word_count"] < df["word_count"].median()]

    neg_pool = short_reviews[short_reviews["label"] == 0]
    pos_pool = short_reviews[short_reviews["label"] == 1]
    v2_neg = neg_pool.sample(n=4000, random_state=42, replace=len(neg_pool) < 4000)
    v2_pos = pos_pool.sample(n=1000, random_state=42, replace=len(pos_pool) < 1000)
    v2 = pd.concat([v2_neg, v2_pos]).sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    v1.to_csv("data/dataset_v1_original.csv", index=False)
    v2.to_csv("data/dataset_v2_drifted.csv", index=False)

    logger.info(f"v1: {len(v1)} samples, positive ratio = {v1['label'].mean():.3f}")
    logger.info(f"v2: {len(v2)} samples, positive ratio = {v2['label'].mean():.3f}")

    return v1, v2


def dataset_stats(df: pd.DataFrame) -> dict:
    word_counts = df["text"].str.split().str.len()
    return {
        "num_samples": int(len(df)),
        "positive_ratio": float(df["label"].mean()),
        "avg_review_length": float(word_counts.mean()),
    }


# ---------------------------------------------------------------------------
# Step 2 — Train a model on a given dataset version and log to MLflow
# ---------------------------------------------------------------------------
def train_and_log(df: pd.DataFrame, version_name: str, dataset_description: str):
    logger.info(f"Preprocessing text for {version_name}...")
    df = df.copy()
    df["clean_text"] = df["text"].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    with mlflow.start_run(run_name=version_name) as run:
        # --- Log the dataset as an MLflow input (Data Registry) ---
        stats = dataset_stats(df)
        dataset_obj = mlflow.data.from_pandas(
            df[["text", "label"]], name=f"imdb-{version_name}"
        )
        mlflow.log_input(dataset_obj, context="training")
        mlflow.set_tag("dataset_version", version_name)
        mlflow.set_tag("dataset_description", dataset_description)
        mlflow.log_params({
            "dataset_num_samples": stats["num_samples"],
            "dataset_positive_ratio": round(stats["positive_ratio"], 3),
            "dataset_avg_review_length": round(stats["avg_review_length"], 1),
        })

        # --- Train ---
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        hyperparams = dict(max_iter=1000, C=1.0, solver="liblinear", random_state=42)
        model = LogisticRegression(**hyperparams)
        model.fit(X_train_tfidf, y_train)

        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_params(hyperparams)
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})

        # --- Register model version in the MLflow Model Registry ---
        bundle = {
            "model": model,
            "vectorizer": tfidf,
            "preprocessing_source": (
                inspect.getsource(clean_text)
                + "\n"
                + inspect.getsource(tokenize_and_lemmatize)
                + "\n"
                + inspect.getsource(preprocess)
            ),
        }
        bundle_path = f"/tmp/model_{version_name}.pkl"
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)

        mlflow.log_artifact(bundle_path, artifact_path="model_bundle")
        mlflow.sklearn.log_model(
            model,
            artifact_path="sklearn-model",
            registered_model_name="sentiment-classifier",
        )

        logger.info(f"{version_name}: accuracy={acc:.4f}, f1={f1:.4f}, run_id={run.info.run_id}")

        os.makedirs("models", exist_ok=True)
        with open(f"models/model_{version_name}.pkl", "wb") as f:
            pickle.dump(bundle, f)

        return {
            "run_id": run.info.run_id,
            "accuracy": acc,
            "f1_score": f1,
            "stats": stats,
        }


# ---------------------------------------------------------------------------
# Step 3 — Write dataset/model stats + drift metrics to Postgres (for Grafana)
# ---------------------------------------------------------------------------
def write_to_postgres(v1_stats, v2_stats, v1_result, v2_result):
    conn = psycopg2.connect(**PG_CONFIG)
    cur = conn.cursor()

    for version_name, stats, desc in [
        ("v1_original", v1_stats, "Balanced random sample of IMDB reviews (baseline)"),
        ("v2_drifted", v2_stats, "Class-imbalanced (80/20) sample biased toward shorter reviews"),
    ]:
        cur.execute(
            """INSERT INTO app.dataset_versions
               (version_name, num_samples, positive_ratio, avg_review_length, description)
               VALUES (%s, %s, %s, %s, %s)""",
            (version_name, stats["num_samples"], stats["positive_ratio"],
             stats["avg_review_length"], desc),
        )

    for version_name, result in [("v1_original", v1_result), ("v2_drifted", v2_result)]:
        cur.execute(
            """INSERT INTO app.model_versions
               (model_name, version_name, trained_on_dataset, accuracy, f1_score, mlflow_run_id)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            ("sentiment-classifier", version_name, version_name,
             result["accuracy"], result["f1_score"], result["run_id"]),
        )

    # Drift metrics: compare v2 stats against v1 (baseline)
    for metric_name in ["positive_ratio", "avg_review_length"]:
        baseline = v1_stats[metric_name]
        current = v2_stats[metric_name]
        drift_pct = ((current - baseline) / baseline) * 100 if baseline else 0
        cur.execute(
            """INSERT INTO app.drift_metrics
               (dataset_version, metric_name, metric_value, baseline_value, drift_pct)
               VALUES (%s, %s, %s, %s, %s)""",
            ("v2_drifted", metric_name, current, baseline, drift_pct),
        )

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Wrote dataset/model/drift metrics to Postgres.")


if __name__ == "__main__":
    v1, v2 = build_dataset_versions()
    v1_stats = dataset_stats(v1)
    v2_stats = dataset_stats(v2)

    v1_result = train_and_log(v1, "v1_original", "Balanced random sample of IMDB reviews (baseline)")
    v2_result = train_and_log(v2, "v2_drifted", "Class-imbalanced (80/20) sample biased toward shorter reviews")

    write_to_postgres(v1_stats, v2_stats, v1_result, v2_result)

    print("\n=== Summary ===")
    print(f"v1_original: accuracy={v1_result['accuracy']:.4f}, f1={v1_result['f1_score']:.4f}")
    print(f"v2_drifted:  accuracy={v2_result['accuracy']:.4f}, f1={v2_result['f1_score']:.4f}")
    print("\nOpen MLflow UI at http://localhost:5000 to see both runs and registered model versions.")
    print("Open Grafana at http://localhost:3000 (admin/admin) to see the dashboard.")
