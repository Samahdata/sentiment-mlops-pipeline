"""
Streamlit app: MLflow tracking + registry UI
Streamlit app with:
  - Dataset version selector
  - Model version selector
  - Prediction interface
  - Metrics visualization
  - Model comparison charts
"""

import os
import re
import string
import pickle

import streamlit as st
import pandas as pd
import psycopg2
import mlflow
import plotly.graph_objects as go

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

PG_CONFIG = dict(
    host=os.environ.get("POSTGRES_HOST", "postgres"),
    dbname=os.environ.get("POSTGRES_DB", "mlflow"),
    user=os.environ.get("POSTGRES_USER", "mlflow"),
    password=os.environ.get("POSTGRES_PASSWORD", "mlflow"),
)

st.set_page_config(page_title="Sentiment Model Versioning", page_icon="🎬", layout="wide")


@st.cache_data(ttl=30)
def load_dataset_versions():
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(
        "SELECT * FROM app.dataset_versions ORDER BY created_at DESC", conn
    )
    conn.close()
    return df


@st.cache_data(ttl=30)
def load_model_versions():
    conn = psycopg2.connect(**PG_CONFIG)
    df = pd.read_sql(
        "SELECT * FROM app.model_versions ORDER BY created_at DESC", conn
    )
    conn.close()
    return df


@st.cache_resource
def load_model_bundle(version_name: str):
    path = f"models/model_{version_name}.pkl"
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def preprocess_with_bundle(bundle, text: str) -> str:
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    namespace = {
        "re": re, "string": string, "word_tokenize": word_tokenize,
        "stop_words": stop_words, "lemmatizer": lemmatizer,
    }
    exec(bundle["preprocessing_source"], namespace)
    return namespace["preprocess"](text)


st.title("🎬 Sentiment Classifier — Dataset & Model Versioning")
st.caption("MLflow tracking + registry, backed by Postgres, visualized in Grafana")

dataset_versions = load_dataset_versions()
model_versions = load_model_versions()

if dataset_versions.empty or model_versions.empty:
    st.warning(
        "No versions found yet. Run `docker compose exec app python versioning.py` "
        "once to create dataset v1/v2, train both models, and populate MLflow + Postgres."
    )
    st.stop()

# ---------------- Sidebar: selectors ----------------
st.sidebar.header("Version selectors")
dataset_choice = st.sidebar.selectbox(
    "Dataset version", dataset_versions["version_name"].tolist()
)
model_choice = st.sidebar.selectbox(
    "Model version", model_versions["version_name"].tolist()
)

st.sidebar.markdown(f"[Open MLflow UI]({MLFLOW_TRACKING_URI})")

# ---------------- Tabs ----------------
tab_predict, tab_metrics, tab_compare = st.tabs(
    ["🔮 Prediction", "📊 Metrics", "⚖️ Model Comparison"]
)

with tab_predict:
    st.subheader(f"Try the model — version: {model_choice}")
    review = st.text_area("Enter a movie review", placeholder="This movie was absolutely fantastic!")
    if st.button("Predict Sentiment"):
        bundle = load_model_bundle(model_choice)
        cleaned = preprocess_with_bundle(bundle, review)
        features = bundle["vectorizer"].transform([cleaned])
        pred = bundle["model"].predict(features)[0]
        proba = bundle["model"].predict_proba(features)[0]
        confidence = max(proba)
        label = "Positive 😊" if pred == 1 else "Negative 😞"
        st.success(f"**{label}** — confidence: {confidence:.1%}")

    st.divider()
    st.subheader(f"Dataset preview — version: {dataset_choice}")
    row = dataset_versions[dataset_versions["version_name"] == dataset_choice].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", f"{int(row['num_samples']):,}")
    c2.metric("Positive ratio", f"{row['positive_ratio']:.1%}")
    c3.metric("Avg. review length", f"{row['avg_review_length']:.0f} words")
    st.caption(row["description"])

with tab_metrics:
    st.subheader("Model performance by version")
    st.dataframe(
        model_versions[["version_name", "trained_on_dataset", "accuracy", "f1_score", "created_at"]],
        use_container_width=True,
    )

    fig = go.Figure()
    fig.add_bar(name="Accuracy", x=model_versions["version_name"], y=model_versions["accuracy"])
    fig.add_bar(name="F1 score", x=model_versions["version_name"], y=model_versions["f1_score"])
    fig.update_layout(barmode="group", title="Accuracy & F1 by model version", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with tab_compare:
    st.subheader("v1 (original) vs v2 (drifted) — dataset comparison")
    if len(dataset_versions) >= 2:
        v1_row = dataset_versions[dataset_versions["version_name"] == "v1_original"].iloc[0]
        v2_row = dataset_versions[dataset_versions["version_name"] == "v2_drifted"].iloc[0]

        comp_fig = go.Figure()
        comp_fig.add_bar(
            name="v1_original",
            x=["Positive ratio", "Avg. review length (÷100)"],
            y=[v1_row["positive_ratio"], v1_row["avg_review_length"] / 100],
        )
        comp_fig.add_bar(
            name="v2_drifted",
            x=["Positive ratio", "Avg. review length (÷100)"],
            y=[v2_row["positive_ratio"], v2_row["avg_review_length"] / 100],
        )
        comp_fig.update_layout(barmode="group", title="Dataset drift: v1 vs v2")
        st.plotly_chart(comp_fig, use_container_width=True)

        st.info(
            f"**Drift summary:** positive ratio shifted from {v1_row['positive_ratio']:.1%} "
            f"to {v2_row['positive_ratio']:.1%}, and average review length shifted from "
            f"{v1_row['avg_review_length']:.0f} to {v2_row['avg_review_length']:.0f} words."
        )

    st.subheader("Model accuracy: v1 vs v2")
    if len(model_versions) >= 2:
        acc_fig = go.Figure()
        acc_fig.add_bar(x=model_versions["version_name"], y=model_versions["accuracy"])
        acc_fig.update_layout(title="Accuracy comparison", yaxis_range=[0, 1])
        st.plotly_chart(acc_fig, use_container_width=True)
