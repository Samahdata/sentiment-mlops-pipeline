"""
Deployment: FastAPI service for the sentiment model
Loads the trained sentiment model bundle (models/model.pkl)
and exposes a /predict endpoint, plus a minimal web UI at /.
"""

import os
import re
import string
import pickle
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API", description="TF-IDF + Logistic Regression sentiment classifier")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")

# ---------------------------------------------------------------------------
# Load the model, vectorizer, and preprocessing logic saved in model.pkl.
# The training notebook saved the preprocessing function *source code* alongside the model so that
# the API can apply the exact same cleaning/lemmatization steps before vectorizing
# new input text, without duplicating (and risking inconsistency with) that logic.
# ---------------------------------------------------------------------------
logger.info(f"Loading model from {MODEL_PATH} ...")
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

baseline_model = bundle["model"]
vectorizer = bundle["vectorizer"]

# Recreate the preprocessing function(s) from the saved source code.
# The saved functions reference `re`, `string`, `stop_words`, `lemmatizer`, and
# `word_tokenize`, so we provide those in the exec namespace before running the
# saved source. This requires nltk stopwords/wordnet to already be downloaded
# (done above / in the Dockerfile).
_namespace = {
    "re": re,
    "string": string,
    "word_tokenize": word_tokenize,
    "stop_words": set(stopwords.words("english")),
    "lemmatizer": WordNetLemmatizer(),
}
exec(bundle["preprocessing_source"], _namespace)
preprocess = _namespace["preprocess"]

logger.info("Model, vectorizer, and preprocessing pipeline loaded successfully.")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    prediction: str
    label: int
    confidence: float


@app.get("/")
def serve_ui():
    """Serve the simple web UI."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
def health_check():
    """Simple health check endpoint, useful for deployment monitoring."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict sentiment (positive/negative) for a piece of review text.

    Example request body:
        {"text": "This movie was great!"}
    """
    logger.info(f"Received prediction request: {request.text[:80]!r}")

    cleaned = preprocess(request.text)
    features = vectorizer.transform([cleaned])

    pred_label = int(baseline_model.predict(features)[0])
    proba = baseline_model.predict_proba(features)[0]
    confidence = float(max(proba))

    prediction = "positive" if pred_label == 1 else "negative"

    logger.info(f"Prediction: {prediction} (confidence={confidence:.3f})")

    return PredictResponse(
        text=request.text,
        prediction=prediction,
        label=pred_label,
        confidence=round(confidence, 4),
    )


# Serve any other static assets (css/js) if needed later
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
