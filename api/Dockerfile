# --- Python base image ---
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (helpful for Cloud Run logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --- Requirements installation ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the NLTK data needed by the preprocessing function saved in model.pkl
RUN python -m nltk.downloader stopwords wordnet omw-1.4 punkt punkt_tab

# --- Copy application code and model file into the image ---
COPY main.py .
COPY static/ ./static/
COPY models/ ./models/

# --- Exposed port ---
# Cloud Run expects the container to listen on the port given by $PORT (defaults to 8080).
ENV PORT=8080
EXPOSE 8080

# --- API startup command ---
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
