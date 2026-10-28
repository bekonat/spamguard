import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from app.text import preprocess_str

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent.parent / "artifacts" / "model.joblib"))
THRESHOLD = float(os.getenv("SPAM_THRESHOLD", "0.5"))

PREDICTIONS = Counter("model_predictions_total", "Predicted messages by label", ["label"])
LATENCY = Histogram("model_inference_seconds", "Inference time per batch",
                    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
BATCH = Histogram("model_batch_size", "Messages per request", buckets=(1, 2, 4, 8, 16, 32, 64, 128))


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class Prediction(BaseModel):
    label: str
    spam_probability: float


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


model = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    art = joblib.load(MODEL_PATH)
    model["pipeline"], model["version"] = art["pipeline"], art["version"]
    yield
    model.clear()


app = FastAPI(title="spamguard model", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": model.get("version")}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    proba = model["pipeline"].predict_proba([preprocess_str(t) for t in req.texts])[:, 1]
    LATENCY.observe(time.perf_counter() - start)
    BATCH.observe(len(req.texts))

    out = []
    for p in proba:
        label = "spam" if p >= THRESHOLD else "ham"
        PREDICTIONS.labels(label).inc()
        out.append(Prediction(label=label, spam_probability=round(float(p), 4)))
    return PredictResponse(model_version=model["version"], predictions=out)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
