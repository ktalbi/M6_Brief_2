import io
import os
from typing import Generator

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from loguru import logger
from PIL import Image
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.orm import Session

from app.db.database import SessionLocal, engine
from app.db import models
from app.ml.model import ensure_model_loaded, get_model, preprocess_image_uint8
from app.schemas.prediction import PredictResponse, CorrectionRequest, CorrectionResponse

# Logging
logger.add("logs/app.log", rotation="5 MB")

# DB init
models.Base.metadata.create_all(bind=engine)

# App
app = FastAPI(title="MNIST Human-in-the-Loop API", version="1.0.0")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Metrics
PREDICTIONS_TOTAL = Counter("mnist_predictions_total", "Total number of predictions served")
CORRECTIONS_TOTAL = Counter("mnist_corrections_total", "Total number of user corrections submitted")
CORRECTION_RATE = Gauge("mnist_correction_rate", "Corrections / predictions (best-effort)")
LAST_RETRAIN_TS = Gauge(
    "mnist_last_retrain_timestamp",
    "Unix timestamp of last retrain (set by retraining service)"
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def startup():
    os.makedirs("logs", exist_ok=True)
    os.makedirs(os.getenv("MODEL_DIR", "/app/models"), exist_ok=True)
    os.makedirs(os.getenv("DATA_DIR", "/app/data"), exist_ok=True)

    # IMPORTANT : si le .h5 n'existe pas, ensure_model_loaded() déclenche l'entraînement
    ensure_model_loaded()
    logger.info("API started, model ready.")


def _update_correction_rate(db: Session) -> None:
    try:
        preds = db.query(models.Prediction).count()
        corrs = db.query(models.Prediction).filter(models.Prediction.corrected == True).count()
        if preds > 0:
            CORRECTION_RATE.set(corrs / preds)
    except Exception as e:
        logger.warning("Failed updating correction rate gauge: {}", e)


def _read_as_mnist_28x28(file_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.uint8)

    # Heuristic: invert if background is white (typical canvas)
    if arr.mean() > 127:
        arr = 255 - arr

    return arr


@app.get("/")
def root():
    return {"message": "MNIST HITL API operational"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Sécurité: si l'API démarre sans modèle chargé (cas rare), on bootstrape
    ensure_model_loaded()
    model = get_model()

    file_bytes = await file.read()
    arr28 = _read_as_mnist_28x28(file_bytes)
    x = preprocess_image_uint8(arr28)

    probs = model.predict(x, verbose=0)[0].astype(float)
    pred = int(np.argmax(probs))

    rec = models.Prediction(
        image_png=file_bytes,
        predicted_label=pred,
        true_label=None,
        corrected=False
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    PREDICTIONS_TOTAL.inc()
    _update_correction_rate(db)

    return PredictResponse(
        prediction_id=rec.id,
        predicted_label=pred,
        probabilities=probs.tolist()
    )


@app.post("/correct", response_model=CorrectionResponse)
def correct(payload: CorrectionRequest, db: Session = Depends(get_db)):
    rec = db.get(models.Prediction, payload.prediction_id)
    if not rec:
        raise HTTPException(status_code=404, detail="prediction_id not found")

    rec.true_label = int(payload.true_label)
    rec.corrected = True
    db.add(rec)
    db.commit()

    CORRECTIONS_TOTAL.inc()
    _update_correction_rate(db)

    return CorrectionResponse(status="ok")


@app.post("/reload")
def reload_model():
    """
    Recharge le modèle.
    (Si le .h5 manque, ensure_model_loaded() relance un entraînement.)
    """
    # Si tu veux "forcer" un retrain ici, fais-le côté ml/model.py (ex: force=True)
    ensure_model_loaded()
    return {"status": "reloaded"}


@app.post("/retrain_mark")
def retrain_mark(ts_unix: float):
    # appelé par le service de retraining après déploiement réussi
    LAST_RETRAIN_TS.set(ts_unix)
    return {"status": "ok"}


@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    total = db.query(models.Prediction).count()
    corrected = db.query(models.Prediction).filter(models.Prediction.corrected == True).count()

    if corrected:
        wrong = db.query(models.Prediction).filter(
            models.Prediction.corrected == True,
            models.Prediction.true_label != models.Prediction.predicted_label
        ).count()
        acc = 1 - (wrong / corrected)
    else:
        acc = None

    return {
        "total_predictions": total,
        "corrected": corrected,
        "corrected_accuracy": acc
    }
