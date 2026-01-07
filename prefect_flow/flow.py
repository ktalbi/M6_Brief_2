import os
import io
import time
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import optuna
import mlflow
import requests
from loguru import logger
from PIL import Image
from prefect import flow, task
from sqlalchemy import create_engine, text

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

DB_PATH = os.getenv("FEEDBACK_DB_PATH", "/app/data/feedback.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:////{DB_PATH.lstrip('/')}")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
LATEST_MODEL_PATH = os.getenv("LATEST_MODEL_PATH", os.path.join(MODEL_DIR, "latest.keras"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "mnist-hitl")

API_URL = os.getenv("API_URL", "http://api:8000")

FAIL_THRESHOLD = int(os.getenv("FAIL_THRESHOLD", "20"))
MIN_NEW_LABELS = int(os.getenv("MIN_NEW_LABELS", "50"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))  # Wasserstein on mean intensity (heuristic)

@dataclass
class FeedbackBatch:
    x: np.ndarray  # (n,28,28,1)
    y: np.ndarray  # (n,)
    meta: pd.DataFrame

def build_cnn(lr: float = 1e-3, dropout: float = 0.25) -> keras.Model:
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def decode_png_to_28x28(png_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.uint8)
    # keep MNIST convention: white digit on black
    if arr.mean() > 127:
        arr = 255 - arr
    x = (arr.astype("float32") / 255.0)[..., None]
    return x

@task(retries=2, retry_delay_seconds=10)
def load_feedback() -> FeedbackBatch:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
    with engine.connect() as conn:
        rows = conn.execute(text(
            """
            SELECT id, created_at, image_png, predicted_label, true_label
            FROM predictions
            WHERE corrected = 1 AND true_label IS NOT NULL
            """
        )).fetchall()

    if not rows:
        return FeedbackBatch(x=np.zeros((0,28,28,1), dtype=np.float32), y=np.zeros((0,), dtype=np.int64), meta=pd.DataFrame())

    xs: List[np.ndarray] = []
    ys: List[int] = []
    meta_rows = []
    for r in rows:
        try:
            x = decode_png_to_28x28(r.image_png)
            y = int(r.true_label)
            if y < 0 or y > 9:
                continue
            xs.append(x)
            ys.append(y)
            meta_rows.append({"id": r.id, "created_at": str(r.created_at), "predicted_label": int(r.predicted_label), "true_label": y})
        except Exception:
            continue

    meta = pd.DataFrame(meta_rows)
    if len(xs) == 0:
        return FeedbackBatch(x=np.zeros((0,28,28,1), dtype=np.float32), y=np.zeros((0,), dtype=np.int64), meta=meta)

    x = np.stack(xs, axis=0)
    y = np.array(ys, dtype=np.int64)
    return FeedbackBatch(x=x, y=y, meta=meta)

@task
def compute_failure_counts(meta: pd.DataFrame) -> Tuple[int, dict]:
    if meta is None or meta.empty:
        return 0, {}
    wrong = meta[meta["predicted_label"] != meta["true_label"]]
    counts = wrong.groupby("true_label").size().to_dict()
    total_wrong = int(wrong.shape[0])
    return total_wrong, {int(k): int(v) for k, v in counts.items()}

@task
def simple_drift_score(x_new: np.ndarray) -> float:
    """Heuristic drift: compare mean intensity distribution new vs MNIST train."""
    if x_new.shape[0] < 10:
        return 0.0
    (x_train, _), _ = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
    base_means = x_train.mean(axis=(1,2,3))
    new_means = x_new.mean(axis=(1,2,3))
    # Wasserstein distance (1D)
    base_means = np.sort(base_means)
    new_means = np.sort(new_means)
    n = min(len(base_means), len(new_means))
    base_means = base_means[:n]
    new_means = new_means[:n]
    return float(np.mean(np.abs(base_means - new_means)))

@task
def should_retrain(total_wrong: int, per_class_wrong: dict, n_new: int, drift_score: float) -> bool:
    if n_new < MIN_NEW_LABELS:
        logger.info("Not enough new labeled samples ({} < {})", n_new, MIN_NEW_LABELS)
        return False
    if drift_score >= DRIFT_THRESHOLD:
        logger.warning("Drift detected (score {:.4f} >= {:.4f})", drift_score, DRIFT_THRESHOLD)
        return True
    for cls, cnt in per_class_wrong.items():
        if cnt >= FAIL_THRESHOLD:
            logger.warning("Threshold reached: class {} wrong count {}", cls, cnt)
            return True
    logger.info("No retrain condition met (total_wrong={}, per_class_wrong={})", total_wrong, per_class_wrong)
    return False

def _prepare_training_data(x_feedback: np.ndarray, y_feedback: np.ndarray):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    if x_feedback.shape[0] > 0:
        x_all = np.concatenate([x_train, x_feedback], axis=0)
        y_all = np.concatenate([y_train, y_feedback], axis=0)
    else:
        x_all, y_all = x_train, y_train

    x_tr, x_val, y_tr, y_val = train_test_split(x_all, y_all, test_size=0.1, random_state=42, stratify=y_all)
    return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)

def _optuna_objective(trial, x_tr, y_tr, x_val, y_val):
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    model = build_cnn(lr=lr, dropout=dropout)
    model.fit(x_tr, y_tr, epochs=1, batch_size=batch_size, verbose=0)
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    return float(val_acc)

@task
def retrain_with_optuna(x_feedback: np.ndarray, y_feedback: np.ndarray) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)

    (x_tr, y_tr), (x_val, y_val), (x_test, y_test) = _prepare_training_data(x_feedback, y_feedback)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"retrain_{int(time.time())}") as run:
        # Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: _optuna_objective(t, x_tr, y_tr, x_val, y_val), n_trials=20)

        best = study.best_params
        mlflow.log_params(best)
        mlflow.log_metric("optuna_best_value", float(study.best_value))

        # Train best model longer
        model = build_cnn(lr=float(best["lr"]), dropout=float(best["dropout"]))
        history = model.fit(x_tr, y_tr, epochs=3, batch_size=int(best["batch_size"]), validation_data=(x_val, y_val), verbose=2)

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_loss", float(test_loss))

        # Save model version
        ts = int(time.time())
        model_path = os.path.join(MODEL_DIR, f"model_{ts}.keras")
        model.save(model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # Update latest
        try:
            if os.path.exists(LATEST_MODEL_PATH):
                os.remove(LATEST_MODEL_PATH)
        except Exception:
            pass
        # copy to keep simple across filesystems
        import shutil
        shutil.copyfile(model_path, LATEST_MODEL_PATH)

        # also log training curves
        for k, vals in history.history.items():
            for i, v in enumerate(vals):
                mlflow.log_metric(k, float(v), step=i)

        return model_path

@task
def notify_api_model_reload():
    try:
        requests.post(f"{API_URL}/reload", timeout=10)
        requests.post(f"{API_URL}/retrain_mark", params={"ts_unix": time.time()}, timeout=10)
    except Exception as e:
        logger.warning("Failed to notify API: {}", e)

@flow(name="mnist-hitl-hourly")
def hourly_monitor_and_retrain():
    fb = load_feedback()
    total_wrong, per_class_wrong = compute_failure_counts(fb.meta)
    drift = simple_drift_score(fb.x)
    do = should_retrain(total_wrong, per_class_wrong, int(fb.x.shape[0]), drift)

    # Log monitoring to MLflow even if no retrain
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"monitor_{int(time.time())}"):
        mlflow.log_metric("labeled_samples_total", int(fb.x.shape[0]))
        mlflow.log_metric("wrong_total", int(total_wrong))
        mlflow.log_metric("drift_score", float(drift))
        for cls, cnt in per_class_wrong.items():
            mlflow.log_metric(f"wrong_class_{cls}", int(cnt))

    if do:
        model_path = retrain_with_optuna(fb.x, fb.y)
        logger.info("Retrain done. New model saved at {}", model_path)
        notify_api_model_reload()

if __name__ == "__main__":

    interval_seconds = int(os.getenv("SCHEDULE_INTERVAL_SECONDS", "3600"))

    hourly_monitor_and_retrain.serve(
        name="mnist-hitl-hourly-deployment",
        interval=interval_seconds
    )

