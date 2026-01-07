import os
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import optuna
import mlflow
import requests
from loguru import logger
from prefect import flow, task
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from tensorflow import keras

from app.ml.model import build_cnn_model, preprocess_image_bytes



# Config

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/data/predictions.db")

MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "latest.keras"))  # aligné API + prefect

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "mnist-hitl")

API_URL = os.getenv("API_URL", "http://api:8000")

# nb d’erreurs par classe à partir duquel on retrain
FAIL_THRESHOLD = int(os.getenv("FAIL_THRESHOLD", "20"))
# minimum de nouveaux labels corrigés avant de retrain
MIN_NEW_LABELS = int(os.getenv("MIN_NEW_LABELS", "50"))


@dataclass
class FeedbackBatch:
    x: np.ndarray        # (n,28,28,1) images corrigées (préparées pour le modèle)
    y: np.ndarray        # (n,) vrais labels
    meta: pd.DataFrame   # infos (id, dates, pred vs true)


# lire les corrections humaines
@task(retries=2, retry_delay_seconds=10)
def load_feedback() -> FeedbackBatch:
    connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    engine = create_engine(DATABASE_URL, connect_args=connect_args)

    with engine.connect() as conn:
        # prend les lignes predictions où corrected=1 et true_label non null
        rows = conn.execute(text("""
            SELECT id, created_at, image_png, predicted_label, true_label
            FROM predictions
            WHERE corrected = 1 AND true_label IS NOT NULL
        """)).fetchall()

    # S’il n’y a rien : renvoie un batch vide.
    if not rows:
        return FeedbackBatch(
            x=np.zeros((0, 28, 28, 1), dtype=np.float32),
            y=np.zeros((0,), dtype=np.int64),
            meta=pd.DataFrame()
        )

    xs: List[np.ndarray] = []
    ys: List[int] = []
    meta_rows = []

    for r in rows:
        try:
            x = preprocess_image_bytes(r.image_png)[0]  # (28,28,1)
            y = int(r.true_label)
            if y < 0 or y > 9:
                continue

            xs.append(x)
            ys.append(y)
            meta_rows.append({
                "id": int(r.id),
                "created_at": str(r.created_at),
                "predicted_label": int(r.predicted_label) if r.predicted_label is not None else None,
                "true_label": y,
            })
        except Exception:
            continue

    meta = pd.DataFrame(meta_rows)

    if len(xs) == 0:
        return FeedbackBatch(
            x=np.zeros((0, 28, 28, 1), dtype=np.float32),
            y=np.zeros((0,), dtype=np.int64),
            meta=meta
        )

    x = np.stack(xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return FeedbackBatch(x=x, y=y, meta=meta)


# compter les erreurs du modèle
@task
def compute_failure_counts(meta: pd.DataFrame) -> Tuple[int, Dict[int, int]]:
    if meta is None or meta.empty:
        return 0, {}

    wrong = meta[meta["predicted_label"] != meta["true_label"]]
    counts = wrong.groupby("true_label").size().to_dict()
    total_wrong = int(wrong.shape[0])

    # cast clés/valeurs en int
    per_class_wrong = {int(k): int(v) for k, v in counts.items()}
    return total_wrong, per_class_wrong


# décision retrain
@task
def should_retrain(total_wrong: int, per_class_wrong: Dict[int, int], n_new: int) -> bool:
    # si pas assez de nouveaux labels (n_new < MIN_NEW_LABELS) → non
    if n_new < MIN_NEW_LABELS:
        logger.info("Not enough new labeled samples ({} < {})", n_new, MIN_NEW_LABELS)
        return False

    # sinon si une classe a trop d’erreurs (cnt >= FAIL_THRESHOLD) → oui
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

    # Ré-entraîner sur données historiques + nouvelles données
    if x_feedback.shape[0] > 0:
        x_all = np.concatenate([x_train, x_feedback], axis=0)
        y_all = np.concatenate([y_train, y_feedback], axis=0)
    else:
        x_all, y_all = x_train, y_train

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_all, y_all, test_size=0.1, random_state=42, stratify=y_all
    )
    return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)


# objectif Optuna, Optuna teste des hyperparamètres
def _optuna_objective(trial, x_tr, y_tr, x_val, y_val):
    lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    model = build_cnn_model(lr=lr, dropout=dropout, sparse_labels=True)
    model.fit(x_tr, y_tr, epochs=1, batch_size=batch_size, verbose=0)
    _, val_acc = model.evaluate(x_val, y_val, verbose=0)
    return float(val_acc)


"""
Retrain optima :
- prépare données train/val/test
- configure MLflow (URI + experiment)
- démarre un run MLflow retrain_<timestamp>
- lance Optuna sur 20 trials
- log meilleurs params
"""
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
        model = build_cnn_model(lr=float(best["lr"]), dropout=float(best["dropout"]), sparse_labels=True)
        history = model.fit(
            x_tr, y_tr,
            epochs=3,
            batch_size=int(best["batch_size"]),
            validation_data=(x_val, y_val),
            verbose=2
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_loss", float(test_loss))

        # log métriques finales (scalaires)
        if "val_accuracy" in history.history:
            mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][-1]))
        if "val_loss" in history.history:
            mlflow.log_metric("val_loss", float(history.history["val_loss"][-1]))
        if "accuracy" in history.history:
            mlflow.log_metric("train_accuracy", float(history.history["accuracy"][-1]))
        if "loss" in history.history:
            mlflow.log_metric("train_loss", float(history.history["loss"][-1]))

        # Save model version
        ts = int(time.time())
        model_version_path = os.path.join(MODEL_DIR, f"model_{ts}.keras")
        model.save(model_version_path)
        mlflow.log_artifact(model_version_path, artifact_path="model")

        # Update latest (API charge MODEL_PATH)
        model.save(MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")

        # also log training curves
        for k, vals in history.history.items():
            for i, v in enumerate(vals):
                mlflow.log_metric(f"{k}_epoch", float(v), step=i)

        return model_version_path


# informer le service API
# reload : l’API recharge latest.keras
# retrain_mark?ts_unix=... : marque le timestamp (pour monitoring)
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
    do = should_retrain(total_wrong, per_class_wrong, int(fb.x.shape[0]))

    # Log monitoring to MLflow even if no retrain
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"monitor_{int(time.time())}"):
        mlflow.log_metric("labeled_samples_total", float(fb.x.shape[0]))
        mlflow.log_metric("wrong_total", float(total_wrong))
        for cls, cnt in per_class_wrong.items():
            mlflow.log_metric(f"wrong_class_{cls}", float(cnt))

    if do:
        model_path = retrain_with_optuna(fb.x, fb.y)
        logger.info("Retrain done. New model saved at {}", model_path)
        notify_api_model_reload()
    else:
        logger.info("No retrain triggered.")


if __name__ == "__main__":
    interval_seconds = int(os.getenv("SCHEDULE_INTERVAL_SECONDS", "3600"))

    hourly_monitor_and_retrain.serve(
        name="mnist-hitl-hourly-deployment",
        interval=interval_seconds
    )
