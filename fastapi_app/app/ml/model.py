import os
import shutil
import numpy as np
from loguru import logger

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import mlflow


# -----------------------------
# Config
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/mnist_cnn_model.h5")

EPOCHS = int(os.getenv("TRAIN_EPOCHS", "10"))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "128"))
VAL_SPLIT = float(os.getenv("TRAIN_VAL_SPLIT", "0.1"))

BOOTSTRAP_FROM_MLFLOW = os.getenv("BOOTSTRAP_FROM_MLFLOW", "true").lower() == "true"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "mnist-hitl")
MLFLOW_METRIC = os.getenv("MLFLOW_METRIC", "val_accuracy")
MLFLOW_MODEL_ARTIFACT_PATH = os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", "model/mnist_cnn_model.h5")

_MODEL: tf.keras.Model | None = None


# -----------------------------
# CNN definition (notebook)
# -----------------------------
def create_cnn_model(input_shape=(28, 28, 1), num_classes=10) -> tf.keras.Model:
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# -----------------------------
# MLflow helpers
# -----------------------------
def _mlflow_setup() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def _try_bootstrap_from_mlflow(to_local_path: str = MODEL_PATH) -> bool:
    """
    Télécharge le meilleur modèle (artefact .h5) depuis MLflow et le copie vers MODEL_PATH.
    Retourne True si succès.
    """
    if not BOOTSTRAP_FROM_MLFLOW:
        return False

    try:
        _mlflow_setup()
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if exp is None:
            logger.warning("MLflow experiment not found: {}", MLFLOW_EXPERIMENT)
            return False

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{MLFLOW_METRIC} DESC"],
            max_results=1,
        )
        if not runs:
            logger.warning("No MLflow runs found in experiment: {}", MLFLOW_EXPERIMENT)
            return False

        best = runs[0]
        run_id = best.info.run_id
        best_val = best.data.metrics.get(MLFLOW_METRIC)
        logger.info("Best MLflow run: run_id={} {}={}", run_id, MLFLOW_METRIC, best_val)

        downloaded = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=MLFLOW_MODEL_ARTIFACT_PATH,
        )

        # downloaded peut être fichier ou dossier
        if os.path.isdir(downloaded):
            candidates = []
            for root, _, files in os.walk(downloaded):
                for f in files:
                    if f.endswith(".h5"):
                        candidates.append(os.path.join(root, f))
            if not candidates:
                logger.warning("Downloaded artifact dir contains no .h5: {}", downloaded)
                return False
            src = candidates[0]
        else:
            src = downloaded

        os.makedirs(os.path.dirname(to_local_path), exist_ok=True)
        shutil.copyfile(src, to_local_path)
        logger.warning("Bootstrapped model from MLflow -> {}", os.path.abspath(to_local_path))
        return True

    except Exception as e:
        logger.warning("MLflow bootstrap failed: {}", e)
        return False


# -----------------------------
# Training + MLflow logging
# -----------------------------
def train_and_save_and_log(model_path: str = MODEL_PATH) -> None:
    """
    Entraîne (logique notebook) + sauvegarde local .h5 + log MLflow:
    - params
    - metrics (val + test)
    - artefact (model/mnist_cnn_model.h5)
    """
    logger.warning("Training bootstrap CNN (no local model available).")
    logger.info("Params: epochs={} batch_size={} val_split={}", EPOCHS, BATCH_SIZE, VAL_SPLIT)

    _mlflow_setup()

    # Data MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = (X_train.astype("float32") / 255.0)[..., None]
    X_test = (X_test.astype("float32") / 255.0)[..., None]

    y_train_cat = to_categorical(y_train, num_classes=10)
    y_test_cat = to_categorical(y_test, num_classes=10)

    model = create_cnn_model()

    with mlflow.start_run(run_name="bootstrap-train"):
        # Params
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("val_split", VAL_SPLIT)
        mlflow.log_param("arch", "cnn_32_64_dense128_dropout05")

        # Train
        history = model.fit(
            X_train,
            y_train_cat,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VAL_SPLIT,
            verbose=1,
        )

        # Metrics val (si dispo)
        if "val_accuracy" in history.history:
            mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][-1]))
        if "val_loss" in history.history:
            mlflow.log_metric("val_loss", float(history.history["val_loss"][-1]))

        # Metrics test
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_loss", float(test_loss))

        logger.info("Bootstrap test_accuracy={}", float(test_acc))

        # Save local
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        logger.warning("Saved local model: {}", os.path.abspath(model_path))

        # Log artifact in MLflow
        mlflow.log_artifact(model_path, artifact_path="model")
        logger.info("Logged model artifact to MLflow at artifact_path=model")


# -----------------------------
# Public API
# -----------------------------
def ensure_model_loaded(force_reload: bool = False) -> None:
    """
    - Si modèle en mémoire et pas force_reload -> return
    - Si .h5 absent:
        * bootstrap MLflow (best run) si possible
        * sinon train + log MLflow
    - Puis load en mémoire
    """
    global _MODEL

    if _MODEL is not None and not force_reload:
        return

    if not os.path.exists(MODEL_PATH):
        ok = _try_bootstrap_from_mlflow(MODEL_PATH)
        if not ok and not os.path.exists(MODEL_PATH):
            train_and_save_and_log(MODEL_PATH)

    _MODEL = load_model(MODEL_PATH)
    logger.info("Model loaded in memory from {}", MODEL_PATH)


def get_model() -> tf.keras.Model:
    ensure_model_loaded()
    return _MODEL


def preprocess_image_uint8(arr28: np.ndarray) -> np.ndarray:
    """
    Input: (28,28) uint8 0..255
    Output: (1,28,28,1) float32 0..1
    """
    x = arr28.astype("float32")
    if x.max() > 1.0:
        x /= 255.0
    return x[None, ..., None]
