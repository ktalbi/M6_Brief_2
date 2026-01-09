import io
import os
import shutil
from typing import Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import mlflow


# Config

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/latest.keras")

EPOCHS = int(os.getenv("TRAIN_EPOCHS", "10"))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "128"))
VAL_SPLIT = float(os.getenv("TRAIN_VAL_SPLIT", "0.1"))

BOOTSTRAP_FROM_MLFLOW = os.getenv("BOOTSTRAP_FROM_MLFLOW", "true").lower() == "true"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "mnist-hitl")
MLFLOW_METRIC = os.getenv("MLFLOW_METRIC", "val_accuracy")

# Chemin de l'artefact modèle dans MLflow (le fichier loggé sous artifact_path="model")
MLFLOW_MODEL_ARTIFACT_PATH = os.getenv("MLFLOW_MODEL_ARTIFACT_PATH", "model/latest.keras")

_MODEL: Optional[tf.keras.Model] = None


# Model definition 

def build_cnn_model(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 10,
    lr: float = 1e-3,
    dropout: float = 0.5,
    sparse_labels: bool = True,
) -> tf.keras.Model:
    """
    CNN MNIST.
    sparse_labels=True  => y = int (0..9), loss=sparse_categorical_crossentropy
    sparse_labels=False => y = one-hot,     loss=categorical_crossentropy
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(dropout),
        Dense(num_classes, activation="softmax"),
    ])

    loss = "sparse_categorical_crossentropy" if sparse_labels else "categorical_crossentropy"
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss, metrics=["accuracy"])
    return model

# Preprocessing 

def read_as_mnist_28x28_uint8(file_bytes: bytes) -> np.ndarray:
    """
    Convertit des bytes d'image en tableau uint8 (28x28) au format "MNIST-like" :
    - Conversion en niveaux de gris
    - Inversion si le fond est clair (chiffre clair sur fond sombre)
    - Découpage serré autour du chiffre
    - Mise à l'échelle dans une boîte 20x20 (comme MNIST)
    - Padding vers 28x28
    - Recentrage via le centre de masse
    """
    # --- Décodage de l'image ---
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("L")
    except Exception:
        raise ValueError("Impossible de décoder l'image")

    arr = np.array(img, dtype=np.uint8)

    # --- Inversion si nécessaire (convention MNIST : chiffre clair sur fond sombre) ---
    if arr.mean() > 127:
        arr = 255 - arr

    # --- Détection des pixels appartenant au chiffre ---
    # Seuil faible pour ignorer le bruit très léger
    seuil = 30
    masque = arr > seuil

    if not np.any(masque):
        # Aucun trait détecté → image vide
        return np.zeros((28, 28), dtype=np.uint8)

    # --- Bounding box du chiffre ---
    ys, xs = np.where(masque)
    y_min, y_max = ys.min(), ys.max() + 1
    x_min, x_max = xs.min(), xs.max() + 1

    # --- Découpage serré ---
    decoupe = arr[y_min:y_max, x_min:x_max]

    # --- Mise au carré par padding (pour éviter la distorsion à la mise à l'échelle) ---
    h, w = decoupe.shape
    taille = max(h, w)

    carre = np.zeros((taille, taille), dtype=np.uint8)
    y_offset = (taille - h) // 2
    x_offset = (taille - w) // 2
    carre[y_offset:y_offset + h, x_offset:x_offset + w] = decoupe

    # --- Redimensionnement vers 20x20 (taille typique MNIST) ---
    img_carre = Image.fromarray(carre, mode="L")
    if taille != 20:
        img_carre = img_carre.resize((20, 20), resample=Image.BILINEAR)

    chiffre_20 = np.array(img_carre, dtype=np.uint8)

    # --- Padding final vers 28x28 ---
    out28 = np.zeros((28, 28), dtype=np.uint8)
    out28[4:24, 4:24] = chiffre_20

    # --- Recentrage par centre de masse (pondéré par l'intensité) ---
    m = out28.astype(np.float32)
    somme = float(m.sum())

    if somme > 0:
        ys, xs = np.indices((28, 28))
        centre_y = float((ys * m).sum() / somme)
        centre_x = float((xs * m).sum() / somme)

        # Centre cible ≈ centre MNIST
        cible = 13.5
        dy = int(round(cible - centre_y))
        dx = int(round(cible - centre_x))

        if dx != 0 or dy != 0:
            recentre = np.zeros_like(out28)

            # Calcul des plages source/destination en évitant les débordements
            y_src0 = max(0, -dy)
            y_src1 = min(28, 28 - dy)
            x_src0 = max(0, -dx)
            x_src1 = min(28, 28 - dx)

            y_dst0 = max(0, dy)
            y_dst1 = min(28, 28 + dy)
            x_dst0 = max(0, dx)
            x_dst1 = min(28, 28 + dx)

            recentre[y_dst0:y_dst1, x_dst0:x_dst1] = out28[y_src0:y_src1, x_src0:x_src1]
            out28 = recentre

    return out28



def preprocess_image_uint8(arr28: np.ndarray) -> np.ndarray:
    """
    Input: (28,28) uint8 0..255
    Output: (1,28,28,1) float32 0..1
    """
    x = arr28.astype("float32")
    if x.max() > 1.0:
        x /= 255.0
    return x[None, ..., None]


def preprocess_image_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Input: image bytes
    Output: (1,28,28,1) float32 0..1 compatible MNIST
    """
    arr28 = read_as_mnist_28x28_uint8(file_bytes)
    return preprocess_image_uint8(arr28)

# MLflow helpers

def _mlflow_setup() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def _try_bootstrap_from_mlflow(to_local_path: str = MODEL_PATH) -> bool:
    """
    Télécharge le meilleur modèle depuis MLflow et le copie vers MODEL_PATH.
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
        src = downloaded
        if os.path.isdir(downloaded):
            candidates = []
            for root, _, files in os.walk(downloaded):
                for f in files:
                    if f.endswith((".keras", ".h5")):
                        candidates.append(os.path.join(root, f))
            if not candidates:
                logger.warning("Downloaded artifact dir contains no model file: {}", downloaded)
                return False
            src = candidates[0]

        os.makedirs(os.path.dirname(to_local_path), exist_ok=True)
        shutil.copyfile(src, to_local_path)
        logger.warning("Bootstrapped model from MLflow -> {}", os.path.abspath(to_local_path))
        return True

    except Exception as e:
        logger.warning("MLflow bootstrap failed: {}", e)
        return False


# Training + MLflow logging

def train_and_save_and_log(model_path: str = MODEL_PATH) -> None:
    """
    Entraîne un modèle MNIST de base, sauvegarde localement, et log MLflow.
    """
    logger.warning("Training bootstrap CNN (no local model available).")
    logger.info("Params: epochs={} batch_size={} val_split={}", EPOCHS, BATCH_SIZE, VAL_SPLIT)

    _mlflow_setup()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    # On utilise sparse labels pour rester compatible avec le retrain (flow)
    model = build_cnn_model(lr=1e-3, dropout=0.5, sparse_labels=True)

    with mlflow.start_run(run_name="bootstrap-train"):
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("val_split", VAL_SPLIT)
        mlflow.log_param("arch", "cnn_32_64_dense128_dropout05")
        mlflow.log_param("label_mode", "sparse")

        history = model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VAL_SPLIT,
            verbose=1,
        )

        if "val_accuracy" in history.history:
            mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][-1]))
        if "val_loss" in history.history:
            mlflow.log_metric("val_loss", float(history.history["val_loss"][-1]))
        if "accuracy" in history.history:
            mlflow.log_metric("train_accuracy", float(history.history["accuracy"][-1]))
        if "loss" in history.history:
            mlflow.log_metric("train_loss", float(history.history["loss"][-1]))

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_loss", float(test_loss))
        logger.info("Bootstrap test_accuracy={}", float(test_acc))

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        logger.warning("Saved local model: {}", os.path.abspath(model_path))

        mlflow.log_artifact(model_path, artifact_path="model")
        logger.info("Logged model artifact to MLflow at artifact_path=model")


# Public API

def ensure_model_loaded(force_reload: bool = False) -> None:
    """
    - Si modèle en mémoire et pas force_reload -> return
    - Si absent sur disque:
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
    assert _MODEL is not None
    return _MODEL
