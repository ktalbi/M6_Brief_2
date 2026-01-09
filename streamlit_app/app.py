import io
import os
import numpy as np
import requests
import streamlit as st
from loguru import logger
from PIL import Image
from streamlit_drawable_canvas import st_canvas

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="MNIST - Human Feedback Loop", layout="centered")
st.title("MNIST en production (boucle de feedback humain)")

st.markdown(
    """Dessine un chiffre (0–9), puis lance une prédiction.  
Si la prédiction est incorrecte, corrige-la : l'image + le label sont stockés pour ré-entraînement."""
)

# Canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.0)",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def canvas_to_png_bytes() -> bytes:
    if canvas_result.image_data is None:
        raise ValueError("No image")
    img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
    img = img.convert("L")
    arr = np.array(img, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

col1, col2 = st.columns(2)

with col1:
    if st.button("Prédire", use_container_width=True):
        try:
            png_bytes = canvas_to_png_bytes()
            files = {"file": ("canvas.png", png_bytes, "image/png")}
            resp = requests.post(f"{API_URL}/predict", files=files, timeout=10)
            if resp.status_code != 200:
                st.error(f"Erreur API: {resp.status_code} - {resp.text}")
            else:
                payload = resp.json()
                st.session_state["last_prediction"] = payload
                st.success(f"Prédiction: **{payload['predicted_label']}** (id={payload['prediction_id']})")
        except Exception as e:
            logger.exception(e)
            st.error("Impossible de prédire. Vérifie que tu as dessiné un chiffre et que l'API est up.")

with col2:
    if st.button("Effacer", use_container_width=True):
        st.session_state.pop("last_prediction", None)
        st.experimental_rerun()

pred = st.session_state.get("last_prediction")
st.divider()

if pred:
    st.subheader("Correction")
    st.write(f"ID prédiction: `{pred['prediction_id']}` • Prédit: **{pred['predicted_label']}**")
    true_label = st.selectbox("Label correct (0–9)", list(range(10)), index=int(pred["predicted_label"]))
    if st.button("Corriger & enregistrer", use_container_width=True):
        r = requests.post(
            f"{API_URL}/correct",
            json={"prediction_id": int(pred["prediction_id"]), "true_label": int(true_label)},
            timeout=10,
        )
        if r.status_code == 200:
            st.success("Correction enregistrée")
        else:
            st.error(f"Erreur correction: {r.status_code} - {r.text}")

    with st.expander("Probabilités"):
        probs = pred.get("probabilities", [])
        for i, p in enumerate(probs):
            st.write(f"{i}: {p:.4f}")

st.caption("Stack: Streamlit (UI) → FastAPI (prédiction + stockage) → Prefect (analyse + retrain) → MLflow/Prometheus/Grafana (monitoring)")
