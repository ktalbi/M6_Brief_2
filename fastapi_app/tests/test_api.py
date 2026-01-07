import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_and_correct():
    # create simple fake digit-like blob
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[10:18, 12:16] = 255
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("x.png", buf.getvalue(), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    payload = r.json()
    assert "prediction_id" in payload
    assert 0 <= payload["predicted_label"] <= 9

    r2 = client.post("/correct", json={"prediction_id": payload["prediction_id"], "true_label": 0})
    assert r2.status_code == 200
