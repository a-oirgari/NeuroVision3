import cv2
import numpy as np
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.title("NeuroVision AI - Détection de tumeurs cérébrales")
conf = st.slider("Seuil de confiance", 0.05, 0.95, 0.25, 0.05)
uploaded = st.file_uploader("Charger une IRM", type=None)   # type=None : on valide nous-mêmes

if st.button("Analyser"):
    if uploaded is None:
        st.error("Veuillez d'abord charger une image.")
        st.stop()

    try:
        resp = requests.post(
            f"{API_URL}/predict", params={"conf": conf},
            files={"file": (uploaded.name, uploaded.getvalue())}, timeout=60,
        )
    except requests.exceptions.ConnectionError:
        st.error("API indisponible. Vérifiez que FastAPI est lancé.")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("L'API met trop de temps à répondre.")
        st.stop()

    if resp.status_code == 400:
        st.error(resp.json().get("detail", "Format non supporté ou image illisible."))
        st.stop()
    if resp.status_code != 200:
        st.error(f"Erreur API ({resp.status_code}) : {resp.text}")
        st.stop()

    result = resp.json()
    img = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if result["count"] == 0:
        st.info("Aucune tumeur détectée")
    else:
        for d in result["detections"]:
            b = d["bbox"]
            p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
            cv2.rectangle(img, p1, p2, (255, 0, 0), 3)
            cv2.putText(img, f'{d["class_name"]} {d["confidence"]:.2f}', (p1[0], max(p1[1] - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        st.success(result["message"])

    st.image(img, caption="Résultat", use_container_width=True)
    st.json(result["detections"])