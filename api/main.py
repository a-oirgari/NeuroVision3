from contextlib import asynccontextmanager
import cv2
import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

MLFLOW_URI = "http://127.0.0.1:5000"
MODEL_URI = "models:/neurovision-yolov8@champion"
ALLOWED_EXT = (".jpeg", ".jpg", ".bmp", ".png")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Exécuté UNE SEULE FOIS au démarrage : on charge le modèle champion
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)
        state["yolo"] = pyfunc_model.unwrap_python_model().model   # l'objet YOLO
    except Exception as e:
        print("Erreur chargement modèle :", e)
        state["yolo"] = None
    yield
    state.clear()


app = FastAPI(title="NeuroVision AI", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    ok = state.get("yolo") is not None
    return JSONResponse({"api": "ok", "model_loaded": ok}, status_code=200 if ok else 503)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Seuil de confiance (0 à 1)"),
):
    model = state.get("yolo")
    if model is None:
        raise HTTPException(503, "Modèle non disponible")

    # 1) Validation du format
    filename = (file.filename or "").lower()
    if not filename.endswith(ALLOWED_EXT):
        raise HTTPException(400, "Format non supporté : jpeg, jpg, bmp ou png uniquement")

    # 2) Validation : image lisible
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Image illisible ou corrompue")

    # 3) Prédiction
    result = model.predict(img, conf=conf, verbose=False)[0]
    detections = []
    for b in result.boxes:
        x1, y1, x2, y2 = [round(v, 1) for v in b.xyxy[0].tolist()]
        detections.append({
            "class_id": int(b.cls),
            "class_name": result.names[int(b.cls)],
            "confidence": round(float(b.conf), 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    message = "Aucune tumeur détectée" if not detections else f"{len(detections)} tumeur(s) détectée(s)"
    return {"message": message, "count": len(detections), "detections": detections,
            "image_size": {"width": img.shape[1], "height": img.shape[0]}}