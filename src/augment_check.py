from ultralytics import YOLO, settings
from config import DATA_YAML, RUNS_DIR

settings.update({"mlflow": False})
YOLO("yolov8n.pt").train(
    data=str(DATA_YAML), epochs=1, imgsz=640, batch=16, project=str(RUNS_DIR), name="augment_check",
    exist_ok=True, degrees=10, translate=0.1, scale=0.3, fliplr=0.0, hsv_h=0.0, hsv_s=0.0,
    hsv_v=0.3, mosaic=0.5, plots=True,
)
print("Regarde runs/detect/augment_check/train_batch0.jpg")
