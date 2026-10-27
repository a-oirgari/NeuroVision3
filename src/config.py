from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DATASET = ROOT / "data" / "dataset"
DATA_YAML = DATASET / "data.yaml"
RUNS_DIR = ROOT / "runs" / "detect"
EXPORT = ROOT / "export"

NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
NC = len(NAMES)

IMG_EXT = {".jpeg", ".jpg", ".bmp", ".png"}

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT = "neurovision-yolov8"
REGISTERED_MODEL = "neurovision-yolov8"
LAST_RUN_FILE = ROOT / "last_run_id.txt"