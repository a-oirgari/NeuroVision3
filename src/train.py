import argparse
import time
from pathlib import Path
import mlflow
from ultralytics import YOLO, settings
from config import DATA_YAML, RUNS_DIR, MLFLOW_URI, EXPERIMENT, LAST_RUN_FILE

settings.update({"mlflow": False})

p = argparse.ArgumentParser()
p.add_argument("--model", default="yolov8n.pt")
p.add_argument("--name", default="yolov8n_aug_medium")
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--mosaic", type=float, default=0.5)
p.add_argument("--degrees", type=float, default=10)
p.add_argument("--resume", action="store_true")
args = p.parse_args()

run_dir = RUNS_DIR / args.name


if args.resume:
    YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True)
    raise SystemExit

train_args = dict(
    data=str(DATA_YAML), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
    patience=args.patience, seed=42, device=0,
    project=str(RUNS_DIR), name=args.name, exist_ok=True,
    degrees=args.degrees, translate=0.1, scale=0.3,
    fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
    mosaic=args.mosaic, close_mosaic=10,
)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name=args.name) as run:
    mlflow.log_params({"model": args.model, **{k: v for k, v in train_args.items() if k not in ("data", "project", "name", "exist_ok", "device")}})

    model = YOLO(args.model)                 # poids COCO pré-entraînés
    t0 = time.time()
    model.train(**train_args)
    duration = time.time() - t0

    best = run_dir / "weights" / "best.pt"

    val = YOLO(str(best)).val(data=str(DATA_YAML), split="val", plots=False, verbose=False,
                              project=str(RUNS_DIR), name=f"{args.name}_val", exist_ok=True)
    mlflow.log_metrics({
        "train_duration_s": duration,
        "val_precision": float(val.box.mp),
        "val_recall": float(val.box.mr),
        "val_map50": float(val.box.map50),
        "val_map50_95": float(val.box.map),
    })

    mlflow.log_artifact(str(best), "weights")
    for f in ("results.csv", "results.png", "confusion_matrix.png", "args.yaml"):
        if (run_dir / f).exists():
            mlflow.log_artifact(str(run_dir / f), "training_outputs")

    LAST_RUN_FILE.write_text(run.info.run_id)
    print(f"Durée : {duration/60:.1f} min | recall={val.box.mr:.3f} mAP50={val.box.map50:.3f}")