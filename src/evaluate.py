import argparse
import json
import shutil
import cv2
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from ultralytics import YOLO
from config import DATA_YAML, RUNS_DIR, NAMES, EXPORT, MLFLOW_URI, LAST_RUN_FILE
from common import label_path_for, read_label_lines, yolo_to_xyxy, iou, list_images
from config import DATASET

p = argparse.ArgumentParser()
p.add_argument("--name", required=True, help="nom du run d'entraînement (dossier runs/detect/<name>)")
p.add_argument("--conf", type=float, default=0.25)
args = p.parse_args()

best = RUNS_DIR / args.name / "weights" / "best.pt"
model = YOLO(str(best))


res = model.val(data=str(DATA_YAML), split="test", plots=True,
                project=str(RUNS_DIR), name=f"{args.name}_test", exist_ok=True)
print(f"Global : P={res.box.mp:.3f} R={res.box.mr:.3f} mAP50={res.box.map50:.3f} mAP50-95={res.box.map:.3f}")

rows = []
for i, c in enumerate(res.box.ap_class_index):
    p_, r_, ap50, ap = res.box.class_result(i)
    rows.append({"classe": NAMES[int(c)], "precision": p_, "rappel": r_, "mAP50": ap50, "mAP50-95": ap})
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
print("Graphiques dans :", RUNS_DIR / f"{args.name}_test")


if LAST_RUN_FILE.exists():
    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_id=LAST_RUN_FILE.read_text().strip()):
        mlflow.log_metrics({"test_precision": float(res.box.mp), "test_recall": float(res.box.mr),
                            "test_map50": float(res.box.map50), "test_map50_95": float(res.box.map)})


cases = {"reussite": [], "faux_positif": [], "faux_negatif": []}
for img_path in list_images(DATASET / "images" / "test"):
    img = cv2.imread(str(img_path)); H, W = img.shape[:2]
    gts = []
    for line in read_label_lines(label_path_for(img_path)):
        c, xc, yc, w, h = line.split()
        gts.append((int(c), yolo_to_xyxy(float(xc), float(yc), float(w), float(h), W, H)))
    r = model.predict(img, conf=args.conf, verbose=False)[0]
    preds = [(int(b.cls), tuple(map(int, b.xyxy[0].tolist())), float(b.conf)) for b in r.boxes]

    matched_gt, fp = set(), 0
    best_ious = []
    for pc, pb, ps in preds:
        scores = [(iou(pb, gb), j) for j, (gc, gb) in enumerate(gts) if gc == pc and j not in matched_gt]
        if scores and max(scores)[0] >= 0.5:
            v, j = max(scores); matched_gt.add(j); best_ious.append(v)
        else:
            fp += 1
    fn = len(gts) - len(matched_gt)
    item = (img_path, gts, preds, best_ious)
    if fn > 0:
        cases["faux_negatif"].append(item)
    elif fp > 0:
        cases["faux_positif"].append(item)
    elif gts:
        cases["reussite"].append(item)

print({k: len(v) for k, v in cases.items()})

def show(kind, n=6):
    items = cases[kind][:n]
    if not items:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (path, gts, preds, ious) in zip(axes.ravel(), items):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        for gc, gb in gts:                                # réel = vert
            cv2.rectangle(img, gb[:2], gb[2:], (0, 200, 0), 2)
        for pc, pb, ps in preds:                          # prédit = rouge
            cv2.rectangle(img, pb[:2], pb[2:], (255, 0, 0), 2)
            cv2.putText(img, f"{NAMES[pc]} {ps:.2f}", (pb[0], max(pb[1] - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        iou_txt = f"IoU={ious[0]:.2f}" if ious else ""
        ax.imshow(img); ax.set_title(f"{kind} {iou_txt}", fontsize=9); ax.axis("off")
    plt.tight_layout(); plt.savefig(f"analyse_{kind}.png", dpi=110); plt.close()

for k in cases:
    show(k)


low = [(p.name, round(v, 2)) for p, _, _, ious in cases["reussite"] for v in ious if v < 0.6]
print("Boîtes peu précises (IoU < 0.6) :", low[:10])


EXPORT.mkdir(exist_ok=True)
shutil.copy2(best, EXPORT / "best.pt")
shutil.copy2(DATA_YAML, EXPORT / "data.yaml")
(EXPORT / "classes.json").write_text(json.dumps(NAMES, ensure_ascii=False, indent=2))
(EXPORT / "inference_params.json").write_text(json.dumps({"imgsz": 640, "conf": args.conf, "iou_nms": 0.7}, indent=2))
print("Export OK dans", EXPORT)