import os
from pathlib import Path
from config import IMG_EXT, NC


def label_path_for(img):
    s = str(img).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    cand = Path(s).with_suffix(".txt")
    return cand if cand.exists() else img.with_suffix(".txt")


def read_label_lines(txt):
    return [l.strip() for l in txt.read_text().splitlines() if l.strip()]


def validate_label(txt, nc: int = NC):
    for i, line in enumerate(read_label_lines(txt), 1):
        parts = line.split()
        if len(parts) != 5:
            return False, f"ligne {i}: {len(parts)} valeurs au lieu de 5"
        try:
            cls = int(parts[0])
            coords = [float(p) for p in parts[1:]]
        except ValueError:
            return False, f"ligne {i}: valeur non numérique / classe non entière"
        if not 0 <= cls < nc:
            return False, f"ligne {i}: classe {cls} invalide"
        if any(not 0.0 <= c <= 1.0 for c in coords):
            return False, f"ligne {i}: coordonnées hors [0,1]"
    return True, ""


def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
    x2, y2 = (xc + w / 2) * W, (yc + h / 2) * H
    return int(x1), int(y1), int(x2), int(y2)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]