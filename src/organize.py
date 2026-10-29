import shutil
from collections import Counter
from pathlib import Path
import cv2
import yaml
from sklearn.model_selection import train_test_split
from config import RAW, DATASET, DATA_YAML, NAMES, NC, IMG_EXT
from common import label_path_for, read_label_lines, validate_label

KEEP_EMPTY = True
SEED = 42

def split_name(path: Path):
    parts = [p.lower() for p in path.relative_to(RAW).parts]
    for key, out in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")):
        if key in parts:
            return out
    return None

stats = Counter()
valid_items = []

for img in RAW.rglob("*"):
    if not img.is_file():
        continue
    if img.suffix.lower() not in IMG_EXT:
        if img.suffix.lower() != ".txt":
            stats["extension_non_conforme"] += 1
        continue
    txt = label_path_for(img)
    if not txt.exists():
        stats["orphelin_sans_txt"] += 1
        continue
    try:
        if cv2.imread(str(img)) is None:
            raise ValueError("image illisible")
    except Exception:
        stats["image_corrompue"] += 1
        continue
    ok, reason = validate_label(txt)
    if not ok:
        stats["annotation_invalide"] += 1
        print("INVALIDE :", txt.name, reason)
        continue
    lines = read_label_lines(txt)
    if not lines and not KEEP_EMPTY:
        stats["annotation_vide_ecartee"] += 1
        continue
    if not lines:
        stats["annotation_vide_gardee"] += 1
    main_cls = int(lines[0].split()[0]) if lines else -1
    valid_items.append((img, txt, split_name(img), main_cls))

print("Bilan nettoyage :", dict(stats), "| valides :", len(valid_items))

#  Découpage
by_split = {"train": [], "val": [], "test": []}
has_splits = any(s is not None for _, _, s, _ in valid_items)
if has_splits:
    for it in valid_items:
        by_split[it[2] or "train"].append(it)
    if not by_split["test"] or not by_split["val"]:
        # Il manque un split : on le crée depuis train
        train = by_split["train"]
        strat = [c for *_, c in train]
        rest_ratio = 0.30 if not by_split["val"] and not by_split["test"] else 0.15
        train, tmp = train_test_split(train, test_size=rest_ratio, random_state=SEED, stratify=strat)
        by_split["train"] = train
        if not by_split["val"] and not by_split["test"]:
            tmp_strat = [c for *_, c in tmp]
            by_split["val"], by_split["test"] = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=tmp_strat)
        elif not by_split["val"]:
            by_split["val"] = tmp
        else:
            by_split["test"] = tmp
else:
    strat = [c for *_, c in valid_items]
    train, tmp = train_test_split(valid_items, test_size=0.30, random_state=SEED, stratify=strat)
    tmp_strat = [c for *_, c in tmp]
    val, test = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=tmp_strat)
    by_split = {"train": train, "val": val, "test": test}

#  Copie (image + label toujours ensemble)
if DATASET.exists():
    shutil.rmtree(DATASET)
for split, items in by_split.items():
    (DATASET / "images" / split).mkdir(parents=True)
    (DATASET / "labels" / split).mkdir(parents=True)
    for img, txt, _, _ in items:
        shutil.copy2(img, DATASET / "images" / split / img.name)
        shutil.copy2(txt, DATASET / "labels" / split / (img.stem + ".txt"))

#  Contrôle de concordance + fuite
names_per_split = {}
for split in by_split:
    n_img = len(list((DATASET / "images" / split).iterdir()))
    n_lab = len(list((DATASET / "labels" / split).iterdir()))
    names_per_split[split] = {p.stem for p in (DATASET / "images" / split).iterdir()}
    cls = Counter(c for *_, c in by_split[split])
    print(f"{split:5s} images={n_img} labels={n_lab} OK={n_img == n_lab} classes={dict(sorted(cls.items()))}")
assert not (names_per_split["train"] & names_per_split["val"]), "Fuite train/val !"
assert not (names_per_split["train"] & names_per_split["test"]), "Fuite train/test !"
assert not (names_per_split["val"] & names_per_split["test"]), "Fuite val/test !"

#  data.yaml
cfg = {
    "path": str(DATASET),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": NC,
    "names": {i: n for i, n in enumerate(NAMES)}
}
DATA_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("data.yaml écrit :", DATA_YAML)