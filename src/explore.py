import random
from collections import Counter
import cv2
import matplotlib.pyplot as plt
from config import RAW, NAMES
from common import list_images, label_path_for, read_label_lines, yolo_to_xyxy

images = list_images(RAW)
print("Images trouvées :", len(images))

# 1) Répartition par classe (images et boîtes)
box_count, img_count, empty, orphans = Counter(), Counter(), 0, 0
for img in images:
    txt = label_path_for(img)
    if not txt.exists():
        orphans += 1
        continue
    lines = read_label_lines(txt)
    if not lines:
        empty += 1
    classes = [int(float(l.split()[0])) for l in lines]
    box_count.update(classes)
    img_count.update(set(classes))
print("Boîtes par classe :", dict(sorted(box_count.items())))
print("Images par classe :", dict(sorted(img_count.items())))
print("Annotations vides :", empty, "| Images sans .txt :", orphans)

#  2) 9 images avec boîtes
random.seed(42)
sample = random.sample([i for i in images if label_path_for(i).exists()], 9)
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for ax, path in zip(axes.ravel(), sample):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)  # OpenCV lit en BGR
    H, W = img.shape[:2]
    for line in read_label_lines(label_path_for(path)):
        c, xc, yc, w, h = line.split()
        x1, y1, x2, y2 = yolo_to_xyxy(float(xc), float(yc), float(w), float(h), W, H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        name = NAMES[int(float(c))] if int(float(c)) < len(NAMES) else str(c)
        cv2.putText(img, name, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    ax.imshow(img); ax.set_title(path.name, fontsize=8); ax.axis("off")
plt.tight_layout(); plt.savefig("explore_samples.png", dpi=120)

#  3) Histogramme
plt.figure(figsize=(6, 4))
plt.bar([str(k) for k in sorted(box_count)], [box_count[k] for k in sorted(box_count)])
plt.title("Boîtes par classe"); plt.savefig("explore_classes.png", dpi=120)
