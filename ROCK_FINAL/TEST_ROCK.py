import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import Counter
import os

# ================= PATHS =================
YOLO_MODEL = r"C:\Users\Admin\Downloads\PERFECT_ROCK_CLASS\best.pt"
H5_MODEL   = r"C:\Users\Admin\Downloads\ROCK_CLASS\rock_classifier.h5"
INPUT_IMG  = r"C:\Users\Admin\Downloads\ROCK_CLASS\images\dataset_origin\gravel\img_000505.jpg"
OUTPUT_IMG = r"C:\Users\Admin\Downloads\PERFECT_ROCK_CLASS\output.jpg"

# ================= CONFIG =================
GRID_SIZE = 4
IMG_SIZE = 224

# CORRECT CLASS ORDER (ALPHABETICAL)
CLASS_NAMES = [
    "gravel",     # 0
    "pebble",     # 1
    "red_brick",  # 2
    "unknown"     # 3
]

# ================= LOAD MODELS =================
yolo = YOLO(YOLO_MODEL)
clf = load_model(H5_MODEL)

assert len(CLASS_NAMES) == clf.output_shape[-1], \
    "❌ CLASS_NAMES mismatch with H5 output"

# ================= LOAD IMAGE =================
img = cv2.imread(INPUT_IMG)
if img is None:
    print("❌ Image load aagala")
    exit()

H, W, _ = img.shape
tile_h = H // GRID_SIZE
tile_w = W // GRID_SIZE

# ================= PREPROCESS =================
def preprocess(crop):
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop = crop.astype("float32") / 255.0
    return np.expand_dims(crop, axis=0)

# ================= 5 JUDGES =================
def five_judges(crop):
    h, w, _ = crop.shape
    judges = [
        crop,
        crop[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)],
        cv2.flip(crop, 1),
        cv2.convertScaleAbs(crop, alpha=1.2, beta=20),
        cv2.convertScaleAbs(crop, alpha=0.8, beta=-20)
    ]

    votes, confs = [], []

    for j in judges:
        j = cv2.resize(j, (w, h))
        pred = clf.predict(preprocess(j), verbose=0)
        idx = np.argmax(pred)
        votes.append(idx)
        confs.append(pred[0][idx])

    final_idx = Counter(votes).most_common(1)[0][0]
    avg_conf = sum(confs) / len(confs)

    return CLASS_NAMES[final_idx], avg_conf

# ================= MAIN PIPELINE =================
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):

        y1, y2 = r * tile_h, (r + 1) * tile_h
        x1, x2 = c * tile_w, (c + 1) * tile_w
        tile = img[y1:y2, x1:x2]

        results = yolo(tile, conf=0.15, iou=0.4)[0]
        if results.boxes is None:
            continue

        for box in results.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])

            # convert tile → original coords
            bx1 += x1; bx2 += x1
            by1 += y1; by2 += y1

            crop = img[by1:by2, bx1:bx2]
            if crop.size == 0:
                continue

            label, conf = five_judges(crop)

            # YOLO → BOX ONLY
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

            # H5 → TEXT ONLY
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (bx1, by1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

# ================= SAVE OUTPUT =================
cv2.imwrite(OUTPUT_IMG, img)
print("DONE – Full pipeline success")
