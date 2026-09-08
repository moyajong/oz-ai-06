"""
학습 완료 후 실행: 최종 선택 모델(best_key)로 Internal Test 이미지 몇 장에 대해
예측 vs 정답 바운딩박스를 나란히 그린 비교 이미지를 만들고, 간단한 FP/FN 사례를 집계한다.
(과제 요구 산출물 2, 3: "예측 vs 정답 이미지", "FP·FN 사례 코멘트")
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "yolo_dataset"
RESULT_DIR = ROOT / "result"

with open(RESULT_DIR / "metrics.json", encoding="utf-8") as f:
    M = json.load(f)

best_key = M["best_key"]
best_weights = M["experiments"][best_key]["best_weights"]
imgsz = M["experiments"][best_key]["imgsz"]
class_names = M["dataset"]["classes"]

model = YOLO(best_weights)

IOU_MATCH = 0.5
CONF_THRES = 0.25

test_img_dir = DATA_DIR / "images" / "test"
test_lbl_dir = DATA_DIR / "labels" / "test"
img_paths = sorted(test_img_dir.glob("*.png"))

COLORS = [
    (239, 68, 68), (59, 130, 246), (16, 185, 129), (245, 158, 11), (168, 85, 247),
    (236, 72, 153), (20, 184, 166), (234, 179, 8), (99, 102, 241), (34, 197, 94),
    (244, 63, 94), (14, 165, 233), (163, 230, 53), (251, 146, 60),
]


def load_gt_boxes(img_id, w, h):
    lbl_path = test_lbl_dir / f"{img_id}.txt"
    boxes = []
    if lbl_path.exists() and lbl_path.stat().st_size > 0:
        for line in lbl_path.read_text().strip().splitlines():
            cls, xc, yc, bw, bh = map(float, line.split())
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            boxes.append((int(cls), x1, y1, x2, y2))
    return boxes


def iou(b1, b2):
    x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def draw_boxes(img, boxes, label_fn, color_fn):
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        cls = b[0]
        color = color_fn(cls)
        draw.rectangle(b[1:5], outline=color, width=3)
        draw.text((b[1] + 2, max(0, b[2] - 14) if False else b[2] - 14), label_fn(cls), fill=color)
    return img


# ------------------------------------------------------------------
# 1. FP/FN 집계 (Internal Test 전체, confidence 0.25 / IoU 0.5 매칭)
# ------------------------------------------------------------------
total_tp, total_fp, total_fn = 0, 0, 0
per_class_fp = {c: 0 for c in class_names}
per_class_fn = {c: 0 for c in class_names}
example_records = []  # (img_id, gt_boxes, pred_boxes, fp_count, fn_count)

results_gen = model.predict(source=str(test_img_dir), imgsz=imgsz, conf=CONF_THRES, verbose=False, stream=True)

for r in results_gen:
    img_id = Path(r.path).stem
    h, w = r.orig_shape
    gt_boxes = load_gt_boxes(img_id, w, h)
    pred_boxes = []
    if r.boxes is not None:
        for b in r.boxes:
            cls = int(b.cls.item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            pred_boxes.append((cls, x1, y1, x2, y2))

    matched_gt = set()
    img_fp, img_fn = 0, 0
    for p in pred_boxes:
        best_iou, best_j = 0, -1
        for j, g in enumerate(gt_boxes):
            if j in matched_gt or g[0] != p[0]:
                continue
            v = iou(p[1:5], g[1:5])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= IOU_MATCH:
            matched_gt.add(best_j)
            total_tp += 1
        else:
            total_fp += 1
            img_fp += 1
            per_class_fp[class_names[p[0]]] += 1
    for j, g in enumerate(gt_boxes):
        if j not in matched_gt:
            total_fn += 1
            img_fn += 1
            per_class_fn[class_names[g[0]]] += 1

    example_records.append({
        "img_id": img_id, "gt": gt_boxes, "pred": pred_boxes,
        "fp": img_fp, "fn": img_fn, "path": r.path,
    })

print(f"Internal Test 전체: TP={total_tp} FP={total_fp} FN={total_fn}")

# ------------------------------------------------------------------
# 2. 예측 vs 정답 비교 이미지 (대표 사례 4장: FN 많은 사례 2장 + 잘 맞은 사례 2장)
# ------------------------------------------------------------------
examples_dir = RESULT_DIR / "examples"
examples_dir.mkdir(exist_ok=True)

by_fn = sorted([e for e in example_records if e["gt"]], key=lambda e: -(e["fp"] + e["fn"]))
worst = by_fn[:2]
best_matches = sorted([e for e in example_records if e["gt"]], key=lambda e: (e["fp"] + e["fn"]))[:2]
chosen = worst + best_matches

saved_examples = []
for rec in chosen:
    img = Image.open(rec["path"])
    gt_img = draw_boxes(img, rec["gt"], lambda c: class_names[c], lambda c: COLORS[c % len(COLORS)])
    pred_img = draw_boxes(img, rec["pred"], lambda c: class_names[c], lambda c: COLORS[c % len(COLORS)])

    combo = Image.new("RGB", (img.width * 2 + 10, img.height), "white")
    combo.paste(gt_img, (0, 0))
    combo.paste(pred_img, (img.width + 10, 0))
    out_path = examples_dir / f"{rec['img_id']}_gt_vs_pred.png"
    combo.save(out_path)
    saved_examples.append({
        "img_id": rec["img_id"], "file": out_path.name,
        "n_gt": len(rec["gt"]), "n_pred": len(rec["pred"]),
        "fp": rec["fp"], "fn": rec["fn"],
    })
    print("saved:", out_path.name, "fp=", rec["fp"], "fn=", rec["fn"])

output = {
    "iou_match_threshold": IOU_MATCH,
    "conf_threshold": CONF_THRES,
    "total": {"tp": total_tp, "fp": total_fp, "fn": total_fn},
    "per_class_fp": per_class_fp,
    "per_class_fn": per_class_fn,
    "examples": saved_examples,
}
with open(RESULT_DIR / "fp_fn_analysis.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n저장 완료:", RESULT_DIR / "fp_fn_analysis.json")
