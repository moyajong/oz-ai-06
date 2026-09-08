"""
3일차 자유과제: 객체 탐지 (VinBigData 흉부 X-ray)

강사님 제공 실습("Day 3 실습: config로 반복하는 객체 탐지")의 학습 흐름을 따른다.
  1. 흉부 X-ray에서 기본 2,000장을 비율(No finding vs 소견 있음) 보존 방식으로 추출한다.
  2. 같은 train/validation/internal-test 분할을 모든 실험에 재사용한다 (seed=42 고정).
  3. VinBigData 원본 bbox 좌표(원본 해상도 기준)를 512x512 PNG 기준으로 재조정해 YOLO 포맷으로 변환한다.

이 스크립트는 YOLO 학습용 디렉터리 구조(images/labels train·val·test + data.yaml)를 생성한다.
"""

import random
import shutil
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

SRC_ROOT = Path(r"C:\Users\kimjongmin\Desktop\archive (1)\vinbigdata")
TRAIN_CSV = SRC_ROOT / "train.csv"
TRAIN_IMG_DIR = SRC_ROOT / "train"

OUT_ROOT = Path(__file__).resolve().parent / "yolo_dataset"
IMG_SIZE = 512
SEED = 42
N_SAMPLE = 2000
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}  # internal test = "test"

random.seed(SEED)

df = pd.read_csv(TRAIN_CSV)

# 클래스: class_id 0~13 = 실제 소견, 14 = No finding(배경, bbox 없음)
classes_df = df[df["class_id"] != 14][["class_id", "class_name"]].drop_duplicates()
CLASS_NAMES = [None] * 14
for _, row in classes_df.iterrows():
    CLASS_NAMES[int(row["class_id"])] = row["class_name"]
print("classes:", CLASS_NAMES)

no_finding_images = set(df[df["class_name"] == "No finding"]["image_id"])
all_images = sorted(df["image_id"].unique())
finding_images = sorted(set(all_images) - no_finding_images)
no_finding_images = sorted(no_finding_images)

# 원본 비율(약 70.7% No finding / 29.3% 소견 있음)을 유지한 채 2,000장 샘플링
n_no_finding = round(N_SAMPLE * len(no_finding_images) / len(all_images))
n_finding = N_SAMPLE - n_no_finding

rng = random.Random(SEED)
sampled_no_finding = rng.sample(no_finding_images, n_no_finding)
sampled_finding = rng.sample(finding_images, n_finding)
sampled_images = sampled_no_finding + sampled_finding
print(f"샘플링: No finding {n_no_finding}장 + 소견 있음 {n_finding}장 = 총 {len(sampled_images)}장")

# stratify 라벨(0=no finding, 1=finding)로 고정 train/val/test 분할
labels_for_split = [0] * len(sampled_no_finding) + [1] * len(sampled_finding)
train_ids, temp_ids, train_lbl, temp_lbl = train_test_split(
    sampled_images, labels_for_split,
    train_size=SPLIT_RATIOS["train"], random_state=SEED, stratify=labels_for_split,
)
val_ratio_of_temp = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"])
val_ids, test_ids, _, _ = train_test_split(
    temp_ids, temp_lbl,
    train_size=val_ratio_of_temp, random_state=SEED, stratify=temp_lbl,
)
print(f"Train {len(train_ids)} / Val {len(val_ids)} / Internal Test {len(test_ids)}")

split_map = {"train": train_ids, "val": val_ids, "test": test_ids}

# 이미지별 bbox 그룹 (No finding 이미지는 그룹 없음 -> 빈 라벨 파일)
bbox_groups = {
    img_id: g for img_id, g in df[df["class_id"] != 14].groupby("image_id")
}

for split, ids in split_map.items():
    img_out = OUT_ROOT / "images" / split
    lbl_out = OUT_ROOT / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_id in ids:
        src_img = TRAIN_IMG_DIR / f"{img_id}.png"
        shutil.copy(src_img, img_out / f"{img_id}.png")

        label_lines = []
        if img_id in bbox_groups:
            g = bbox_groups[img_id]
            orig_w = g["width"].iloc[0]
            orig_h = g["height"].iloc[0]
            sx, sy = IMG_SIZE / orig_w, IMG_SIZE / orig_h
            for _, row in g.iterrows():
                xmin, ymin, xmax, ymax = row["x_min"] * sx, row["y_min"] * sy, row["x_max"] * sx, row["y_max"] * sy
                xc = ((xmin + xmax) / 2) / IMG_SIZE
                yc = ((ymin + ymax) / 2) / IMG_SIZE
                w = (xmax - xmin) / IMG_SIZE
                h = (ymax - ymin) / IMG_SIZE
                label_lines.append(f"{int(row['class_id'])} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        (lbl_out / f"{img_id}.txt").write_text("\n".join(label_lines), encoding="utf-8")

data_yaml = {
    "path": str(OUT_ROOT),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES,
}
with open(OUT_ROOT / "data.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

print("\nYOLO 데이터셋 구성 완료:", OUT_ROOT)
print("data.yaml:", OUT_ROOT / "data.yaml")
