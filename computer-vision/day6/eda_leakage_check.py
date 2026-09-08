"""
6일차 파이널 프로젝트: Brain Tumor MRI Dataset - EDA / 함정(트랩) 재현

강의자료가 예고한 함정: "meningioma 클래스만 증강 이미지로 수를 채웠고, 그 증강본이
평가(Test) 폴더에도 들어 있습니다." 파일명으로 1차 확인(Tr-aug-me_*, Te-aug-me_*)했고,
여기서는 perceptual hash(phash)로 Train/Test 간 '거의 동일한 원본에서 나온' 이미지 쌍이
실제로 존재하는지(=Data Leakage) 정량적으로 확인한다.
"""

import json
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(r"C:\Users\kimjongmin\Desktop\archive")
TRAIN_DIR = ROOT / "Training"
TEST_DIR = ROOT / "Testing"
RESULT_DIR = Path(__file__).resolve().parent / "result"
RESULT_DIR.mkdir(exist_ok=True)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
HASH_DIST_THRESHOLD = 6  # phash Hamming distance <= 6 이면 "거의 동일 원본" 로 판정


def compute_hashes(folder: Path):
    out = {}
    for p in sorted(folder.glob("*")):
        try:
            out[p.name] = imagehash.phash(Image.open(p).convert("L"))
        except Exception as e:
            print("hash fail:", p, e)
    return out


print("클래스별 Train/Test 장수 및 증강본(aug) 비율 확인")
class_stats = {}
for cls in CLASSES:
    train_files = list((TRAIN_DIR / cls).glob("*"))
    test_files = list((TEST_DIR / cls).glob("*"))
    train_aug = [f for f in train_files if "aug" in f.name.lower()]
    test_aug = [f for f in test_files if "aug" in f.name.lower()]
    class_stats[cls] = {
        "train_total": len(train_files), "train_aug": len(train_aug),
        "test_total": len(test_files), "test_aug": len(test_aug),
    }
    print(f"  {cls}: train={len(train_files)}(aug {len(train_aug)}) "
          f"test={len(test_files)}(aug {len(test_aug)})")

# ------------------------------------------------------------------
# meningioma에 대해서만 Train<->Test phash 근접쌍(leakage 후보) 탐지
# (다른 클래스는 aug 이미지가 아예 없어 원본 그대로이므로 대조군으로 사용)
# ------------------------------------------------------------------
print("\nmeningioma Train/Test perceptual hash 계산 중...")
train_hashes = compute_hashes(TRAIN_DIR / "meningioma")
test_hashes = compute_hashes(TEST_DIR / "meningioma")
print(f"  train hashes: {len(train_hashes)}, test hashes: {len(test_hashes)}")

leak_pairs = []
for test_name, test_h in test_hashes.items():
    best_dist, best_train_name = 999, None
    for train_name, train_h in train_hashes.items():
        d = test_h - train_h
        if d < best_dist:
            best_dist, best_train_name = d, train_name
    if best_dist <= HASH_DIST_THRESHOLD:
        leak_pairs.append({"test": test_name, "train": best_train_name, "hash_distance": int(best_dist)})

leak_pairs.sort(key=lambda x: x["hash_distance"])
print(f"\nmeningioma Test 이미지 중 Train과 거의 동일(phash distance<={HASH_DIST_THRESHOLD}) 매칭: "
      f"{len(leak_pairs)} / {len(test_hashes)}")
for lp in leak_pairs[:10]:
    print(" ", lp)

# 대조군: notumor(증강 없음)에 대해서도 같은 검사를 돌려 "정상 범위" 확인
print("\n[대조군] notumor Train/Test perceptual hash 근접쌍 검사...")
train_hashes_ctrl = compute_hashes(TRAIN_DIR / "notumor")
test_hashes_ctrl = compute_hashes(TEST_DIR / "notumor")
leak_pairs_ctrl = []
for test_name, test_h in test_hashes_ctrl.items():
    best_dist = min((test_h - h) for h in train_hashes_ctrl.values())
    if best_dist <= HASH_DIST_THRESHOLD:
        leak_pairs_ctrl.append({"test": test_name, "hash_distance": int(best_dist)})
print(f"  notumor Test 중 Train과 거의 동일 매칭: {len(leak_pairs_ctrl)} / {len(test_hashes_ctrl)}")

output = {
    "class_stats": class_stats,
    "hash_dist_threshold": HASH_DIST_THRESHOLD,
    "meningioma_leak_pairs": leak_pairs,
    "meningioma_leak_count": len(leak_pairs),
    "meningioma_test_total": len(test_hashes),
    "control_notumor_leak_count": len(leak_pairs_ctrl),
    "control_notumor_test_total": len(test_hashes_ctrl),
}
with open(RESULT_DIR / "eda_leakage.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n저장 완료:", RESULT_DIR / "eda_leakage.json")
