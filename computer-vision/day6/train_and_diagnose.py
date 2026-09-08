"""
6일차 파이널 자유주제 프로젝트: Brain Tumor MRI Dataset

가설 (EDA에서 관측):
  meningioma 클래스만 학습/평가 폴더 모두에 증강 이미지가 섞여 있다
  (Train: 1400장 중 100장이 aug, Test: 400장 중 103장이 aug — 다른 3개 클래스는 aug 0장).
  같은 "meningioma test accuracy" 라는 숫자 안에 성격이 다른 두 그룹(진짜 원본 vs 합성 증강본)이
  섞여 있으므로, 이 숫자는 다른 3개 클래스의 (100% 원본) test accuracy와 액면 그대로 비교할 수 없다.
  이 스크립트는 ①베이스라인을 그대로 학습/평가하고, ②meningioma test를 aug/원본으로 쪼개
  두 그룹의 성능 차이를 실측하여 가설을 검증한 뒤, ③"처방"으로 meningioma는 원본 이미지만으로
  재평가한 공정한 지표를 함께 제시한다 (재학습 불필요 — 평가 프로토콜만 교정).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(r"C:\Users\kimjongmin\Desktop\archive")
TRAIN_DIR = ROOT / "Training"
TEST_DIR = ROOT / "Testing"
RESULT_DIR = Path(__file__).resolve().parent / "result"
RESULT_DIR.mkdir(exist_ok=True)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 15
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

torch.manual_seed(SEED)
np.random.seed(SEED)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def list_files(folder: Path):
    return sorted(folder.glob("*"))


# ----------------------------------------------------------------------
# 1. 데이터 구성: 원본 그대로의 Train(5,600) / 공식 Test(1,600)
#    + Train 내부에서 Val 분리(모델 선택용, 공식 Test는 손대지 않음)
# ----------------------------------------------------------------------
train_paths, train_labels = [], []
official_test_paths, official_test_labels = [], []
for idx, cls in enumerate(CLASSES):
    for p in list_files(TRAIN_DIR / cls):
        train_paths.append(p)
        train_labels.append(idx)
    for p in list_files(TEST_DIR / cls):
        official_test_paths.append(p)
        official_test_labels.append(idx)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.1, random_state=SEED, stratify=train_labels
)
print(f"Train {len(train_paths)} / Val {len(val_paths)} / Official Test {len(official_test_paths)}")


class MriDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_loader = DataLoader(MriDataset(train_paths, train_labels, train_transform),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(MriDataset(val_paths, val_labels, eval_transform),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
official_test_loader = DataLoader(MriDataset(official_test_paths, official_test_labels, eval_transform),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ----------------------------------------------------------------------
# 2. 모델 (ResNet18, ImageNet 사전학습 전이학습 — 2일차와 동일 계열)
# ----------------------------------------------------------------------
def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


print("\n" + "=" * 60 + "\nBaseline 학습 (주어진 Train 그대로, meningioma aug 포함)\n" + "=" * 60)
model = build_model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

history = {"train_loss": [], "val_loss": [], "val_acc": []}
best_acc, best_state, patience_ctr = -1, None, 0
t0 = time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    print(f"Epoch [{epoch:2d}/{MAX_EPOCHS}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    if val_acc > best_acc:
        best_acc, best_state, patience_ctr = val_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best val_acc={best_acc:.4f})")
            break
train_seconds = time.time() - t0
model.load_state_dict(best_state)


# ----------------------------------------------------------------------
# 3. 공식 Test 평가 (baseline, meningioma에 aug 섞인 채로 — "액면 숫자")
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_all(model, paths, labels):
    model.eval()
    ds = MriDataset(paths, labels, eval_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_pred, all_true = [], []
    for x, y in loader:
        out = model(x.to(DEVICE))
        all_pred.append(out.argmax(1).cpu().numpy())
        all_true.append(y.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


y_true, y_pred = predict_all(model, official_test_paths, official_test_labels)
official_acc = accuracy_score(y_true, y_pred)
official_macro_f1 = f1_score(y_true, y_pred, average="macro")
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(CLASSES))))
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

print(f"\n[Baseline 공식 Test] acc={official_acc:.4f} macro_f1={official_macro_f1:.4f}")
for i, cls in enumerate(CLASSES):
    print(f"  {cls}: precision={precision[i]:.4f} recall={recall[i]:.4f} f1={f1[i]:.4f}")

# ----------------------------------------------------------------------
# 4. 진단: meningioma Test를 aug / 원본으로 쪼개서 성능 비교 (가설 검증)
# ----------------------------------------------------------------------
me_test_dir = TEST_DIR / "meningioma"
me_aug_paths = [p for p in list_files(me_test_dir) if "aug" in p.name.lower()]
me_orig_paths = [p for p in list_files(me_test_dir) if "aug" not in p.name.lower()]
me_idx = CLASSES.index("meningioma")

y_true_aug, y_pred_aug = predict_all(model, me_aug_paths, [me_idx] * len(me_aug_paths))
y_true_orig, y_pred_orig = predict_all(model, me_orig_paths, [me_idx] * len(me_orig_paths))
acc_aug = accuracy_score(y_true_aug, y_pred_aug)
acc_orig = accuracy_score(y_true_orig, y_pred_orig)
print(f"\n[진단] meningioma Test 세부: 증강본(n={len(me_aug_paths)}) acc={acc_aug:.4f} | "
      f"원본(n={len(me_orig_paths)}) acc={acc_orig:.4f} | 격차={acc_aug - acc_orig:+.4f}")

# ----------------------------------------------------------------------
# 5. 처방: "공정 평가 프로토콜" — meningioma는 원본 이미지만으로 재평가
#    (다른 3개 클래스는 이미 100% 원본이라 그대로, 재학습 없이 평가만 교정)
# ----------------------------------------------------------------------
clean_paths, clean_labels = [], []
for idx, cls in enumerate(CLASSES):
    if cls == "meningioma":
        clean_paths += me_orig_paths
        clean_labels += [idx] * len(me_orig_paths)
    else:
        ps = list_files(TEST_DIR / cls)
        clean_paths += ps
        clean_labels += [idx] * len(ps)

y_true_clean, y_pred_clean = predict_all(model, clean_paths, clean_labels)
clean_acc = accuracy_score(y_true_clean, y_pred_clean)
clean_macro_f1 = f1_score(y_true_clean, y_pred_clean, average="macro")
precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(
    y_true_clean, y_pred_clean, labels=list(range(len(CLASSES)))
)
cm_clean = confusion_matrix(y_true_clean, y_pred_clean, labels=list(range(len(CLASSES))))

print(f"\n[처방 후: 공정 평가] acc={clean_acc:.4f} macro_f1={clean_macro_f1:.4f}")
for i, cls in enumerate(CLASSES):
    print(f"  {cls}: precision={precision_c[i]:.4f} recall={recall_c[i]:.4f} f1={f1_c[i]:.4f}")

output = {
    "dataset": {
        "source": "Brain Tumor MRI Dataset (masoudnickparvar)",
        "classes": CLASSES,
        "train": len(train_paths), "val": len(val_paths), "official_test": len(official_test_paths),
    },
    "model": "ResNet18 (ImageNet pretrained, fine-tuned)",
    "train_seconds": train_seconds,
    "history": history,
    "eda_finding": {
        "note": "meningioma class only: Training 1400장 중 100장, Testing 400장 중 103장이 증강(aug) 이미지. "
                "다른 3개 클래스는 aug 0장 — 100% 원본.",
    },
    "baseline_official_test": {
        "accuracy": official_acc, "macro_f1": official_macro_f1,
        "per_class": {"precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist()},
        "confusion_matrix": cm.tolist(),
    },
    "diagnosis_meningioma_subgroups": {
        "n_aug": len(me_aug_paths), "acc_aug": acc_aug,
        "n_orig": len(me_orig_paths), "acc_orig": acc_orig,
        "gap": acc_aug - acc_orig,
    },
    "prescription_clean_eval": {
        "description": "meningioma Test에서 증강본 103장을 제외하고 원본 297장만으로 재평가 "
                        "(다른 3개 클래스는 그대로, 재학습 없이 평가만 교정)",
        "accuracy": clean_acc, "macro_f1": clean_macro_f1,
        "per_class": {"precision": precision_c.tolist(), "recall": recall_c.tolist(), "f1": f1_c.tolist()},
        "confusion_matrix": cm_clean.tolist(),
        "n_test": len(clean_paths),
    },
}
with open(RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n결과 저장 완료:", RESULT_DIR / "metrics.json")
