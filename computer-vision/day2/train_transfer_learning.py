"""
2일차 자유과제: 이미지 분류(전이학습) - 실험한 이미지 분류 성능을 최대한 높게 찍어보기

1일차와 동일한 Intel Image Classification 데이터셋(6클래스, 동일 seed/split)에
ImageNet 사전학습 ResNet18을 적용해 정확도를 최대한 끌어올린다.
Feature Extraction(백본 동결, FC head만 학습)과 Fine-tuning(전체 미세조정)을
비교하고, Fine-tuning은 Feature Extraction의 best 가중치에서 이어서 학습한다
(warm start로 적은 epoch에도 빠르게 수렴하도록).

결과는 result/metrics.json 에 저장하고, build_capture.py 가 이를 읽어
과제 제출용 결과 요약 이미지(capture)를 만든다.
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ----------------------------------------------------------------------
# 0. 경로와 설정 (1일차와 동일한 데이터/분할/seed)
# ----------------------------------------------------------------------
DATA_ROOT = Path(r"C:\Users\kimjongmin\Desktop\archive")
TRAIN_DIR = DATA_ROOT / "seg_train" / "seg_train"
TEST_DIR = DATA_ROOT / "seg_test" / "seg_test"
RESULT_DIR = Path(__file__).resolve().parent / "result"
RESULT_DIR.mkdir(exist_ok=True)

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 64
PER_CLASS_TRAIN = 800
PER_CLASS_VAL = 200
PER_CLASS_TEST = 200

FE_MAX_EPOCHS = 8
FE_PATIENCE = 3
FT_MAX_EPOCHS = 8
FT_PATIENCE = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(max(1, torch.get_num_threads()))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

CLASSES = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
NUM_CLASSES = len(CLASSES)
print("classes:", CLASSES)


def sample_paths(class_dir: Path, n: int, seed: int) -> list[Path]:
    all_paths = sorted(class_dir.glob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(all_paths)
    return all_paths[:n]


train_paths, train_labels = [], []
val_paths, val_labels = [], []
test_paths, test_labels = [], []

for idx, cls in enumerate(CLASSES):
    cls_dir = TRAIN_DIR / cls
    picked = sample_paths(cls_dir, PER_CLASS_TRAIN + PER_CLASS_VAL, seed=SEED)
    cls_train, cls_val = train_test_split(picked, train_size=PER_CLASS_TRAIN, random_state=SEED)
    train_paths += cls_train
    train_labels += [idx] * len(cls_train)
    val_paths += cls_val
    val_labels += [idx] * len(cls_val)

    test_cls_dir = TEST_DIR / cls
    picked_test = sample_paths(test_cls_dir, PER_CLASS_TEST, seed=SEED)
    test_paths += picked_test
    test_labels += [idx] * len(picked_test)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")


def load_images(paths: list[Path]) -> np.ndarray:
    arr = np.zeros((len(paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr[i] = np.array(img)
    return arr


print("이미지 로딩 중...")
train_images = load_images(train_paths)
val_images = load_images(val_paths)
test_images = load_images(test_paths)
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)
print("로딩 완료.")

# ImageNet 사전학습 모델이므로 ImageNet 통계를 그대로 사용한다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class IntelSceneDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        img = self.transform(img)
        return img, self.labels[idx]


base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def make_loaders(train_transform):
    train_ds = IntelSceneDataset(train_images, train_labels, train_transform)
    val_ds = IntelSceneDataset(val_images, val_labels, base_transform)
    test_ds = IntelSceneDataset(test_images, test_labels, base_transform)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    return total_loss / total, correct / total, macro_auroc(all_labels, all_probs)


def evaluate_test(model, test_loader):
    test_labels_all, test_probs_all = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            probs = torch.softmax(model(images), dim=1).cpu().numpy()
            test_probs_all.append(probs)
            test_labels_all.append(labels.numpy())
    test_labels_all = np.concatenate(test_labels_all)
    test_probs_all = np.concatenate(test_probs_all)
    test_preds_all = test_probs_all.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels_all, test_preds_all, labels=list(range(NUM_CLASSES))
    )
    per_class_auroc = roc_auc_score(test_labels_all, test_probs_all, multi_class="ovr", average=None)
    cm = confusion_matrix(test_labels_all, test_preds_all, labels=list(range(NUM_CLASSES)))

    return {
        "test_accuracy": accuracy_score(test_labels_all, test_preds_all),
        "test_macro_f1": f1_score(test_labels_all, test_preds_all, average="macro"),
        "test_macro_auroc": macro_auroc(test_labels_all, test_probs_all),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "auroc": per_class_auroc.tolist(),
        },
        "confusion_matrix": cm.tolist(),
    }


def train_loop(model, name, optimizer, max_epochs, patience, train_transform):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    train_loader, val_loader, test_loader = make_loaders(train_transform)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auroc": []}
    best_auroc = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = 0
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auroc = run_epoch(model, val_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auroc"].append(val_auroc)

        print(f"Epoch [{epoch:2d}/{max_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auroc={val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch={best_epoch}, best val_auroc={best_auroc:.4f})")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_test(model, test_loader)

    return {
        "name": name,
        "history": history,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "val_macro_auroc": best_auroc,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **test_metrics,
    }, model


if __name__ == "__main__":
    results = {}

    # ---- Exp 1. Feature Extraction: 백본 동결, FC head만 학습 ----
    fe_model = build_model()
    for param in fe_model.parameters():
        param.requires_grad = False
    for param in fe_model.fc.parameters():
        param.requires_grad = True
    fe_optimizer = optim.Adam(fe_model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    results["feature_extraction"], fe_model = train_loop(
        fe_model, "Exp 1. ResNet18 Feature Extraction (backbone frozen)",
        fe_optimizer, FE_MAX_EPOCHS, FE_PATIENCE, augment_transform,
    )

    # ---- Exp 2. Fine-tuning: FE의 best 가중치에서 전체 미세조정 (warm start) ----
    ft_model = build_model()
    ft_model.load_state_dict(copy.deepcopy(fe_model.state_dict()))
    for param in ft_model.parameters():
        param.requires_grad = True
    ft_optimizer = optim.Adam(ft_model.parameters(), lr=1e-4, weight_decay=1e-4)
    results["fine_tuning"], ft_model = train_loop(
        ft_model, "Exp 2. ResNet18 Fine-tuning (full unfreeze, warm start from FE)",
        ft_optimizer, FT_MAX_EPOCHS, FT_PATIENCE, augment_transform,
    )

    output = {
        "classes": CLASSES,
        "num_classes": NUM_CLASSES,
        "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "dataset": {
            "train": len(train_paths), "val": len(val_paths), "test": len(test_paths),
            "per_class_train": PER_CLASS_TRAIN, "per_class_val": PER_CLASS_VAL, "per_class_test": PER_CLASS_TEST,
            "image_size": IMAGE_SIZE,
        },
        "device": str(DEVICE),
        "day1_reference": {
            "note": "1일차 동일 데이터/분할 기준 처음부터 학습한 커스텀 CNN 결과 (result/metrics.json)",
            "baseline_test_accuracy": 0.7942,
            "augmented_test_accuracy": 0.8350,
        },
        "experiments": results,
    }

    with open(RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n결과 저장 완료:", RESULT_DIR / "metrics.json")
