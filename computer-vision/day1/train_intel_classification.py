"""
1일차 실습과제: Intel Image Classification 이미지 분류

Intel Natural Scenes 데이터셋(6클래스: buildings, forest, glacier, mountain, sea, street)에
경량 커스텀 CNN을 처음부터(from scratch) 학습시키고, 데이터 증강 적용 여부에 따른 성능 차이를
비교한다. 각 실험은 Val Macro AUROC 기준으로 최적 모델을 선택하고, 최종적으로 Test set에서
정확도/Macro F1/Macro AUROC와 클래스별 세부 지표, 혼동행렬, 오분류 패턴을 평가한다.

실행 결과는 result/metrics.json 에 저장되고, build_report.py 가 이를 읽어 HTML 리포트를 만든다.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_recall_fscore_support,
                              roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ----------------------------------------------------------------------
# 0. 경로와 설정
# ----------------------------------------------------------------------
DATA_ROOT = Path(r"C:\Users\kimjongmin\Desktop\archive")
TRAIN_DIR = DATA_ROOT / "seg_train" / "seg_train"
TEST_DIR = DATA_ROOT / "seg_test" / "seg_test"
RESULT_DIR = Path(__file__).resolve().parent / "result"
RESULT_DIR.mkdir(exist_ok=True)

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 64
MAX_EPOCHS = 20
PATIENCE = 5
PER_CLASS_TRAIN = 800
PER_CLASS_VAL = 200
PER_CLASS_TEST = 200

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

CLASSES = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
NUM_CLASSES = len(CLASSES)
print("classes:", CLASSES)


# ----------------------------------------------------------------------
# 1. 클래스별 균형 샘플링 (train 800 / val 200 / test 200)
# ----------------------------------------------------------------------
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
    cls_train, cls_val = train_test_split(
        picked, train_size=PER_CLASS_TRAIN, random_state=SEED
    )
    train_paths += cls_train
    train_labels += [idx] * len(cls_train)
    val_paths += cls_val
    val_labels += [idx] * len(cls_val)

    test_cls_dir = TEST_DIR / cls
    picked_test = sample_paths(test_cls_dir, PER_CLASS_TEST, seed=SEED)
    test_paths += picked_test
    test_labels += [idx] * len(picked_test)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")


# ----------------------------------------------------------------------
# 2. 이미지를 메모리에 미리 로드 (매 epoch 디스크 I/O 방지)
# ----------------------------------------------------------------------
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

# 학습셋 기준 채널별 평균/표준편차 계산 (ImageNet 통계 대신 이 데이터셋 고유 통계 사용)
train_float = train_images.astype(np.float32) / 255.0
CHANNEL_MEAN = train_float.mean(axis=(0, 1, 2)).tolist()
CHANNEL_STD = train_float.std(axis=(0, 1, 2)).tolist()
print(f"channel mean={CHANNEL_MEAN}, std={CHANNEL_STD}")


class IntelSceneDataset(Dataset):
    """메모리에 올려둔 uint8 이미지 배열을 감싸는 Dataset. transform은 PIL 기준으로 적용한다."""

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
    transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
])

augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(CHANNEL_MEAN, CHANNEL_STD),
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


# ----------------------------------------------------------------------
# 3. 경량 4-Stage CNN (처음부터 학습, 전이학습 아님)
# ----------------------------------------------------------------------
class SceneCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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
    avg_loss = total_loss / total
    acc = correct / total
    auroc = macro_auroc(all_labels, all_probs)
    return avg_loss, acc, auroc


def train_experiment(name: str, train_transform):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    train_loader, val_loader, test_loader = make_loaders(train_transform)

    model = SceneCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auroc": []}
    best_auroc = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auroc = run_epoch(model, val_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auroc"].append(val_auroc)

        print(f"Epoch [{epoch:2d}/{MAX_EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auroc={val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch={best_epoch}, best val_auroc={best_auroc:.4f})")
                break

    model.load_state_dict(best_state)

    # 최종 test 평가
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

    test_acc = accuracy_score(test_labels_all, test_preds_all)
    test_macro_f1 = f1_score(test_labels_all, test_preds_all, average="macro")
    test_macro_auroc = macro_auroc(test_labels_all, test_probs_all)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels_all, test_preds_all, labels=list(range(NUM_CLASSES))
    )
    per_class_auroc = roc_auc_score(
        test_labels_all, test_probs_all, multi_class="ovr", average=None
    )
    cm = confusion_matrix(test_labels_all, test_preds_all, labels=list(range(NUM_CLASSES)))

    return {
        "name": name,
        "history": history,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "val_macro_auroc": best_auroc,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_macro_auroc": test_macro_auroc,
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "auroc": per_class_auroc.tolist(),
        },
        "confusion_matrix": cm.tolist(),
        "trainable_params": sum(p.numel() for p in model.parameters()),
    }


if __name__ == "__main__":
    results = {}
    results["baseline"] = train_experiment("Exp 1. Baseline CNN (Resize + Normalize)", base_transform)
    results["augmented"] = train_experiment(
        "Exp 2. Augmented CNN (+ HorizontalFlip + Rotation + ColorJitter)", augment_transform
    )

    output = {
        "classes": CLASSES,
        "num_classes": NUM_CLASSES,
        "channel_mean": CHANNEL_MEAN,
        "channel_std": CHANNEL_STD,
        "dataset": {
            "train": len(train_paths),
            "val": len(val_paths),
            "test": len(test_paths),
            "per_class_train": PER_CLASS_TRAIN,
            "per_class_val": PER_CLASS_VAL,
            "per_class_test": PER_CLASS_TEST,
            "image_size": IMAGE_SIZE,
        },
        "device": str(DEVICE),
        "experiments": results,
    }

    with open(RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n결과 저장 완료:", RESULT_DIR / "metrics.json")
