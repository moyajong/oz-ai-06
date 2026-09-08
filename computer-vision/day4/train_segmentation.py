"""
4일차 자유과제: 이미지 분할(Semantic Segmentation) - 내가 찾은 데이터로 분할하기

데이터: Chest X-ray Masks and Labels (Montgomery + Shenzhen 흉부 X-ray 폐 영역 이진 마스크)
강의 baseline_semantic.ipynb 흐름을 따라, 같은 데이터/분할/seed/epoch/optimizer/loss 조건에서
1) BasicUNet (skip connection 있음)
2) BasicUNet (skip connection 없음, bottleneck 정보만으로 복원)
3) SMP U-Net (ResNet34, ImageNet 사전학습)
세 모델을 학습하고 val Dice/mIoU를 비교한 뒤, 최종 모델로 internal test를 1회 평가한다.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT = Path(r"C:\Users\kimjongmin\Desktop\archive (1)\Lung Segmentation")
IMG_DIR = ROOT / "CXR_png"
MASK_DIR = ROOT / "masks"
RESULT_DIR = Path(__file__).resolve().parent / "result"
RESULT_DIR.mkdir(exist_ok=True)

SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# 1. 데이터 분할 (이미지-마스크 짝이 있는 704장, 70/15/15 고정 분할)
# ----------------------------------------------------------------------
mask_paths = sorted(MASK_DIR.glob("*_mask.png"))
pairs = []
for mp in mask_paths:
    img_id = mp.stem.replace("_mask", "")
    ip = IMG_DIR / f"{img_id}.png"
    if ip.exists():
        pairs.append((ip, mp))
print(f"이미지-마스크 짝 총 {len(pairs)}장")

train_pairs, temp_pairs = train_test_split(pairs, train_size=0.70, random_state=SEED)
val_pairs, test_pairs = train_test_split(temp_pairs, train_size=0.5, random_state=SEED)
print(f"Train {len(train_pairs)} / Val {len(val_pairs)} / Internal Test {len(test_pairs)}")


class LungDataset(Dataset):
    def __init__(self, pairs, augment=False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

        img_arr = np.array(img, dtype=np.float32) / 255.0
        mask_arr = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.augment and random.random() < 0.5:
            img_arr = np.fliplr(img_arr).copy()
            mask_arr = np.fliplr(mask_arr).copy()

        img_t = torch.from_numpy(img_arr).unsqueeze(0).repeat(3, 1, 1)  # 1ch -> 3ch (SMP 사전학습 인코더 호환)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)
        return img_t, mask_t


def make_loaders(augment_train=True):
    train_ds = LungDataset(train_pairs, augment=augment_train)
    val_ds = LungDataset(val_pairs, augment=False)
    test_ds = LungDataset(test_pairs, augment=False)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    )


# ----------------------------------------------------------------------
# 2. BasicUNet (스킵 커넥션 on/off 가능한 직접 구현)
# ----------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BasicUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        chs = [32, 64, 128, 256, 512]

        self.enc1 = DoubleConv(in_ch, chs[0])
        self.enc2 = DoubleConv(chs[0], chs[1])
        self.enc3 = DoubleConv(chs[1], chs[2])
        self.enc4 = DoubleConv(chs[2], chs[3])
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(chs[3], chs[4])

        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.dec4 = DoubleConv(chs[3] * 2, chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = DoubleConv(chs[2] * 2, chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = DoubleConv(chs[1] * 2, chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = DoubleConv(chs[0] * 2, chs[0])

        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def _merge(self, up_out, skip_feat):
        if self.use_skip:
            return torch.cat([up_out, skip_feat], dim=1)
        return torch.cat([up_out, torch.zeros_like(skip_feat)], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self._merge(self.up4(b), e4))
        d3 = self.dec3(self._merge(self.up3(d4), e3))
        d2 = self.dec2(self._merge(self.up2(d3), e2))
        d1 = self.dec1(self._merge(self.up1(d2), e1))
        return self.head(d1)


# ----------------------------------------------------------------------
# 3. 손실함수(BCE + Dice), 평가지표(Dice, mIoU)
# ----------------------------------------------------------------------
def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


@torch.no_grad()
def compute_metrics(logits, target, thres=0.5, eps=1e-6):
    pred = (torch.sigmoid(logits) > thres).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union_iou = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union_iou + eps)
    dice = (2 * inter + eps) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
    return iou.mean().item(), dice.mean().item()


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_iou, total_dice, n = 0.0, 0.0, 0.0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = combined_loss(logits, masks)
            if is_train:
                loss.backward()
                optimizer.step()

            iou, dice = compute_metrics(logits.detach(), masks)
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_iou += iou * bs
            total_dice += dice * bs
            n += bs

    return total_loss / n, total_iou / n, total_dice / n


def train_model(name, model):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    train_loader, val_loader, test_loader = make_loaders()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    best_dice, best_state = -1.0, None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_iou, val_dice = run_epoch(model, val_loader, optimizer=None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)
        print(f"Epoch [{epoch:2d}/{EPOCHS}] train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} val_mIoU={val_iou:.4f} val_Dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_seconds = time.time() - t0
    model.load_state_dict(best_state)

    return {
        "name": name,
        "history": history,
        "best_val_dice": best_dice,
        "train_seconds": train_seconds,
        "n_params": sum(p.numel() for p in model.parameters()),
    }, model, test_loader


if __name__ == "__main__":
    results = {}
    models_trained = {}

    r, m, test_loader = train_model("basic_unet_skip", BasicUNet(use_skip=True))
    results["basic_unet_skip"] = r
    models_trained["basic_unet_skip"] = m

    r, m, _ = train_model("basic_unet_no_skip", BasicUNet(use_skip=False))
    results["basic_unet_no_skip"] = r
    models_trained["basic_unet_no_skip"] = m

    smp_model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    r, m, _ = train_model("smp_unet_resnet34_imagenet", smp_model)
    results["smp_unet_resnet34_imagenet"] = r
    models_trained["smp_unet_resnet34_imagenet"] = m

    best_key = max(results, key=lambda k: results[k]["best_val_dice"])
    print(f"\n최종 선택 모델: {best_key} (val Dice={results[best_key]['best_val_dice']:.4f})")
    best_model = models_trained[best_key]

    test_loss, test_iou, test_dice = run_epoch(best_model, test_loader, optimizer=None)
    print(f"Internal Test: loss={test_loss:.4f} mIoU={test_iou:.4f} Dice={test_dice:.4f}")

    # ------------------------------------------------------------------
    # 예측 마스크 시각화 (internal test에서 4장: 대표 사례)
    # ------------------------------------------------------------------
    examples_dir = RESULT_DIR / "examples"
    examples_dir.mkdir(exist_ok=True)
    best_model.eval()
    saved_examples = []
    with torch.no_grad():
        for i, (img_path, mask_path) in enumerate(test_pairs[:4]):
            img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            mask = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            img_arr = np.array(img, dtype=np.float32) / 255.0
            gt_arr = (np.array(mask, dtype=np.float32) > 127).astype(np.float32)

            img_t = torch.from_numpy(img_arr).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
            pred_logits = best_model(img_t)
            pred_arr = (torch.sigmoid(pred_logits)[0, 0].cpu().numpy() > 0.5).astype(np.float32)

            inter = (pred_arr * gt_arr).sum()
            dice_i = (2 * inter) / (pred_arr.sum() + gt_arr.sum() + 1e-6)

            # 시각화: 원본 / 정답 마스크 오버레이(초록) / 예측 마스크 오버레이(빨강)
            base_rgb = (np.stack([img_arr] * 3, axis=-1) * 255).astype(np.uint8)
            gt_overlay = base_rgb.copy()
            gt_overlay[gt_arr > 0.5] = [0, 200, 0]
            pred_overlay = base_rgb.copy()
            pred_overlay[pred_arr > 0.5] = [220, 40, 40]

            combo = np.concatenate([base_rgb, gt_overlay, pred_overlay], axis=1)
            out_path = examples_dir / f"example_{i}_{img_path.stem}.png"
            Image.fromarray(combo).save(out_path)
            saved_examples.append({"file": out_path.name, "img_id": img_path.stem, "dice": float(dice_i)})
            print("saved example:", out_path.name, "dice=", dice_i)

    output = {
        "dataset": {
            "source": "Chest Xray Masks and Labels (Montgomery+Shenzhen lung field masks)",
            "n_pairs": len(pairs), "train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs),
            "img_size": IMG_SIZE,
        },
        "device": str(DEVICE),
        "experiments": results,
        "best_key": best_key,
        "internal_test": {"loss": test_loss, "miou": test_iou, "dice": test_dice},
        "examples": saved_examples,
    }
    with open(RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n결과 저장 완료:", RESULT_DIR / "metrics.json")
