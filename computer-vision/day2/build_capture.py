"""
2일차 자유과제 제출용 "캡처" 이미지 생성.
result/metrics.json 의 실측 결과를 그대로 사용해 1일차(from-scratch CNN) 대비
2일차(ResNet18 전이학습) 성능 향상을 한 장의 이미지로 요약한다.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

RESULT_DIR = Path(__file__).resolve().parent / "result"

with open(RESULT_DIR / "metrics.json", encoding="utf-8") as f:
    M = json.load(f)

fe = M["experiments"]["feature_extraction"]
ft = M["experiments"]["fine_tuning"]
d1 = M["day1_reference"]

# 한글 폰트 (Windows 기본 맑은 고딕)
for candidate in ["Malgun Gothic", "NanumGothic", "AppleGothic"]:
    if any(candidate == f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = candidate
        break
plt.rcParams["axes.unicode_minus"] = False

BG = "#0f172a"
PANEL = "#1e293b"
GRID = "#334155"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
BLUE = "#3b82f6"
PURPLE = "#a855f7"
EMERALD = "#34d399"
AMBER = "#fbbf24"
ROSE = "#f43f5e"

fig = plt.figure(figsize=(13, 8), facecolor=BG)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.38, wspace=0.28,
                       left=0.07, right=0.96, top=0.86, bottom=0.08)

fig.suptitle("2일차 자유과제 — 이미지 분류(전이학습) 성능 실험 결과",
             color=TEXT, fontsize=18, fontweight="bold", x=0.07, ha="left", y=0.965)
fig.text(0.07, 0.925,
          "Intel Image Classification · 6클래스 · ResNet18(ImageNet 사전학습) · 1일차와 동일 데이터 분할(4800/1200/1200)",
          color=MUTED, fontsize=10.5, ha="left")

# ---- (1) 정확도 비교 막대그래프 ----
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL)
labels = ["1일차\nBaseline\nCNN", "1일차\nAugmented\nCNN", "2일차\nFeature\nExtraction", "2일차\nFine-\ntuning"]
values = [d1["baseline_test_accuracy"], d1["augmented_test_accuracy"], fe["test_accuracy"], ft["test_accuracy"]]
colors = [MUTED, "#64748b", BLUE, EMERALD]
bars = ax1.bar(labels, [v * 100 for v in values], color=colors, width=0.6, zorder=3)
bars[-1].set_edgecolor(AMBER)
bars[-1].set_linewidth(2.5)
for b, v in zip(bars, values):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{v*100:.2f}%",
              ha="center", color=TEXT, fontsize=10.5, fontweight="bold")
ax1.set_ylim(0, 100)
ax1.set_ylabel("Test Accuracy (%)", color=MUTED, fontsize=10)
ax1.set_title("모델별 최종 Test Accuracy 비교", color=TEXT, fontsize=12, fontweight="bold", pad=10)
ax1.tick_params(colors=MUTED, labelsize=9)
ax1.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
for spine in ax1.spines.values():
    spine.set_visible(False)

# ---- (2) 핵심 지표 요약 텍스트 패널 ----
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(PANEL)
ax2.axis("off")
summary_lines = [
    ("최고 성능 모델", "ResNet18 Fine-tuning", AMBER, 13),
    ("Test Accuracy", f"{ft['test_accuracy']*100:.2f}%", EMERALD, 15),
    ("Test Macro F1", f"{ft['test_macro_f1']:.4f}", TEXT, 12),
    ("Test Macro AUROC", f"{ft['test_macro_auroc']:.4f}", TEXT, 12),
    ("1일차 대비 향상", f"+{(ft['test_accuracy']-d1['augmented_test_accuracy'])*100:.2f}%p", ROSE, 13),
]
y0 = 0.90
for label, val, color, size in summary_lines:
    ax2.text(0.06, y0, label, color=MUTED, fontsize=10, transform=ax2.transAxes)
    ax2.text(0.94, y0, val, color=color, fontsize=size, fontweight="bold",
              ha="right", transform=ax2.transAxes)
    y0 -= 0.19
ax2.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.98, transform=ax2.transAxes,
                              fill=False, edgecolor=GRID, linewidth=1))
ax2.set_title("최종 채택 모델 핵심 지표", color=TEXT, fontsize=12, fontweight="bold", pad=10, loc="left", x=0.06)

# ---- (3) Feature Extraction 학습 곡선 ----
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(PANEL)
ep_fe = list(range(1, len(fe["history"]["val_acc"]) + 1))
ep_ft = list(range(1, len(ft["history"]["val_acc"]) + 1))
ax3.plot(ep_fe, [v * 100 for v in fe["history"]["val_acc"]], color=BLUE, marker="o", markersize=4,
          linewidth=2, label="Feature Extraction (val_acc)")
ax3.plot(ep_ft, [v * 100 for v in ft["history"]["val_acc"]], color=EMERALD, marker="o", markersize=4,
          linewidth=2.4, label="Fine-tuning (val_acc)")
ax3.axvline(ft["best_epoch"], color=AMBER, linestyle="--", linewidth=1.2, alpha=0.8)
ax3.text(ft["best_epoch"] + 0.1, ax3.get_ylim()[0] + 2, f"best epoch={ft['best_epoch']}",
          color=AMBER, fontsize=8.5)
ax3.set_xlabel("Epoch", color=MUTED, fontsize=9.5)
ax3.set_ylabel("Validation Accuracy (%)", color=MUTED, fontsize=9.5)
ax3.set_title("Validation Accuracy 추이 (Feature Extraction → Fine-tuning)", color=TEXT, fontsize=11.5,
               fontweight="bold", pad=8)
ax3.tick_params(colors=MUTED, labelsize=8.5)
ax3.grid(color=GRID, linewidth=0.5, alpha=0.7)
ax3.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8.5, loc="lower right")
for spine in ax3.spines.values():
    spine.set_color(GRID)

# ---- (4) 클래스별 F1 (최종 모델) ----
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(PANEL)
classes_kr = {"buildings": "buildings", "forest": "forest", "glacier": "glacier",
              "mountain": "mountain", "sea": "sea", "street": "street"}
cls_names = [classes_kr[c] for c in M["classes"]]
f1s = ft["per_class"]["f1"]
bar_colors = [EMERALD if f >= 0.9 else (BLUE if f >= 0.8 else ROSE) for f in f1s]
bars4 = ax4.barh(cls_names, [f * 100 for f in f1s], color=bar_colors, zorder=3)
for b, f in zip(bars4, f1s):
    ax4.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{f*100:.1f}%",
              va="center", color=TEXT, fontsize=9.5)
ax4.set_xlim(0, 105)
ax4.set_xlabel("F1-score (%)", color=MUTED, fontsize=9.5)
ax4.set_title("클래스별 F1-score (Fine-tuning 모델, Test set)", color=TEXT, fontsize=11.5,
               fontweight="bold", pad=8)
ax4.tick_params(colors=MUTED, labelsize=9.5)
ax4.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.7, zorder=0)
for spine in ax4.spines.values():
    spine.set_visible(False)

fig.text(0.07, 0.02,
          "Device: CPU · ResNet18 Feature Extraction(8 epoch, backbone 동결) -> Fine-tuning(warm start, 전체 미세조정, best epoch 6) · seed=42",
          color=MUTED, fontsize=8.3, ha="left")

out_path = RESULT_DIR / "day2_result_capture.png"
fig.savefig(out_path, dpi=150, facecolor=BG)
print("캡처 저장:", out_path)
