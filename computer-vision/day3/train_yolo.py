"""
3일차 자유과제: 객체 탐지 - Baseline vs Optimized YOLOv8n 성능 비교

BASELINE_CONFIG(imgsz=384)와 EXPERIMENT_QUEUE의 최적화 설정(imgsz=640) 두 가지만
다르게 하여(다른 조건은 모두 고정) 같은 train/val/internal-test 분할에 대해 학습하고,
Validation 성능(mAP50/mAP50-95)으로 최종 모델을 고른 뒤 Internal Test는 한 번만 평가한다.
"""

import json
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "yolo_dataset" / "data.yaml"
RESULT_DIR = ROOT / "result"
RESULT_DIR.mkdir(exist_ok=True)
RUNS_DIR = ROOT / "runs"

SEED = 42
COMMON = dict(data=str(DATA_YAML), epochs=30, patience=8, batch=32, seed=SEED,
              workers=4, device=0, plots=False, verbose=True, project=str(RUNS_DIR))

CONFIGS = {
    "baseline": {**COMMON, "imgsz": 384, "name": "baseline"},
    "optimized": {**COMMON, "imgsz": 640, "name": "optimized"},
}


def run_experiment(key, cfg):
    print(f"\n{'=' * 60}\nExp: {key}  (imgsz={cfg['imgsz']})\n{'=' * 60}")
    model = YOLO("yolov8n.pt")
    t0 = time.time()
    train_results = model.train(**cfg)
    train_seconds = time.time() - t0

    val_metrics = train_results.results_dict
    val_summary = {
        "precision": val_metrics.get("metrics/precision(B)"),
        "recall": val_metrics.get("metrics/recall(B)"),
        "map50": val_metrics.get("metrics/mAP50(B)"),
        "map50_95": val_metrics.get("metrics/mAP50-95(B)"),
    }

    best_weights = Path(train_results.save_dir) / "weights" / "best.pt"
    results_csv = Path(train_results.save_dir) / "results.csv"
    epochs_ran = cfg["epochs"]
    if results_csv.exists():
        with open(results_csv, encoding="utf-8") as f:
            epochs_ran = max(sum(1 for _ in f) - 1, 1)

    return {
        "imgsz": cfg["imgsz"],
        "epochs_ran": epochs_ran,
        "train_seconds": train_seconds,
        "val": val_summary,
        "best_weights": str(best_weights),
        "save_dir": str(train_results.save_dir),
    }, best_weights


if __name__ == "__main__":
    results = {}
    weights = {}
    for key, cfg in CONFIGS.items():
        results[key], weights[key] = run_experiment(key, cfg)
        print(f"{key} val: {results[key]['val']}  (train_seconds={results[key]['train_seconds']:.1f})")

    # Validation mAP50-95 기준으로 최종 모델 선택
    best_key = max(results, key=lambda k: (results[k]["val"]["map50_95"] or -1))
    print(f"\n최종 선택 모델: {best_key}")

    best_model = YOLO(str(weights[best_key]))
    test_metrics = best_model.val(data=str(DATA_YAML), split="test", imgsz=results[best_key]["imgsz"],
                                    workers=0, device=0, plots=False, project=str(RUNS_DIR), name="internal_test")
    test_dict = test_metrics.results_dict
    internal_test_summary = {
        "precision": test_dict.get("metrics/precision(B)"),
        "recall": test_dict.get("metrics/recall(B)"),
        "map50": test_dict.get("metrics/mAP50(B)"),
        "map50_95": test_dict.get("metrics/mAP50-95(B)"),
    }
    print("Internal test:", internal_test_summary)

    # 클래스별 AP50 (선택된 모델, internal test 기준)
    class_names = best_model.names
    per_class_ap50 = {}
    try:
        ap50_per_class = test_metrics.box.ap50  # array indexed by class present in val set
        ap_class_indices = test_metrics.box.ap_class_index
        for idx, cls_idx in enumerate(ap_class_indices):
            per_class_ap50[class_names[int(cls_idx)]] = float(ap50_per_class[idx])
    except Exception as e:
        print("per-class AP 추출 실패:", e)

    output = {
        "dataset": {
            "source": "VinBigData Chest X-ray (512x512 PNG, awsaf49 변환본)",
            "n_sample": 2000, "train": 1400, "val": 300, "test": 300,
            "classes": list(class_names.values()) if hasattr(class_names, "values") else class_names,
        },
        "model": "YOLOv8n (yolov8n.pt, COCO pretrained)",
        "best_key": best_key,
        "experiments": results,
        "internal_test": internal_test_summary,
        "per_class_ap50": per_class_ap50,
    }

    with open(RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n결과 저장 완료:", RESULT_DIR / "metrics.json")
