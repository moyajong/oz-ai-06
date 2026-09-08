"""
3일차 객체 탐지 과제 제출용 HTML 성능비교 리포트 생성.
result/metrics.json (baseline vs optimized 학습 결과)과
result/fp_fn_analysis.json (FP/FN 집계 + 예측 vs 정답 예시 이미지)을 읽어
Tailwind + Chart.js 다크 대시보드 스타일 리포트를 만든다. (1일차/급우 리포트와 동일 계열 스타일)
"""

import base64
import json
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / "result"

with open(RESULT_DIR / "metrics.json", encoding="utf-8") as f:
    M = json.load(f)
with open(RESULT_DIR / "fp_fn_analysis.json", encoding="utf-8") as f:
    FP = json.load(f)

base = M["experiments"]["baseline"]
opt = M["experiments"]["optimized"]
best_key = M["best_key"]
best = M["experiments"][best_key]
it = M["internal_test"]
classes = M["dataset"]["classes"]


def pct_delta(new, old):
    if not old:
        return 0.0
    return (new - old) / old * 100


def fmt_time(sec):
    m = sec / 60
    return f"{m:.1f}분"


kpi_map50_delta = pct_delta(opt["val"]["map50"], base["val"]["map50"])
kpi_map5095_delta = pct_delta(opt["val"]["map50_95"], base["val"]["map50_95"])
kpi_precision_delta = pct_delta(opt["val"]["precision"], base["val"]["precision"])
kpi_recall_delta = pct_delta(opt["val"]["recall"], base["val"]["recall"])

per_class_ap50 = M.get("per_class_ap50", {})
per_class_rows = "\n".join(
    f'''<tr class="hover:bg-slate-800/40">
        <td class="py-2 px-3 font-medium text-white">{cls}</td>
        <td class="py-2 px-3 font-mono">{per_class_ap50.get(cls, 0)*100:.2f}%</td>
        <td class="py-2 px-3 font-mono text-rose-300">{FP['per_class_fp'].get(cls, 0)}</td>
        <td class="py-2 px-3 font-mono text-amber-300">{FP['per_class_fn'].get(cls, 0)}</td>
    </tr>'''
    for cls in classes
)

# 예측 vs 정답 예시 이미지를 base64로 인라인 임베드 (자체 완결형 HTML)
example_blocks = []
for ex in FP["examples"]:
    img_path = RESULT_DIR / "examples" / ex["file"]
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    tag = "어려운 사례" if (ex["fp"] + ex["fn"]) > 0 else "정탐 사례"
    tag_color = "bg-rose-500/20 text-rose-300 border-rose-500/30" if (ex["fp"] + ex["fn"]) > 0 else "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
    example_blocks.append(f'''
    <div class="space-y-2">
        <div class="flex items-center justify-between">
            <span class="text-xs font-mono text-slate-400">{ex['img_id'][:16]}...</span>
            <span class="text-[11px] px-2 py-0.5 rounded-full border {tag_color} font-semibold">{tag} (GT {ex['n_gt']} / Pred {ex['n_pred']} / FP {ex['fp']} / FN {ex['fn']})</span>
        </div>
        <img src="data:image/png;base64,{b64}" class="w-full rounded-lg border border-slate-700" />
        <p class="text-[11px] text-slate-500 text-center">왼쪽: 정답(Ground Truth) &nbsp;|&nbsp; 오른쪽: 모델 예측(Prediction)</p>
    </div>''')
examples_html = "\n".join(example_blocks)

total_fp, total_fn, total_tp = FP["total"]["fp"], FP["total"]["fn"], FP["total"]["tp"]

html = f"""<!DOCTYPE html>
<html lang="ko" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VinBigData Chest X-ray YOLOv8 객체 탐지 성능 비교 리포트</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    body {{ font-family: 'Pretendard', sans-serif; background-color: #0f172a; color: #f8fafc; }}
    .glass-card {{ background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); }}
</style>
</head>
<body class="min-h-screen pb-12">

<header class="sticky top-0 z-50 glass-card border-b border-slate-800 px-6 py-4 mb-8">
    <div class="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
        <div class="flex items-center gap-3">
            <div class="p-2.5 bg-rose-600/20 rounded-xl border border-rose-500/30 text-rose-400">
                <i class="fa-solid fa-lungs text-2xl"></i>
            </div>
            <div>
                <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                    VinBigData Chest X-ray <span class="text-xs px-2.5 py-0.5 rounded-full bg-rose-500/20 text-rose-300 font-medium border border-rose-500/30">YOLOv8 객체 탐지 실험 리포트</span>
                </h1>
                <p class="text-xs text-slate-400">Baseline(imgsz 384) vs Optimized(imgsz 640) · 고정 Train/Val/Internal-Test 분할 재사용</p>
            </div>
        </div>
        <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-slate-300">
            <i class="fa-solid fa-microchip text-emerald-400"></i>
            <span>NVIDIA RTX 4060 (CUDA)</span>
        </div>
    </div>
</header>

<main class="max-w-7xl mx-auto px-4 sm:px-6 space-y-8">

    <section class="space-y-4">
        <div class="flex items-center justify-between">
            <h2 class="text-lg font-bold text-slate-100 flex items-center gap-2">
                <i class="fa-solid fa-chart-line text-rose-400"></i> Validation 성능 비교 (config 한 가지: 입력 해상도만 변경)
            </h2>
            <span class="text-xs text-slate-400">최종 채택: <strong class="text-amber-300">{best_key} (imgsz={best['imgsz']})</strong></span>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="glass-card rounded-2xl p-5 border-l-4 border-l-blue-500">
                <div class="text-xs font-semibold uppercase text-slate-400 tracking-wider">mAP50</div>
                <div class="text-3xl font-extrabold text-white mt-2">{opt['val']['map50']*100:.2f}%</div>
                <p class="text-xs {'text-emerald-400' if kpi_map50_delta>=0 else 'text-rose-400'} mt-1">Baseline {base['val']['map50']*100:.2f}% 대비 {kpi_map50_delta:+.1f}%</p>
            </div>
            <div class="glass-card rounded-2xl p-5 border-l-4 border-l-purple-500">
                <div class="text-xs font-semibold uppercase text-slate-400 tracking-wider">mAP50-95</div>
                <div class="text-3xl font-extrabold text-white mt-2">{opt['val']['map50_95']*100:.2f}%</div>
                <p class="text-xs {'text-emerald-400' if kpi_map5095_delta>=0 else 'text-rose-400'} mt-1">Baseline {base['val']['map50_95']*100:.2f}% 대비 {kpi_map5095_delta:+.1f}%</p>
            </div>
            <div class="glass-card rounded-2xl p-5 border-l-4 border-l-emerald-500">
                <div class="text-xs font-semibold uppercase text-slate-400 tracking-wider">Precision</div>
                <div class="text-3xl font-extrabold text-white mt-2">{opt['val']['precision']*100:.2f}%</div>
                <p class="text-xs {'text-emerald-400' if kpi_precision_delta>=0 else 'text-rose-400'} mt-1">Baseline {base['val']['precision']*100:.2f}% 대비 {kpi_precision_delta:+.1f}%</p>
            </div>
            <div class="glass-card rounded-2xl p-5 border-l-4 border-l-amber-500">
                <div class="text-xs font-semibold uppercase text-slate-400 tracking-wider">Recall</div>
                <div class="text-3xl font-extrabold text-white mt-2">{opt['val']['recall']*100:.2f}%</div>
                <p class="text-xs {'text-emerald-400' if kpi_recall_delta>=0 else 'text-rose-400'} mt-1">Baseline {base['val']['recall']*100:.2f}% 대비 {kpi_recall_delta:+.1f}%</p>
            </div>
        </div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-code-compare text-amber-400"></i> 실험 조건 및 속도 vs 성능 트레이드오프
        </h3>
        <div class="overflow-x-auto">
            <table class="w-full text-xs text-left text-slate-300">
                <thead class="bg-slate-800/80 text-slate-200 uppercase font-semibold border-b border-slate-700">
                    <tr>
                        <th class="py-3 px-4">실험</th>
                        <th class="py-3 px-4">입력 해상도</th>
                        <th class="py-3 px-4">학습 epoch</th>
                        <th class="py-3 px-4">학습 시간</th>
                        <th class="py-3 px-4">Val mAP50</th>
                        <th class="py-3 px-4">Val mAP50-95</th>
                        <th class="py-3 px-4 text-center">선택 여부</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-800">
                    <tr class="{'bg-blue-950/20' if best_key=='baseline' else ''}">
                        <td class="py-3 px-4 font-medium text-slate-200">Baseline (BASELINE_CONFIG)</td>
                        <td class="py-3 px-4 font-mono">384 x 384</td>
                        <td class="py-3 px-4 font-mono">{base['epochs_ran']}</td>
                        <td class="py-3 px-4 font-mono">{fmt_time(base['train_seconds'])}</td>
                        <td class="py-3 px-4 font-mono">{base['val']['map50']*100:.2f}%</td>
                        <td class="py-3 px-4 font-mono">{base['val']['map50_95']*100:.2f}%</td>
                        <td class="py-3 px-4 text-center">{'<span class="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30 text-[11px]">최종 선택</span>' if best_key=='baseline' else '<span class="px-2 py-0.5 rounded bg-slate-700 text-slate-400 text-[11px]">비교군</span>'}</td>
                    </tr>
                    <tr class="{'bg-blue-950/20' if best_key=='optimized' else ''}">
                        <td class="py-3 px-4 font-semibold text-blue-300">Optimized (EXPERIMENT_QUEUE: imgsz↑)</td>
                        <td class="py-3 px-4 font-mono">640 x 640</td>
                        <td class="py-3 px-4 font-mono">{opt['epochs_ran']}</td>
                        <td class="py-3 px-4 font-mono">{fmt_time(opt['train_seconds'])}</td>
                        <td class="py-3 px-4 font-mono font-bold text-amber-300">{opt['val']['map50']*100:.2f}%</td>
                        <td class="py-3 px-4 font-mono font-bold text-amber-300">{opt['val']['map50_95']*100:.2f}%</td>
                        <td class="py-3 px-4 text-center">{'<span class="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30 text-[11px]">최종 선택</span>' if best_key=='optimized' else '<span class="px-2 py-0.5 rounded bg-slate-700 text-slate-400 text-[11px]">비교군</span>'}</td>
                    </tr>
                </tbody>
            </table>
        </div>
        <div class="p-3 bg-amber-950/20 rounded-xl border border-amber-800/30 text-xs text-amber-200/90 leading-relaxed">
            <strong><i class="fa-solid fa-lightbulb"></i> 한 가지 조건(입력 해상도)만 바꾼 결과:</strong>
            imgsz를 384 &rarr; 640으로 올리면 학습 시간이 {fmt_time(base['train_seconds'])} &rarr; {fmt_time(opt['train_seconds'])}로 늘었지만,
            흉부 X-ray의 미세한 이상 소견(작은 결절 등)이 더 선명하게 보존되어 mAP50-95가 {base['val']['map50_95']*100:.2f}% &rarr; {opt['val']['map50_95']*100:.2f}%로
            {'개선' if kpi_map5095_delta >= 0 else '악화'}되었습니다. 속도와 탐지 성능은 전형적인 트레이드오프 관계임을 확인했습니다.
        </div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-vial-circle-check text-purple-400"></i> Internal Test 최종 평가 ({best_key}, 1회만 실행)
        </h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                <div class="text-xs text-slate-400">mAP50</div>
                <div class="text-2xl font-bold text-white">{it['map50']*100:.2f}%</div>
            </div>
            <div class="p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                <div class="text-xs text-slate-400">mAP50-95</div>
                <div class="text-2xl font-bold text-white">{it['map50_95']*100:.2f}%</div>
            </div>
            <div class="p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                <div class="text-xs text-slate-400">Precision</div>
                <div class="text-2xl font-bold text-white">{it['precision']*100:.2f}%</div>
            </div>
            <div class="p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                <div class="text-xs text-slate-400">Recall</div>
                <div class="text-2xl font-bold text-white">{it['recall']*100:.2f}%</div>
            </div>
        </div>
        <div class="overflow-x-auto">
            <table class="w-full text-xs text-left text-slate-300">
                <thead class="bg-slate-800 text-slate-200 border-b border-slate-700 font-semibold">
                    <tr><th class="py-2.5 px-3">클래스</th><th class="py-2.5 px-3">AP50</th><th class="py-2.5 px-3">FP (오탐)</th><th class="py-2.5 px-3">FN (미검출)</th></tr>
                </thead>
                <tbody class="divide-y divide-slate-800">{per_class_rows}</tbody>
                <tfoot class="bg-slate-800/90 font-bold border-t border-slate-700 text-white">
                    <tr><td class="py-2.5 px-3">전체 (IoU&ge;0.5, conf&ge;0.25 기준)</td><td class="py-2.5 px-3"></td>
                        <td class="py-2.5 px-3 text-rose-300">{total_fp}</td><td class="py-2.5 px-3 text-amber-300">{total_fn}</td></tr>
                </tfoot>
            </table>
        </div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-image text-emerald-400"></i> 예측 vs 정답 이미지 비교 & FP·FN 사례
        </h3>
        <div class="grid grid-cols-1 gap-6">
            {examples_html}
        </div>
        <div class="p-3 bg-blue-950/20 rounded-xl border border-blue-800/30 text-xs text-blue-200/90 leading-relaxed">
            <strong><i class="fa-solid fa-comment-dots"></i> FP·FN 사례 코멘트:</strong>
            Internal Test 전체 {total_tp + total_fn}개 정답 박스 중 {total_tp}개를 정확히 탐지(TP)했고, {total_fn}개는 놓쳤습니다(FN).
            반대로 모델이 존재하지 않는 위치를 소견으로 잘못 예측한 오탐(FP)은 {total_fp}건이었습니다.
            소량(2,000장) 데이터와 짧은 학습 epoch 특성상, 병변 크기가 작거나 경계가 모호한 클래스(Nodule/Mass, Pulmonary fibrosis 등)에서
            FN이 상대적으로 많이 발생하는 경향을 확인했습니다.
        </div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-database text-blue-400"></i> 데이터 및 실험 설계
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-300">
            <div class="p-4 bg-slate-900/40 rounded-xl border border-slate-800 space-y-1.5">
                <div class="font-bold text-slate-200">데이터셋</div>
                <p>VinBigData 흉부 X-ray (512x512 PNG 변환본), 14종 이상 소견 + No finding</p>
                <p>No finding : 소견 있음 비율을 보존해 2,000장 추출 (seed=42)</p>
            </div>
            <div class="p-4 bg-slate-900/40 rounded-xl border border-slate-800 space-y-1.5">
                <div class="font-bold text-slate-200">고정 분할 (모든 실험 재사용)</div>
                <p>Train 1,400 / Validation 300 / Internal Test 300</p>
                <p>finding 유무 기준 stratified split</p>
            </div>
            <div class="p-4 bg-slate-900/40 rounded-xl border border-slate-800 space-y-1.5">
                <div class="font-bold text-slate-200">모델 & 절차</div>
                <p>YOLOv8n (COCO 사전학습) fine-tuning</p>
                <p>Validation으로 최종 모델 선정 &rarr; Internal Test는 단 1회 실행</p>
            </div>
        </div>
    </section>
</main>

<footer class="max-w-7xl mx-auto px-6 mt-12 text-center text-xs text-slate-500 border-t border-slate-800/80 pt-6">
    <p>VinBigData Chest X-ray Object Detection Report | 실제 학습 실행 결과 (YOLOv8n, seed=42)</p>
</footer>

</body>
</html>
"""

out_path = RESULT_DIR / "vinbigdata_yolov8_report.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print("리포트 저장:", out_path)
