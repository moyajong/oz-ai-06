"""
4일차 이미지 분할 과제 제출용 HTML 리포트 생성.
result/metrics.json (BasicUNet skip/no-skip + SMP U-Net 비교 결과)을 읽어
Tailwind + Chart.js 다크 대시보드 스타일 리포트를 만든다.
"""

import base64
import json
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / "result"

with open(RESULT_DIR / "metrics.json", encoding="utf-8") as f:
    M = json.load(f)

exp = M["experiments"]
best_key = M["best_key"]
it = M["internal_test"]
ds = M["dataset"]

NAME_KR = {
    "basic_unet_skip": "BasicUNet (Skip O)",
    "basic_unet_no_skip": "BasicUNet (Skip X)",
    "smp_unet_resnet34_imagenet": "SMP U-Net (ResNet34, ImageNet)",
}


def fmt_time(sec):
    return f"{sec/60:.1f}분"


rows = "\n".join(f'''<tr class="{'bg-blue-950/20' if k==best_key else ''} hover:bg-slate-800/40">
    <td class="py-3 px-4 font-medium text-slate-200">{NAME_KR[k]}</td>
    <td class="py-3 px-4 font-mono">{exp[k]['n_params']:,}</td>
    <td class="py-3 px-4 font-mono">{fmt_time(exp[k]['train_seconds'])}</td>
    <td class="py-3 px-4 font-mono font-bold text-amber-300">{exp[k]['best_val_dice']*100:.2f}%</td>
    <td class="py-3 px-4 font-mono">{max(exp[k]['history']['val_iou'])*100:.2f}%</td>
    <td class="py-3 px-4 text-center">{'<span class="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30 text-[11px]">최종 선택</span>' if k==best_key else '<span class="px-2 py-0.5 rounded bg-slate-700 text-slate-400 text-[11px]">비교군</span>'}</td>
</tr>''' for k in ["basic_unet_skip", "basic_unet_no_skip", "smp_unet_resnet34_imagenet"])

skip_dice = exp["basic_unet_skip"]["best_val_dice"]
noskip_dice = exp["basic_unet_no_skip"]["best_val_dice"]
skip_effect = (skip_dice - noskip_dice) / noskip_dice * 100 if noskip_dice else 0

example_blocks = []
for ex in M["examples"]:
    img_path = RESULT_DIR / "examples" / ex["file"]
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    example_blocks.append(f'''
    <div class="space-y-2">
        <div class="flex items-center justify-between">
            <span class="text-xs font-mono text-slate-400">{ex['img_id']}</span>
            <span class="text-[11px] px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30 font-semibold">Dice {ex['dice']*100:.1f}%</span>
        </div>
        <img src="data:image/png;base64,{b64}" class="w-full rounded-lg border border-slate-700" />
        <p class="text-[11px] text-slate-500 text-center">왼쪽: 원본 &nbsp;|&nbsp; 가운데: 정답 마스크(초록) &nbsp;|&nbsp; 오른쪽: 예측 마스크(빨강)</p>
    </div>''')
examples_html = "\n".join(example_blocks)

epochs = list(range(1, len(exp["basic_unet_skip"]["history"]["val_dice"]) + 1))

html = f"""<!DOCTYPE html>
<html lang="ko" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>흉부 X-ray 폐 영역 세그멘테이션 실험 리포트</title>
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
            <div class="p-2.5 bg-emerald-600/20 rounded-xl border border-emerald-500/30 text-emerald-400">
                <i class="fa-solid fa-lungs text-2xl"></i>
            </div>
            <div>
                <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                    Chest X-ray Lung Field <span class="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-medium border border-emerald-500/30">Semantic Segmentation 실험 리포트</span>
                </h1>
                <p class="text-xs text-slate-400">BasicUNet(Skip O/X) 직접 구현 vs SMP ResNet34 사전학습 U-Net · 20 epoch 동일 조건 비교</p>
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
                <i class="fa-solid fa-chart-line text-emerald-400"></i> 모델별 최종 성능 요약
            </h2>
            <span class="text-xs text-slate-400">최종 채택: <strong class="text-amber-300">{NAME_KR[best_key]}</strong></span>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="glass-card rounded-2xl p-5 border-l-4 border-l-emerald-500">
                <div class="text-xs font-semibold uppercase text-slate-400 tracking-wider">Internal Test Dice</div>
                <div class="text-3xl font-extrabold text-white mt-2">{it['dice']*100:.2f}%</div>
                <p class="text-xs text-slate-400 mt-1">{NAME_KR[best_key]}</p>
            </div>
            <div class="glass-card rounded-2xl p-5 border-l-4 border-l-blue-500">
                <div class="text-xs font-semibold uppercase text-slate-400 tracking-wider">Internal Test mIoU</div>
                <div class="text-3xl font-extrabold text-white mt-2">{it['miou']*100:.2f}%</div>
                <p class="text-xs text-slate-400 mt-1">IoU (Jaccard)</p>
            </div>
            <div class="glass-card rounded-2xl p-5 border-l-4 border-l-purple-500">
                <div class="text-xs font-semibold uppercase text-slate-400 tracking-wider">스킵 커넥션 효과</div>
                <div class="text-3xl font-extrabold text-white mt-2">{skip_effect:+.1f}%</div>
                <p class="text-xs text-slate-400 mt-1">Dice: Skip O {skip_dice*100:.1f}% vs Skip X {noskip_dice*100:.1f}%</p>
            </div>
            <div class="glass-card rounded-2xl p-5 border-l-4 border-l-amber-500">
                <div class="text-xs font-semibold uppercase text-slate-400 tracking-wider">데이터 분할</div>
                <div class="text-3xl font-extrabold text-white mt-2">{ds['train']}/{ds['val']}/{ds['test']}</div>
                <p class="text-xs text-slate-400 mt-1">Train/Val/Internal-Test (seed=42)</p>
            </div>
        </div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-code-compare text-amber-400"></i> 세 모델 비교 (같은 데이터 · 분할 · seed · 20 epoch · optimizer · loss)
        </h3>
        <div class="overflow-x-auto">
            <table class="w-full text-xs text-left text-slate-300">
                <thead class="bg-slate-800/80 text-slate-200 uppercase font-semibold border-b border-slate-700">
                    <tr>
                        <th class="py-3 px-4">모델</th>
                        <th class="py-3 px-4">파라미터 수</th>
                        <th class="py-3 px-4">학습 시간</th>
                        <th class="py-3 px-4">Best Val Dice</th>
                        <th class="py-3 px-4">Best Val mIoU</th>
                        <th class="py-3 px-4 text-center">선택 여부</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-800">{rows}</tbody>
            </table>
        </div>
        <div class="p-3 bg-amber-950/20 rounded-xl border border-amber-800/30 text-xs text-amber-200/90 leading-relaxed">
            <strong><i class="fa-solid fa-lightbulb"></i> 스킵 커넥션 유무 비교:</strong>
            같은 BasicUNet 구조에서 스킵 커넥션만 제거(bottleneck 정보만으로 복원)했더니 Val Dice가 {skip_dice*100:.2f}% &rarr; {noskip_dice*100:.2f}%로
            {'하락' if skip_effect > 0 else '변화'}했습니다. 인코더의 세부 경계 정보가 디코더로 직접 전달되지 않으면 폐 경계처럼 얇고 정밀한 영역 복원이 어려워짐을 확인했습니다.
            한편 ImageNet 사전학습 인코더(ResNet34)를 쓴 SMP U-Net은 처음부터 학습한 두 모델보다 빠르게 수렴하며 가장 높은 Dice를 기록했습니다.
        </div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-chart-area text-blue-400"></i> 학습 곡선 (Validation Dice 추이)
        </h3>
        <div class="h-72"><canvas id="diceChart"></canvas></div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-image text-emerald-400"></i> 예측 결과 (Internal Test 샘플)
        </h3>
        <div class="grid grid-cols-1 gap-6">
            {examples_html}
        </div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-database text-blue-400"></i> 데이터 및 실험 설계
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-300">
            <div class="p-4 bg-slate-900/40 rounded-xl border border-slate-800 space-y-1.5">
                <div class="font-bold text-slate-200">데이터셋</div>
                <p>{ds['source']}</p>
                <p>이미지-마스크 짝 {ds['n_pairs']}장 (이진 마스크: 폐 영역 vs 배경), {ds['img_size']}x{ds['img_size']} 리사이즈</p>
            </div>
            <div class="p-4 bg-slate-900/40 rounded-xl border border-slate-800 space-y-1.5">
                <div class="font-bold text-slate-200">고정 조건</div>
                <p>Train {ds['train']} / Val {ds['val']} / Internal Test {ds['test']} (seed=42)</p>
                <p>Optimizer: Adam(lr=1e-3), Loss: BCE + Dice, 20 epoch</p>
            </div>
            <div class="p-4 bg-slate-900/40 rounded-xl border border-slate-800 space-y-1.5">
                <div class="font-bold text-slate-200">평가 및 절차</div>
                <p>Val Dice 최고 시점 가중치로 각 모델 선정</p>
                <p>세 모델 중 최종 1개 선택 &rarr; Internal Test는 단 1회 실행</p>
            </div>
        </div>
    </section>
</main>

<footer class="max-w-7xl mx-auto px-6 mt-12 text-center text-xs text-slate-500 border-t border-slate-800/80 pt-6">
    <p>Chest X-ray Lung Segmentation Report | 실제 학습 실행 결과 (PyTorch + segmentation_models_pytorch, seed=42)</p>
</footer>

<script>
document.addEventListener('DOMContentLoaded', function() {{
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = 'Pretendard, sans-serif';
    const ctx = document.getElementById('diceChart').getContext('2d');
    new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: {epochs},
            datasets: [
                {{ label: 'BasicUNet (Skip O)', data: {[round(v,4) for v in exp['basic_unet_skip']['history']['val_dice']]}, borderColor: '#3b82f6', borderWidth: 2, pointRadius: 2, tension: 0.3 }},
                {{ label: 'BasicUNet (Skip X)', data: {[round(v,4) for v in exp['basic_unet_no_skip']['history']['val_dice']]}, borderColor: '#f43f5e', borderWidth: 2, pointRadius: 2, tension: 0.3 }},
                {{ label: 'SMP U-Net (ResNet34)', data: {[round(v,4) for v in exp['smp_unet_resnet34_imagenet']['history']['val_dice']]}, borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.1)', fill: true, borderWidth: 2.5, pointRadius: 2, tension: 0.3 }}
            ]
        }},
        options: {{ responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 12, font: {{ size: 10 }} }} }} }},
            scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Validation Dice' }} }} }}
        }}
    }});
}});
</script>
</body>
</html>
"""

out_path = RESULT_DIR / "lung_segmentation_report.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print("리포트 저장:", out_path)
