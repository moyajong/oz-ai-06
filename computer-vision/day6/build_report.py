"""
6일차 파이널 자유주제 프로젝트 제출용 HTML 리포트 생성.
result/metrics.json (베이스라인 → meningioma aug/원본 진단 → 공정 평가 처방)을 읽어
Tailwind + Chart.js 다크 대시보드 스타일 리포트를 만든다.
"""

import json
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / "result"

with open(RESULT_DIR / "metrics.json", encoding="utf-8") as f:
    M = json.load(f)
with open(RESULT_DIR / "eda_leakage.json", encoding="utf-8") as f:
    EDA = json.load(f)

classes = M["dataset"]["classes"]
base = M["baseline_official_test"]
diag = M["diagnosis_meningioma_subgroups"]
fix = M["prescription_clean_eval"]

per_class_rows = "\n".join(
    f'''<tr class="hover:bg-slate-800/40 {'bg-amber-950/10' if c=='meningioma' else ''}">
        <td class="py-2 px-3 font-medium text-white">{c}{' <span class="text-[10px] text-amber-400">(증강 섞임)</span>' if c=='meningioma' else ''}</td>
        <td class="py-2 px-3 font-mono">{base['per_class']['precision'][i]:.3f}</td>
        <td class="py-2 px-3 font-mono">{base['per_class']['recall'][i]:.3f}</td>
        <td class="py-2 px-3 font-mono text-blue-300 font-bold">{base['per_class']['f1'][i]:.3f}</td>
    </tr>'''
    for i, c in enumerate(classes)
)

per_class_rows_fixed = "\n".join(
    f'''<tr class="hover:bg-slate-800/40 {'bg-emerald-950/10' if c=='meningioma' else ''}">
        <td class="py-2 px-3 font-medium text-white">{c}{' <span class="text-[10px] text-emerald-400">(원본만)</span>' if c=='meningioma' else ''}</td>
        <td class="py-2 px-3 font-mono">{fix['per_class']['precision'][i]:.3f}</td>
        <td class="py-2 px-3 font-mono">{fix['per_class']['recall'][i]:.3f}</td>
        <td class="py-2 px-3 font-mono text-emerald-300 font-bold">{fix['per_class']['f1'][i]:.3f}</td>
    </tr>'''
    for i, c in enumerate(classes)
)

class_stats_rows = "\n".join(
    f'''<tr class="hover:bg-slate-800/40">
        <td class="py-2 px-3 font-medium text-white">{c}</td>
        <td class="py-2 px-3 font-mono">{EDA['class_stats'][c]['train_total']}</td>
        <td class="py-2 px-3 font-mono {'text-amber-400 font-bold' if EDA['class_stats'][c]['train_aug']>0 else ''}">{EDA['class_stats'][c]['train_aug']}</td>
        <td class="py-2 px-3 font-mono">{EDA['class_stats'][c]['test_total']}</td>
        <td class="py-2 px-3 font-mono {'text-amber-400 font-bold' if EDA['class_stats'][c]['test_aug']>0 else ''}">{EDA['class_stats'][c]['test_aug']}</td>
    </tr>'''
    for c in classes
)

epochs = list(range(1, len(M["history"]["val_acc"]) + 1))

html = f"""<!DOCTYPE html>
<html lang="ko" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Brain Tumor MRI 데이터 함정 진단 리포트</title>
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
            <div class="p-2.5 bg-purple-600/20 rounded-xl border border-purple-500/30 text-purple-400">
                <i class="fa-solid fa-brain text-2xl"></i>
            </div>
            <div>
                <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                    Brain Tumor MRI <span class="text-xs px-2.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 font-medium border border-purple-500/30">파이널 프로젝트: 데이터 함정 진단</span>
                </h1>
                <p class="text-xs text-slate-400">가설 → 베이스라인 → 클래스 내부 서브그룹 진단 → 처방(평가 프로토콜 교정) → 재측정</p>
            </div>
        </div>
        <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-slate-300">
            <i class="fa-solid fa-microchip text-emerald-400"></i>
            <span>ResNet18 · RTX 4060 (CUDA)</span>
        </div>
    </div>
</header>

<main class="max-w-7xl mx-auto px-4 sm:px-6 space-y-8">

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-magnifying-glass text-blue-400"></i> 1. EDA에서 발견한 것
        </h3>
        <p class="text-sm text-slate-300">클래스별 이미지 수는 완전히 동일하지만(Train 1,400 / Test 400씩), <strong class="text-amber-300">meningioma 클래스만</strong> 파일명에 <code class="text-amber-300">aug</code> 표시가 붙은 증강 이미지가 Train과 Test 양쪽에 섞여 있습니다. 다른 3개 클래스는 증강 이미지가 0장(100% 원본)입니다.</p>
        <div class="overflow-x-auto">
            <table class="w-full text-xs text-left text-slate-300">
                <thead class="bg-slate-800 text-slate-200 border-b border-slate-700 font-semibold">
                    <tr><th class="py-2.5 px-3">클래스</th><th class="py-2.5 px-3">Train 전체</th><th class="py-2.5 px-3">Train 증강본</th><th class="py-2.5 px-3">Test 전체</th><th class="py-2.5 px-3">Test 증강본</th></tr>
                </thead>
                <tbody class="divide-y divide-slate-800">{class_stats_rows}</tbody>
            </table>
        </div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-flask text-amber-400"></i> 2. 가설
        </h3>
        <p class="text-sm text-slate-300 leading-relaxed">
            meningioma의 Test 세트(400장)는 <strong>원본 297장 + 증강본 103장</strong>이 섞여 있다.
            같은 "meningioma test accuracy"라는 하나의 숫자 안에 성격이 다른 두 그룹이 뒤섞여 있으므로,
            이 숫자는 100% 원본으로만 구성된 다른 3개 클래스의 test accuracy와 액면 그대로 비교할 수 없다.
            <strong>증강본 서브그룹과 원본 서브그룹의 정확도를 나눠 재보면 유의미한 격차가 나타날 것이다.</strong>
        </p>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-chart-line text-emerald-400"></i> 3. 베이스라인 학습 결과
        </h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                <div class="text-xs text-slate-400">공식 Test Accuracy</div>
                <div class="text-2xl font-bold text-white">{base['accuracy']*100:.2f}%</div>
            </div>
            <div class="p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                <div class="text-xs text-slate-400">Macro F1</div>
                <div class="text-2xl font-bold text-white">{base['macro_f1']:.4f}</div>
            </div>
            <div class="p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                <div class="text-xs text-slate-400">학습 시간</div>
                <div class="text-2xl font-bold text-white">{M['train_seconds']/60:.1f}분</div>
            </div>
            <div class="p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                <div class="text-xs text-slate-400">모델</div>
                <div class="text-lg font-bold text-white">ResNet18</div>
            </div>
        </div>
        <div class="overflow-x-auto">
            <table class="w-full text-xs text-left text-slate-300">
                <thead class="bg-slate-800 text-slate-200 border-b border-slate-700 font-semibold">
                    <tr><th class="py-2.5 px-3">클래스</th><th class="py-2.5 px-3">Precision</th><th class="py-2.5 px-3">Recall</th><th class="py-2.5 px-3">F1</th></tr>
                </thead>
                <tbody class="divide-y divide-slate-800">{per_class_rows}</tbody>
            </table>
        </div>
        <div class="h-64"><canvas id="curveChart"></canvas></div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-stethoscope text-rose-400"></i> 4. 진단: meningioma 서브그룹 쪼개기 (가설 검증)
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div class="p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                <div class="text-xs text-slate-400">증강본 서브그룹 (n={diag['n_aug']})</div>
                <div class="text-2xl font-bold text-white">{diag['acc_aug']*100:.2f}%</div>
            </div>
            <div class="p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                <div class="text-xs text-slate-400">원본 서브그룹 (n={diag['n_orig']})</div>
                <div class="text-2xl font-bold text-white">{diag['acc_orig']*100:.2f}%</div>
            </div>
            <div class="p-4 rounded-xl border text-center {'bg-rose-950/30 border-rose-800/50' if abs(diag['gap'])>0.03 else 'bg-slate-900/50 border-slate-800'}">
                <div class="text-xs text-slate-400">격차</div>
                <div class="text-2xl font-bold {'text-rose-300' if abs(diag['gap'])>0.03 else 'text-white'}">{diag['gap']*100:+.2f}%p</div>
            </div>
        </div>
        <div class="p-3 bg-rose-950/20 rounded-xl border border-rose-800/30 text-xs text-rose-200/90 leading-relaxed">
            <strong><i class="fa-solid fa-triangle-exclamation"></i> 가설 검증 결과:</strong>
            증강본 서브그룹과 원본 서브그룹의 정확도 격차가 {diag['gap']*100:+.2f}%p로 나타났습니다.
            {'이는 meningioma의 공식 test accuracy가 두 이질적인 그룹을 섞어 계산된 값이라, 단독으로는 신뢰하기 어렵다는 가설을 뒷받침합니다.' if abs(diag['gap'])>0.03 else '격차가 크지 않아, 이번 베이스라인 모델은 증강본과 원본을 비슷하게 다루는 것으로 보이지만 그래도 평가 프로토콜은 공정하게 맞추는 것이 원칙입니다.'}
        </div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-pills text-blue-400"></i> 5. 처방: 공정 평가 프로토콜 (재학습 없이 평가만 교정)
        </h3>
        <p class="text-xs text-slate-400">{fix['description']}</p>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="p-4 bg-slate-900/40 rounded-xl border border-slate-800 space-y-2">
                <div class="text-xs font-bold text-slate-400 uppercase">처방 전 (공식, meningioma 증강 섞임)</div>
                <div class="text-3xl font-bold text-white">{base['accuracy']*100:.2f}%</div>
                <div class="text-xs text-slate-500">Macro F1 {base['macro_f1']:.4f} · n={M['dataset']['official_test']}</div>
            </div>
            <div class="p-4 bg-emerald-950/20 rounded-xl border border-emerald-800/40 space-y-2">
                <div class="text-xs font-bold text-emerald-400 uppercase">처방 후 (meningioma 원본만)</div>
                <div class="text-3xl font-bold text-emerald-300">{fix['accuracy']*100:.2f}%</div>
                <div class="text-xs text-slate-500">Macro F1 {fix['macro_f1']:.4f} · n={fix['n_test']}</div>
            </div>
        </div>
        <div class="overflow-x-auto">
            <table class="w-full text-xs text-left text-slate-300">
                <thead class="bg-slate-800 text-slate-200 border-b border-slate-700 font-semibold">
                    <tr><th class="py-2.5 px-3">클래스</th><th class="py-2.5 px-3">Precision</th><th class="py-2.5 px-3">Recall</th><th class="py-2.5 px-3">F1</th></tr>
                </thead>
                <tbody class="divide-y divide-slate-800">{per_class_rows_fixed}</tbody>
            </table>
        </div>
    </section>

    <section class="glass-card rounded-2xl p-6 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2 border-b border-slate-700/60 pb-3">
            <i class="fa-solid fa-table text-purple-400"></i> 6. 최종 비교 표
        </h3>
        <div class="overflow-x-auto">
            <table class="w-full text-xs text-left text-slate-300">
                <thead class="bg-slate-800/80 text-slate-200 uppercase font-semibold border-b border-slate-700">
                    <tr><th class="py-3 px-4">항목</th><th class="py-3 px-4">Test n</th><th class="py-3 px-4">Accuracy</th><th class="py-3 px-4">Macro F1</th><th class="py-3 px-4">meningioma F1</th></tr>
                </thead>
                <tbody class="divide-y divide-slate-800">
                    <tr>
                        <td class="py-3 px-4 font-medium text-slate-200">처방 전 (베이스라인, 공식 Test)</td>
                        <td class="py-3 px-4 font-mono">{M['dataset']['official_test']}</td>
                        <td class="py-3 px-4 font-mono">{base['accuracy']*100:.2f}%</td>
                        <td class="py-3 px-4 font-mono">{base['macro_f1']:.4f}</td>
                        <td class="py-3 px-4 font-mono">{base['per_class']['f1'][classes.index('meningioma')]:.4f}</td>
                    </tr>
                    <tr class="bg-emerald-950/10">
                        <td class="py-3 px-4 font-semibold text-emerald-300">처방 후 (meningioma 원본만 재평가)</td>
                        <td class="py-3 px-4 font-mono">{fix['n_test']}</td>
                        <td class="py-3 px-4 font-mono font-bold text-emerald-300">{fix['accuracy']*100:.2f}%</td>
                        <td class="py-3 px-4 font-mono font-bold text-emerald-300">{fix['macro_f1']:.4f}</td>
                        <td class="py-3 px-4 font-mono font-bold text-emerald-300">{fix['per_class']['f1'][classes.index('meningioma')]:.4f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </section>
</main>

<footer class="max-w-7xl mx-auto px-6 mt-12 text-center text-xs text-slate-500 border-t border-slate-800/80 pt-6">
    <p>Brain Tumor MRI Final Project | 실제 학습/평가 실행 결과 (PyTorch ResNet18, seed=42)</p>
</footer>

<script>
document.addEventListener('DOMContentLoaded', function() {{
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = 'Pretendard, sans-serif';
    const ctx = document.getElementById('curveChart').getContext('2d');
    new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: {epochs},
            datasets: [
                {{ label: 'Val Accuracy', data: {[round(v,4) for v in M['history']['val_acc']]}, borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.1)', fill: true, borderWidth: 2.5, pointRadius: 2, tension: 0.3 }}
            ]
        }},
        options: {{ responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ position: 'top', labels: {{ boxWidth: 12, font: {{ size: 10 }} }} }} }},
            scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Validation Accuracy' }} }} }}
        }}
    }});
}});
</script>
</body>
</html>
"""

out_path = RESULT_DIR / "brain_tumor_diagnosis_report.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print("리포트 저장:", out_path)
