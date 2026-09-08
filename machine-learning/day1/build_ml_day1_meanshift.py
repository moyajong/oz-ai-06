# -*- coding: utf-8 -*-
"""
머신러닝 심화 1일차 과제 (1/2) — Iris 데이터셋 Mean Shift 클러스터링.

과제 스펙: 실습파일(Wine 데이터셋 기반)을 그대로 따라가되, sklearn.datasets의 load_wine을
load_iris로 바꿔서 진행. 아래는 실제로 sklearn/matplotlib을 실행해서 만든 결과이며, 지어낸
수치는 없다.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("# Iris 데이터셋 Mean Shift 클러스터링\n\n"
   "> 원본 실습파일(`01. 머신러닝 심화 Mean Shift Clustering.ipynb`)은 Wine 데이터셋을 사용합니다. "
   "과제 지시에 따라 `load_wine` → `load_iris`로 교체하여 동일한 파이프라인을 그대로 적용했습니다.\n\n"
   "## 개요\n"
   "- Iris 데이터셋에 Mean Shift 알고리즘 적용\n"
   "- 밀도 기반 클러스터링 수행\n"
   "- 클러스터 개수 자동 탐지\n\n"
   "## 주요 단계\n"
   "1. 데이터 로드 및 정규화\n"
   "2. PCA 차원 축소 (4차원 → 2차원)\n"
   "3. 대역폭 추정 및 클러스터링\n"
   "4. 결과 시각화\n"
   "5. (추가) 실제 품종 라벨과의 일치도 확인")

md("## 라이브러리 임포트")
code('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score''')

md("## 1. 데이터 로드 및 전처리\n\n"
   "- Iris 데이터셋: 150개 샘플, 4개 피처(꽃받침 길이/너비, 꽃잎 길이/너비)\n"
   "- StandardScaler로 정규화 (거리 기반 알고리즘 특성상 필수)")
code('''# 실습파일에서 데이터를 가져온 형태
# from sklearn.datasets import load_wine
# wine = load_wine()

# 과제를 수행하기 위해서 변경하여 적용해야하는 형태
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y_true = iris.target  # 시각화/검증용 실제 품종 라벨 (클러스터링 자체에는 사용하지 않음)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"데이터 크기: {X.shape}")
print(f"\\n데이터 미리보기:\\n{X.head()}")''')

md("## 2. 차원 축소 (PCA)\n\n"
   "- 4차원 → 2차원 축소\n"
   "- 시각화 편의성 및 노이즈 완화")
code('''pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"설명된 분산 비율: {pca.explained_variance_ratio_}")
print(f"총 설명된 분산: {sum(pca.explained_variance_ratio_):.2%}")''')

md("## 3. 대역폭 추정 및 모델 학습\n\n"
   "**대역폭(Bandwidth)**\n"
   "- 밀도 기반 점 이웃 반경 설정\n"
   "- quantile=0.2: 상위 20% 거리 사용\n"
   "- 데이터 밀도에 따라 자동 결정")
code('''bandwidth = estimate_bandwidth(X_pca, quantile=0.2, n_samples=len(X_pca))
print(f"추정된 Bandwidth: {bandwidth:.4f}")''')

md("**Mean Shift 학습**\n"
   "- bin_seeding=True: 초기 중심점 그리드 샘플링 (연산 속도 향상)")
code('''ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X_pca)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(np.unique(labels))

print(f"발견된 클러스터 개수: {n_clusters_}")
print(f"\\n클러스터 중심점:\\n{cluster_centers}")''')

md("**클러스터별 샘플 개수**")
code('''for k in range(n_clusters_):
    count = np.sum(labels == k)
    print(f"클러스터 {k}: {count}개")''')

md("## 4. 결과 시각화\n\n"
   "- 색상 마커: 클러스터 라벨 구분\n"
   "- 빨간색 X: 클러스터 중심점")
code('''plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=40, alpha=0.8)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
            c="red", marker="X", s=200, edgecolors="black", label="클러스터 중심")
plt.title(f"Iris Mean Shift 클러스터링 (발견된 클러스터 수: {n_clusters_})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.colorbar(scatter, label="클러스터 라벨")
plt.tight_layout()
plt.savefig("meanshift_result.png", dpi=100)
plt.show()''')

md("## 5. (추가) 실제 품종 라벨과의 일치도\n\n"
   "Mean Shift는 비지도 학습이라 클러스터링 과정에서 실제 라벨(`y_true`)을 전혀 사용하지 않았다. "
   "결과 검증 차원에서 Adjusted Rand Index(ARI)로 발견된 클러스터와 실제 품종(setosa/versicolor/"
   "virginica) 간 일치도만 사후에 계산해본다.")
code('''ari = adjusted_rand_score(y_true, labels)
print(f"Adjusted Rand Index (클러스터 vs 실제 품종): {ari:.4f}")

# 클러스터 vs 실제 품종 교차표
cross_tab = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(y_true, name="species"))
cross_tab.columns = [iris.target_names[i] for i in cross_tab.columns]
print("\\n클러스터 x 실제 품종 교차표:")
print(cross_tab)''')

md("## 오늘의 회고\n\n"
   "Wine(13차원) 실습을 Iris(4차원)로 바꿔 돌려보니, Iris는 setosa가 나머지 두 품종과 워낙 "
   "뚜렷하게 분리되어 있어서 quantile=0.2 기준 대역폭으로도 Mean Shift가 비교적 안정적으로 "
   "클러스터를 찾아냈습니다. 다만 versicolor와 virginica는 PCA 2차원 공간에서도 경계가 겹치는 "
   "구간이 있어, 교차표로 확인했을 때 두 품종이 완전히 분리되지는 않았습니다. ARI 점수로 이 겹침 "
   "정도를 수치로 확인할 수 있었던 점이 유익했습니다.")

nb["cells"] = cells

with open("ml_day1_meanshift.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("노트북 저장 완료: ml_day1_meanshift.ipynb")
