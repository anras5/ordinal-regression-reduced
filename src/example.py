from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.preprocessing import StandardScaler

from reader.reader import read_csv
from mcda.uta import Criterion
from mcda.report import calculate_heuristics
from methods.autoencoder import AutoencoderModel

df, CRITERIA = read_csv("data/s2.csv")

PREFERENCES = [
    ('a08', 'a09'),
    ('a10', 'a03')
]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

available_points = [2, 3, 4]
n_components = [2, 3, 4]

# results_original = defaultdict(dict)
# for points in available_points:
#     print(f"points: {points}, method: original")
#     criteria = [Criterion(name, points=points) for name in df.columns]
#     f_nec, f_era, f_pwi, f_rai = calculate_heuristics(df, PREFERENCES, criteria)
#     results_original['original'][(f"points: {points}", 'f_nec')] = f_nec
#     results_original['original'][(f"points: {points}", 'f_era')] = f_era
#     results_original['original'][(f"points: {points}", 'f_pwi')] = f_pwi
#     results_original['original'][(f"points: {points}", 'f_rai')] = f_rai
# df_results_original = pd.DataFrame(results_original)
# print(df_results_original)

results = defaultdict(dict)
for points in available_points:
    for n in n_components:
        methods = {
            # 'PCA': PCA(n_components=n, random_state=42),
            'Autoencoder': AutoencoderModel(encoded_dim=n, epochs=300, batch_size=16)
            # 'KernelPCA': KernelPCA(n_components=n, random_state=42),
            # 't-SNE': TSNE(n_components=n, perplexity=10, method='exact', random_state=42),
            # 'MDS': MDS(n_components=n, random_state=42),
            # 'Isomap': Isomap(n_components=n)
        }
        for method_name, method in methods.items():
            print(f"points: {points}, components: {n}, method: {method_name}")
            df_m = pd.DataFrame(
                method.fit_transform(df_scaled), index=df.index, columns=range(n)
            ).map(lambda x: f"{x:.4f}").astype(np.float64)
            criteria = [Criterion(name, points=points) for name in df_m.columns]
            f_nec, f_era, f_pwi, f_rai = calculate_heuristics(df_m, PREFERENCES, criteria)
            results[(method_name, f"dims: {n}")][(f"points: {points}", 'f_nec')] = f_nec
            results[(method_name, f"dims: {n}")][(f"points: {points}", 'f_era')] = f_era
            results[(method_name, f"dims: {n}")][(f"points: {points}", 'f_pwi')] = f_pwi
            results[(method_name, f"dims: {n}")][(f"points: {points}", 'f_rai')] = f_rai
df_results = pd.DataFrame(results)
print(df_results)
