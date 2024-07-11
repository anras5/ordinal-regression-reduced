import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from smaa import calculate_rai, calculate_samples, calculate_pwi
from report import calculate_heuristics
from uta import calculate_uta_gms
from reader import read_csv

df, CRITERIA = read_csv("data/s2-pca.csv")

PREFERENCES = [
    ('a10', 'a03')
]
# df.loc[:, ['g1', 'g2', 'g3']] = df.loc[:, ['g1', 'g2', 'g3']] * -1
# CRITERIA = [
#     Criterion('g1', False, 3),
#     Criterion('g2', False, 3),
#     Criterion('g3', False, 3),
#     Criterion('g4', True, 3),
#     Criterion('g5', True, 3)
# ]
# print(df)

# df_relations = calculate_uta_gms(df, PREFERENCES, CRITERIA)
# print(df_relations.sum().sum())
# print("GMS")
# print(df_relations)
# # df_extreme = extreme_ranking(df, PREFERENCES, CRITERIA)
# # print(df_extreme)
print("SAMPLES")
df_samples = calculate_samples(df, PREFERENCES, CRITERIA)
print("PWI")
print(calculate_pwi(df, df_samples))
# print("RAI")
# df_rai = calculate_rai(df, df_samples)
# print(df_rai.sum(axis=1))

# print(calculate_heuristics(df, PREFERENCES, CRITERIA))
