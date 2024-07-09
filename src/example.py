import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from uta import read_csv, uta_gms, Criterion, extreme_ranking

df, CRITERIA = read_csv("data/s2.csv")

PREFERENCES = [
    ('a08', 'a09'),
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

df_relations = uta_gms(df, PREFERENCES, CRITERIA)
print(df_relations.sum().sum())
print(df_relations)
df_extreme = extreme_ranking(df, PREFERENCES, CRITERIA)
print(df_extreme)
