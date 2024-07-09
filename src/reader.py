import csv
from itertools import islice
from typing import List

import pandas as pd

from uta import Criterion


def read_csv(filepath: str, convert_to_gain: bool = True) -> (pd.DataFrame, List[Criterion]):
    """
    Reads a csv file and converts it into a pandas dataframe (performances) and criteria list.
    """
    df = pd.read_csv(filepath, skiprows=2, sep=';', index_col=0).astype(float)
    types = [True if t.strip().startswith('g') else False
             for t in next(islice(csv.reader(open('data/s2.csv')), 1, 2))[0].split(';') if t]
    points = [int(p) for p in next(islice(csv.reader(open('data/s2.csv')), 0, 1))[0].split(';') if p]
    criteria = [Criterion(name, _type, points) for name, _type, points in zip(df.columns, types, points)]

    if convert_to_gain:
        criteria_to_convert = filter(lambda x: not x.type, criteria)
        for criterion in criteria_to_convert:
            df[criterion.name] = df[criterion.name] * -1

        for i in range(len(criteria)):
            criteria[i].type = True

    return df, criteria
