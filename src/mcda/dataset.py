import csv
from dataclasses import dataclass
from itertools import islice
from typing import List

import pandas as pd


@dataclass
class Criterion:
    name: str  # Name of the criterion
    type: bool = True  # True=gain, False=cost
    points: int = 0  # Number of characteristic points


class MCDADataset:
    def __init__(self, data: pd.DataFrame, criteria: List[Criterion]):
        self.data = data
        self.criteria = criteria

    @staticmethod
    def read_csv(filepath: str, convert_to_gain: bool = True) -> (pd.DataFrame, List[Criterion]):
        """
        Reads a csv file and converts it into a pandas dataframe (performances) and criteria list.
        """
        df = pd.read_csv(filepath, skiprows=2, sep=";", index_col=0).astype(float)
        types = [t.strip().startswith("g") for t in next(islice(csv.reader(open(filepath)), 1, 2))[0].split(";") if t]
        points = [int(p) for p in next(islice(csv.reader(open(filepath)), 0, 1))[0].split(";") if p]
        criteria = [Criterion(name, _type, points) for name, _type, points in zip(df.columns, types, points)]

        if convert_to_gain:
            criteria_to_convert = filter(lambda x: not x.type, criteria)
            for criterion in criteria_to_convert:
                df[criterion.name] = df[criterion.name] * -1

            for i in range(len(criteria)):
                criteria[i].type = True

        return MCDADataset(df, criteria)

    def write_csv(self, filepath: str) -> None:
        """
        Writes the dataset to a csv file.
        """
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow([""] + [c.points for c in self.criteria])
            writer.writerow([""] + ["gain" if c.type else "cost" for c in self.criteria])
            self.data.to_csv(file, sep=";")
