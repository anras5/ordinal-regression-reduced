import csv
from dataclasses import dataclass
from itertools import islice
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pulp import GLPK, LpMaximize, LpProblem, LpVariable, lpSum

NECESSARY = 1


@dataclass
class Criterion:
    name: str  # Name of the criterion
    type: bool = True  # True=gain, False=cost
    points: int = 0  # Number of characteristic points


def _get_alternative_variables(
        performances: Dict[Union[str, int], float],
        decision_variables: Dict[Union[str, int], List[LpVariable]],
        criteria: List[Criterion]
) -> List[LpVariable]:
    """
    Calculates variables used in preference constraints for a given alternative.

    Parameters
    ----------
    - performances (Dict[Union[str, int], float]): Performance of the alternative. Keys in the dictionary are criteria names and values are the performances.
    - decision_variables: (Dict[Union[str, int], List[LpVariable]]): Decision variables for the problem. Keys in the dictionary are criteria names.
    - criteria (List[Criterion]): Data about criteria.

    Returns
    -------
    - List[LpVariable]: Variables used in preference constraints for the alternative.
    """
    alt_variables = []
    for criterion_name, value in performances.items():
        # Get criterion from criteria list
        criterion = next((criterion for criterion in criteria if criterion.name == criterion_name), None)
        # Check if the criterion is general or not
        if criterion.points == 0:
            # If the criterion is general, we just add the variable that represents the value of alt_1 on this criterion
            alt_variables.append(
                next((variable for variable in decision_variables[criterion_name]
                      if str(variable) == f"u_{criterion_name}_{value}"), None)
            )
        else:
            # Check if the value is in characteristic points
            check_variable = next((variable for variable in decision_variables[criterion_name]
                                   if str(variable) == f"u_{criterion_name}_{round(value, 4)}"), None)
            if check_variable is not None:
                alt_variables.append(check_variable)
            else:
                # If the criterion is not general and the value is not already a characteristic point,
                # we need to add a variable calculated using linear interpolation
                # Get all values of characteristic points for this criterion (X axis)
                x_values = np.array(
                    [float(str(variable).split("_")[-1]) for variable in decision_variables[criterion_name]]
                )
                # Get the interval that the alternatives belongs to
                position = np.searchsorted(x_values, value)
                x1, x2 = x_values[position - 1], x_values[position]
                y1, y2 = (
                    decision_variables[criterion_name][position - 1],
                    decision_variables[criterion_name][position]
                )
                alt_variables.append(y1 + (value - x1) * (y2 - y1) / (x2 - x1))
    return alt_variables


def uta_gms(df: pd.DataFrame, preferences: List[Tuple[Union[str, int]]], criteria: List[Criterion]) -> pd.DataFrame:
    """
    Calculate UTA-GMS necessary relations based on provided data.

    Parameters
    ----------
    - df (pd.DataFrame): Performance table of the alternatives.
    - preferences (List[Tuple[Union[str, int]]): Preferences of the user. Alternatives inside preferences have to exist in df.index.
    - criteria (List[Criterion]): Data about criteria. Their names have to match columns in df.columns.

    Returns
    -------
    - pd.DataFrame: Matrix with necessary relations. 0 indicates no relation, 1 indicates necessary relation.
    """
    alternatives = df.index
    df_relations = pd.DataFrame(0, index=alternatives, columns=alternatives)
    for alt_1_id in alternatives:
        for alt_2_id in alternatives:
            if alt_1_id == alt_2_id:
                df_relations.loc[alt_1_id, alt_2_id] = NECESSARY
                continue

            # Testing necessary relation alt_1 > alt_2
            problem = LpProblem("uta-gms", LpMaximize)
            epsilon = LpVariable("epsilon")
            decision_variables = {}

            # Creating decision variables
            for criterion in criteria:
                if criterion.points == 0:
                    # General function
                    unique_values = df[criterion.name].sort_values().unique()
                    decision_variables[criterion.name] = [
                        LpVariable(f"u_{criterion.name}_{value}", 0, 1) for value in unique_values
                    ]
                else:
                    # Function with a given number of characteristic points
                    _min_x, _max_x = df[criterion.name].min(), df[criterion.name].max()
                    decision_variables[criterion.name] = [
                        LpVariable(f"u_{criterion.name}_{round(value, 4)}", 0, 1)
                        for value in np.linspace(_min_x, _max_x, criterion.points)
                    ]

            # Normalization
            # Hypothetical best utilities
            problem += lpSum([
                decision_variables[criterion.name][-1] if criterion.type else decision_variables[criterion.name][0]
                for criterion in criteria
            ]) == 1
            # Hypothetical worst utilities
            problem += lpSum([
                decision_variables[criterion.name][0] if criterion.type else decision_variables[criterion.name][-1]
                for criterion in criteria
            ]) == 0

            # Monotonicity
            for criterion in criteria:
                for i in range(len(decision_variables[criterion.name]) - 1):
                    if criterion.type:
                        problem += decision_variables[criterion.name][i] <= decision_variables[criterion.name][i + 1]
                    else:
                        problem += decision_variables[criterion.name][i] >= decision_variables[criterion.name][i + 1]

            # Preferences
            for preference in preferences:
                alt_1_variables = _get_alternative_variables(
                    df.loc[preference[0]].to_dict(),
                    decision_variables,
                    criteria
                )
                alt_2_variables = _get_alternative_variables(
                    df.loc[preference[1]].to_dict(),
                    decision_variables,
                    criteria
                )
                problem += lpSum(alt_1_variables) >= lpSum(alt_2_variables) + epsilon

            # Necessary relation constraint
            alt_1_variables = _get_alternative_variables(
                df.loc[alt_1_id].to_dict(),
                decision_variables,
                criteria
            )
            alt_2_variables = _get_alternative_variables(
                df.loc[alt_2_id].to_dict(),
                decision_variables,
                criteria
            )
            problem += lpSum(alt_2_variables) >= lpSum(alt_1_variables) + epsilon
            problem += epsilon
            problem.solve(solver=GLPK(msg=False))
            solution = {variable.name: variable.varValue for variable in problem.variables()}
            epsilon_value = solution['epsilon']
            if epsilon_value <= 0:
                df_relations.loc[alt_1_id, alt_2_id] = NECESSARY
                continue

    return df_relations


def read_csv(filepath: str, convert_to_gain: bool = True) -> (pd.DataFrame, List[Criterion]):
    """
    Reads a csv file and converts it into a pandas dataframe (performances) and criteria list.
    """
    df = pd.read_csv(filepath, skiprows=2, sep=';', index_col=0)
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
