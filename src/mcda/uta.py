from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pulp import GLPK, LpMaximize, LpMinimize, LpProblem, LpVariable, lpSum

NECESSARY = 1
BEST, WORST = "best", "worst"


@dataclass
class Criterion:
    name: str  # Name of the criterion
    type: bool = True  # True=gain, False=cost
    points: int = 0  # Number of characteristic points


def _minus_handler(value: float) -> str:
    return "_" + str(value)[1:] if value < 0 else value


def _get_alternative_variables(
    performances: Dict[Union[str, int], float],
    decision_variables: Dict[Union[str, int], List[LpVariable]],
    criteria: List[Criterion],
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
                      if str(variable) == f"u#{criterion_name}#{_minus_handler(value)}"), None)
            )
        else:
            # Check if the value is in characteristic points
            check_variable = next((variable for variable in decision_variables[criterion_name]
                                   if str(variable) == f"u#{criterion_name}#{_minus_handler(round(value, 4))}"), None)
            if check_variable is not None:
                alt_variables.append(check_variable)
            else:
                # If the criterion is not general and the value is not already a characteristic point,
                # we need to add a variable calculated using linear interpolation
                # Get all values of characteristic points for this criterion (X axis)
                x_values = np.array(
                    sorted(list(map(
                        lambda x: -1 * float(x[1:]) if x.startswith("_") else float(x),
                        [str(variable).split("#")[-1] for variable in decision_variables[criterion_name]]
                    )))
                )
                # Get the interval that the alternatives belongs to
                position = np.searchsorted(x_values, value)
                x1, x2 = x_values[position - 1], x_values[position]
                y1, y2 = (
                    decision_variables[criterion_name][position - 1],
                    decision_variables[criterion_name][position],
                )
                coefficient = round((value - x1) / (x2 - x1), 4)
                alt_variables.append(y1 + coefficient * (y2 - y1))
    return alt_variables


def _get_uta_problem(
    df: pd.DataFrame,
    preferences: List[Tuple[Union[str, int], Union[str, int]]],
    criteria: List[Criterion],
    name: str,
    direction: int,
) -> Tuple[LpProblem, Dict[str, List[LpVariable]], LpVariable]:
    """
    Utility function for creating a general problem for UTA. Creates a problem with constraints for:
    - normalization
    - monotonicity
    - preferences

    Parameters
    ----------
    - df (pd.DataFrame): Performance table of the alternatives.
    - preferences (List[Tuple[Union[str, int]]): Preferences of the user. Alternatives inside preferences have to exist in df.index.
    - criteria (List[Criterion]): Data about criteria. Their names have to match columns in df.columns.
    - name (str): Name of the problem.
    - direction (int): Should either be LpMaximize or LpMinimize.

    Returns
    -------
    - LpProblem: The problem to be solved.
    - Dict[str, List[LpVariable]]: Decision variables for each criterion.
    - LpVariable: The epsilon variable.
    """
    problem = LpProblem(name, direction)
    epsilon = LpVariable("epsilon")
    decision_variables = {}

    # Creating decision variables
    for criterion in criteria:
        if criterion.points == 0:
            # General function
            unique_values = df[criterion.name].sort_values().unique()
            decision_variables[criterion.name] = [
                LpVariable(f"u#{criterion.name}#{value}", 0, 1) for value in unique_values
            ]
        else:
            # Function with a given number of characteristic points
            _min_x, _max_x = df[criterion.name].min(), df[criterion.name].max()
            decision_variables[criterion.name] = [
                LpVariable(f"u#{criterion.name}#{round(value, 4)}", 0, 1)
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
        alt_1_variables = _get_alternative_variables(df.loc[preference[0]].to_dict(), decision_variables, criteria)
        alt_2_variables = _get_alternative_variables(df.loc[preference[1]].to_dict(), decision_variables, criteria)
        problem += lpSum(alt_1_variables) >= lpSum(alt_2_variables) + epsilon

    return problem, decision_variables, epsilon


def calculate_uta_gms(
    df: pd.DataFrame,
    preferences: List[Tuple[Union[str, int]]],
    criteria: List[Criterion],
) -> pd.DataFrame:
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
            problem, decision_variables, epsilon = _get_uta_problem(df, preferences, criteria, "uta-gms", LpMaximize)

            # Necessary relation constraint
            alt_1_variables = _get_alternative_variables(df.loc[alt_1_id].to_dict(), decision_variables, criteria)
            alt_2_variables = _get_alternative_variables(df.loc[alt_2_id].to_dict(), decision_variables, criteria)
            problem += lpSum(alt_2_variables) >= lpSum(alt_1_variables) + epsilon
            problem += epsilon
            problem.solve(solver=GLPK(msg=False))
            solution = {variable.name: variable.varValue for variable in problem.variables()}
            epsilon_value = solution["epsilon"]
            if epsilon_value <= 0:
                df_relations.loc[alt_1_id, alt_2_id] = NECESSARY
                continue

    return df_relations


def calculate_extreme_ranking(
    df: pd.DataFrame,
    preferences: List[Tuple[Union[str, int], Union[str, int]]],
    criteria: List[Criterion],
) -> pd.DataFrame:
    alternatives = df.index
    df_extreme = pd.DataFrame(
        [[0, len(alternatives)]] * len(alternatives),
        index=alternatives,
        columns=[BEST, WORST],
    )
    for alt_1_id in alternatives:
        for _type in [BEST, WORST]:
            # Best rank for alternative
            problem, decision_variables, epsilon = _get_uta_problem(
                df, preferences, criteria, "extreme-analysis", LpMinimize
            )
            problem += epsilon == 0.0001

            binary_variables_rank = {}
            M = 100
            for alt_2_id in alternatives:
                if alt_1_id == alt_2_id:
                    continue

                alt_1_variables = _get_alternative_variables(df.loc[alt_1_id].to_dict(), decision_variables, criteria)
                alt_2_variables = _get_alternative_variables(df.loc[alt_2_id].to_dict(), decision_variables, criteria)

                variable_rank = LpVariable(f"r_{alt_2_id}", cat="Binary")
                binary_variables_rank[f"r_{alt_2_id}"] = variable_rank
                if _type == BEST:
                    problem += lpSum(alt_1_variables) >= lpSum(alt_2_variables) - M * variable_rank
                if _type == WORST:
                    problem += lpSum(alt_2_variables) >= lpSum(alt_1_variables) + epsilon - M * variable_rank

            problem += lpSum(binary_variables_rank.values())
            problem.solve(solver=GLPK(msg=False))
            for variable in problem.variables():
                if variable.name.startswith("r_") and variable.varValue == 1:
                    if _type == BEST:
                        df_extreme.loc[alt_1_id, BEST] += 1
                    if _type == WORST:
                        df_extreme.loc[alt_1_id, WORST] -= 1
        df_extreme.loc[alt_1_id, BEST] += 1

    return df_extreme
