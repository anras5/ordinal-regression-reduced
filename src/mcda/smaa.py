import subprocess
from tempfile import TemporaryFile
from typing import Dict, List, Tuple, Union, Any

import numpy as np
import pandas as pd
from pulp import LpMaximize

from .uta import Criterion, _get_uta_problem, _minus_handler


class SamplerException(Exception):
    pass


def calculate_samples(
        df: pd.DataFrame,
        preferences: List[Tuple[Union[str, int]]],
        criteria: List[Criterion],
        number_of_samples: int = 1000,
        seed: int = 42
) -> pd.DataFrame:
    """
    Calculates samples using Polyrun sampler.

    Parameters
    ----------
    - df (pd.DataFrame): Dataframe with performances of alternatives on criteria.
    - preferences (List[Tuple[Union[str, int]]]): List of tuples of preferences.
    - criteria (List[Criterion]): List of criteria.
    - number_of_samples (int): Number of samples to be passed to Polyrun.
    - seed (int): Random seed for Polyrun.

    Returns
    -------
    - pd.DataFrame: Dataframe with samples.
    """
    problem, decision_variables, epsilon = _get_uta_problem(
        df,
        preferences,
        criteria,
        "smaa",
        LpMaximize
    )
    problem += epsilon
    # print(problem)

    all_variables = {str(variable): i for i, variable in enumerate(problem.variables())}
    constraints = problem.constraints.copy()

    with TemporaryFile("w+") as input_file, TemporaryFile("w+") as output_file, TemporaryFile("w+") as error_file:
        # Standard constraints - worst/best bounds, monotonicity and preferences
        for c_name, c in constraints.items():

            c_dict = c.toDict()
            c_variables = {coeff['name']: coeff['value'] for coeff in c_dict['coefficients']}

            # Handle _C2 which is the hypothetical worst performance equal to 0 constraint separately
            if c_name == '_C2':
                for variable in all_variables.keys():
                    if variable in c_variables.keys():
                        constraint = [0] * len(all_variables)
                        constraint[all_variables[variable]] = 1
                        constraint.extend(["=", 0])
                        input_file.write(" ".join(map(str, constraint)) + "\n")
            else:
                constraint = []
                # iterate over variables and add only those that are present in the constraint
                for variable in all_variables.keys():
                    if variable in c_variables.keys():
                        constraint.append(round(c_variables[variable], 4))
                    else:
                        constraint.append(0)

                # choose appropriate sign
                match c_dict['sense']:
                    case -1:
                        constraint.append("<=")
                    case 0:
                        constraint.append("=")
                    case 1:
                        constraint.append(">=")

                # add constant multiplied by -1 (because the value has to be on the right side of the constraint)
                constraint.append(-c_dict['constant'])
                input_file.write(" ".join(map(str, constraint)) + "\n")

        # Add epsilon constraint
        constraint = [0] * len(all_variables)
        constraint[all_variables["epsilon"]] = 1
        constraint.extend([">=", "0.000001"])
        input_file.write(" ".join(map(str, constraint)) + "\n")

        input_file.seek(0)
        # print("Input to the sampler:")
        # for line in input_file:
        #     print(line, end="")

        input_file.seek(0)
        error_file.seek(0)
        subprocess.call(
            [
                'java',
                '-jar',
                "/app/polyrun-1.1.0-jar-with-dependencies.jar",
                '-n',
                str(number_of_samples),
                '-s',
                str(seed)
            ],
            stdin=input_file,
            stdout=output_file,
            stderr=error_file
        )
        error_file.seek(0)
        error = error_file.read()
        if error:
            raise SamplerException(error)
        else:
            output_file.seek(0)
            samples = []
            for line in output_file:
                samples.append([float(value) for value in line.split('\t')])
            return pd.DataFrame(samples, columns=all_variables.keys())


def _get_alternative_utility(
        performances: Dict[Union[str, int], float],
        dv_values: Dict[str, float],
        criteria_abscissa: Dict[Any, List[float]]
) -> float:
    """
    Calculates comprehensive utility for a particular alternative.

    Parameters
    ----------
    - performances (Dict[Union[str, int], float]): Performances of an alternative. Keys are criteria names and values the performances on the criteria.
    - dv_values (Dict[str, float]): Values for the decision variables (marginal functions).
    Keys are names of the decision variables and values are the corresponding values on the marginal function that variables get. Can be possessed by using a sampler.
    - criteria_abscissa (Dict[Union[str, int], List[float]]): Dictionary where the keys are criteria names and values are unique sorted decision variables values on abscissa.

    Returns
    -------
    - float: Comprehensive utility of the alternative.
    """
    utility = 0
    for criterion_name, value in performances.items():
        # Get criterion from criteria list
        # Check if the value is in characteristic points
        check_value = dv_values.get(f"u#{criterion_name}#{_minus_handler(value)}", None)
        if check_value is not None:
            utility += check_value
        else:
            # If the criterion is not general and the value is not already a characteristic point,
            # we need to add a value calculated using linear interpolation
            # Get all values of characteristic points for this criterion (X axis)
            x_values = criteria_abscissa[criterion_name]
            # Get the interval that the alternatives belongs to
            position = np.searchsorted(criteria_abscissa[criterion_name], value)
            x1, x2 = x_values[position - 1], x_values[position]
            y1, y2 = (
                dv_values[f"u#{criterion_name}#{_minus_handler(x_values[position - 1])}"],
                dv_values[f"u#{criterion_name}#{_minus_handler(x_values[position])}"]
            )
            coefficient = round((value - x1) / (x2 - x1), 4)
            utility += y1 + coefficient * (y2 - y1)
    return utility


def calculate_pwi(df: pd.DataFrame, df_samples: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pairwise Winning Indices.

    Parameters
    ----------
    - df (pd.DataFrame): Performance table of the alternatives.
    - df_samples (pd.DataFrame): Samples tables. Rows are samples, columns are decision variables, values are values of the decision variables on marginal functions.

    Returns
    -------
    - pd.DataFrame: Pairwise Winning Indices
    """
    df_pwi = pd.DataFrame(0, index=df.index, columns=df.index)

    # Calculate criteria abscissa
    criteria_abscissa = {criterion_name: [] for criterion_name in df.columns}
    for criterion_name in criteria_abscissa.keys():
        criterion_decision_variables = [name for name in df_samples.columns if name.startswith(f'u#{criterion_name}#')]
        criteria_abscissa[criterion_name] = list(
            sorted(list(map(
                lambda x: -1 * float(x[1:]) if x.startswith("_") else float(x),
                [str(variable).split("#")[-1] for variable in criterion_decision_variables]
            )))
        )

    # Calculate PWI
    for sample_id in df_samples.index:
        utilities = {
            alt_id: _get_alternative_utility(
                df.loc[alt_id].to_dict(),
                df_samples.loc[sample_id].to_dict(),
                criteria_abscissa
            )
            for alt_id in df.index
        }
        alternatives_sorted = list(map(lambda x: x[0], sorted(utilities.items(), key=lambda x: -x[1])))
        for i, alt_id in enumerate(alternatives_sorted):
            df_pwi.loc[alt_id, alternatives_sorted[i + 1:]] += 1
    df_pwi = df_pwi / len(df_samples)
    return df_pwi


def calculate_rai(df: pd.DataFrame, df_samples: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Rank Acceptability Indices.

    Parameters
    ----------
    - df (pd.DataFrame): Performance table of the alternatives.
    - df_samples (pd.DataFrame): Samples tables. Rows are samples, columns are decision variables, values are values of the decision variables on marginal functions.

    Returns
    -------
    - pd.DataFrame: Rank Acceptability Indices
    """

    df_rai = pd.DataFrame(0, index=df.index, columns=range(1, len(df.index) + 1))

    # Calculate criteria abscissa
    criteria_abscissa = {criterion_name: [] for criterion_name in df.columns}
    for criterion_name in criteria_abscissa.keys():
        criterion_decision_variables = [name for name in df_samples.columns if name.startswith(f'u#{criterion_name}#')]
        criteria_abscissa[criterion_name] = list(
            sorted(list(map(
                lambda x: -1 * float(x[1:]) if x.startswith("_") else float(x),
                [str(variable).split("#")[-1] for variable in criterion_decision_variables]
            )))
        )

    # Calculate RAI
    for sample_id in df_samples.index:
        utilities = {
            alt_id: _get_alternative_utility(
                df.loc[alt_id].to_dict(),
                df_samples.loc[sample_id].to_dict(),
                criteria_abscissa
            )
            for alt_id in df.index
        }
        utilities_sorted = sorted(utilities.items(), key=lambda x: -x[1])
        for i, utility in enumerate(utilities_sorted, start=1):
            df_rai.loc[utility[0], i] += 1

    df_rai = df_rai / len(df_samples)
    return df_rai
