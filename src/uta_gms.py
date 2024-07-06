from typing import List, Tuple, Union

import pandas as pd
from pulp import GLPK, LpMaximize, LpProblem, LpVariable, lpSum

NECESSARY = 1
POSSIBLE = 2


def uta_gms(df: pd.DataFrame, preferences: List[Tuple[Union[str, int]]]) -> pd.DataFrame:
    alternatives = df.index
    df_relations = pd.DataFrame(0, index=alternatives, columns=alternatives)
    for alt_1_id in alternatives:
        for alt_2_id in alternatives:
            if alt_1_id == alt_2_id:
                df_relations.loc[alt_1_id, alt_2_id] = NECESSARY
                continue

            # Badanie relacji koniecznej alt_1 > alt_2
            problem = LpProblem("uta-gms", LpMaximize)
            epsilon = LpVariable("epsilon")
            decision_variables = {}
            for column in df.columns:
                unique_values = df[column].sort_values().unique()
                decision_variables[column] = [LpVariable(f"x_{column}_{value}", 0, 1) for value in unique_values]
            problem += lpSum([x[0] for x in decision_variables.values()]) == 1
            problem += lpSum([x[-1] for x in decision_variables.values()]) == 0
            for column in df.columns:
                for i in range(len(decision_variables[column]) - 1):
                    problem += decision_variables[column][i] >= decision_variables[column][i + 1]
            for preference in preferences:
                alt_1 = df.loc[preference[0]].to_dict()
                alt_1_variables = [
                    next((variable for variable in decision_variables[k] if str(variable) == f"x_{k}_{v}"), None) for
                    k, v in alt_1.items()]
                alt_2 = df.loc[preference[1]].to_dict()
                alt_2_variables = [
                    next((variable for variable in decision_variables[k] if str(variable) == f"x_{k}_{v}"), None) for
                    k, v, in alt_2.items()]
                problem += lpSum(alt_1_variables) >= lpSum(alt_2_variables) + epsilon
            # Dodanie preferencji badającej relację konieczną
            alt_1 = df.loc[alt_1_id].to_dict()
            alt_1_variables = [
                next((variable for variable in decision_variables[k] if str(variable) == f"x_{k}_{v}"), None) for k, v
                in alt_1.items()]
            alt_2 = df.loc[alt_2_id].to_dict()
            alt_2_variables = [
                next((variable for variable in decision_variables[k] if str(variable) == f"x_{k}_{v}"), None) for k, v,
                in alt_2.items()]
            problem += lpSum(alt_2_variables) >= lpSum(alt_1_variables) + epsilon
            problem += epsilon
            problem.solve(solver=GLPK(msg=False))
            solution = {variable.name: variable.varValue for variable in problem.variables()}
            epsilon_value = solution['epsilon']
            if epsilon_value <= 0:
                df_relations.loc[alt_1_id, alt_2_id] = NECESSARY
                continue

            # Badanie relacji możliwej alt_1 > alt_2
            problem = LpProblem("uta-gms", LpMaximize)
            epsilon = LpVariable("epsilon")
            decision_variables = {}
            for column in df.columns:
                unique_values = df[column].sort_values().unique()
                decision_variables[column] = [LpVariable(f"x_{column}_{value}", 0, 1) for value in unique_values]
            problem += lpSum([x[0] for x in decision_variables.values()]) == 1
            problem += lpSum([x[-1] for x in decision_variables.values()]) == 0
            for column in df.columns:
                for i in range(len(decision_variables[column]) - 1):
                    problem += decision_variables[column][i] >= decision_variables[column][i + 1]
            for preference in preferences:
                alt_1 = df.loc[preference[0]].to_dict()
                alt_1_variables = [
                    next((variable for variable in decision_variables[k] if str(variable) == f"x_{k}_{v}"), None) for
                    k, v in alt_1.items()]
                alt_2 = df.loc[preference[1]].to_dict()
                alt_2_variables = [
                    next((variable for variable in decision_variables[k] if str(variable) == f"x_{k}_{v}"), None) for
                    k, v, in alt_2.items()]
                problem += lpSum(alt_1_variables) >= lpSum(alt_2_variables) + epsilon
            # Dodanie preferencji badającej relację możliwą
            alt_1 = df.loc[alt_1_id].to_dict()
            alt_1_variables = [
                next((variable for variable in decision_variables[k] if str(variable) == f"x_{k}_{v}"), None) for k, v
                in alt_1.items()]
            alt_2 = df.loc[alt_2_id].to_dict()
            alt_2_variables = [
                next((variable for variable in decision_variables[k] if str(variable) == f"x_{k}_{v}"), None) for k, v,
                in alt_2.items()]
            problem += lpSum(alt_1_variables) >= lpSum(alt_2_variables)
            problem += epsilon
            problem.solve(solver=GLPK(msg=False))
            solution = {variable.name: variable.varValue for variable in problem.variables()}
            epsilon_value = solution['epsilon']
            if epsilon_value > 0:
                df_relations.loc[alt_1_id, alt_2_id] = POSSIBLE

    return df_relations
