import pulp


def round_problem(problem: pulp.LpProblem) -> pulp.LpProblem:
    p_dict = problem.to_dict()
    for i in range(len(p_dict["constraints"])):
        for j in range(len(p_dict["constraints"][i]["coefficients"])):
            p_dict["constraints"][i]["coefficients"][j]["value"] = round(
                p_dict["constraints"][i]["coefficients"][j]["value"], 4
            )

    _, problem = pulp.LpProblem.from_dict(p_dict)
    return problem
