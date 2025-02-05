import pulp


def round_problem(problem: pulp.LpProblem) -> pulp.LpProblem:
    """
    Rounds the coefficients of the problem to 4 decimal places.

    Parameters
    ----------
    problem (pulp.LpProblem): Problem to round.

    Returns
    -------
    pulp.LpProblem: Problem with rounded coefficients.
    """
    p_dict = problem.to_dict()
    for i in range(len(p_dict["constraints"])):
        for j in range(len(p_dict["constraints"][i]["coefficients"])):
            p_dict["constraints"][i]["coefficients"][j]["value"] = round(
                p_dict["constraints"][i]["coefficients"][j]["value"], 4
            )

    _, problem = pulp.LpProblem.from_dict(p_dict)
    return problem
