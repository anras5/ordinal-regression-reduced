from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from .smaa import calculate_samples, SamplerException, calculate_pwi, calculate_rai
from .uta import Criterion, calculate_extreme_ranking, calculate_uta_gms


def calculate_heuristics(
        df: pd.DataFrame,
        preferences: List[Tuple[Union[str, int]]],
        criteria: List[Criterion],
        number_of_samples: int = 1000
) -> Tuple[np.int64, np.float64, np.float64, np.float64]:
    """
    Calculates heuristics for provided dataset and preferences.
    - f_NEC - number of necessary preference relations
    - f_ERA - average difference between the extreme ranks
    - f_PWI - uncertainty for all pairs of alternatives
    - f_RAI - uncertainty in the ranks attained by all alternatives

    Parameters
    ----------
    - df (pd.DataFrame): Performance table of the alternatives.
    - preferences (List[Tuple[Union[str, int]]): Preferences of the user. Alternatives inside preferences have to exist in df.index.
    - criteria (List[Criterion]): Data about criteria. Their names have to match columns in df.columns.
    - number_of_samples (int): Number of samples to use in Polyrun.

    Returns
    -------
    - Tuple[np.int64, np.float64, np.float64, np.float64]: f_NEC, f_ERA, f_PWI, f_RAI
    """

    # Calculate f_NEC
    df_utagms = calculate_uta_gms(df, preferences, criteria)
    np.fill_diagonal(df_utagms.values, 0)
    f_nec = df_utagms.sum().sum()

    # Calculate f_ERA
    df_extreme = calculate_extreme_ranking(df, preferences, criteria)
    f_era = (df_extreme['worst'] - df_extreme['best']).mean()

    # Calculate samples
    try:
        df_samples = calculate_samples(df, preferences, criteria, number_of_samples)
    except SamplerException as e:
        print(e)
        return f_nec, f_era, None, None

    # Calculate f_PWI
    df_pwi = calculate_pwi(df, df_samples)
    df_pwi_safe = np.where(df_pwi == 0, 1e-10, df_pwi)  # so there won't be a 0 passed to the log function
    df_f_pwi = (-df_pwi * np.log2(df_pwi_safe)).round(4).fillna(0.0)
    f_pwi = df_f_pwi.sum().sum() / (len(df_f_pwi) * (len(df_f_pwi) - 1))

    # Calculate f_RAI
    df_rai = calculate_rai(df, df_samples)
    df_rai_safe = np.where(df_rai == 0, 1e-10, df_rai)
    df_f_rai = (-df_rai * np.log2(df_rai_safe)).round(4).fillna(0.0)
    f_rai = (df_f_rai.sum(axis=1) / len(df_f_rai)).sum()

    return f_nec, f_era, f_pwi, f_rai
