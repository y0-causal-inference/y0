"""Examples for smoking/cancer."""

import numpy as np
import pandas as pd

from y0.dsl import C, S, T, U, Variable

__all__ = [
    "generate_data_for_smoke_cancer",
]


def generate_data_for_smoke_cancer(
    num_samples: int,
    treatments: dict[Variable, float] | None = None,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate testing data for the smoking/cancer case study.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names in the front-door back_door example
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)
    if U in treatments:
        u = np.full(num_samples, treatments[U])
    else:
        u = generator.normal(loc=10, scale=5, size=num_samples)

    beta0_s = 1.6
    beta_u_to_s = -0.3
    if S in treatments:
        s = np.full(num_samples, treatments[S])
    else:
        p_s = 1 / (1 + np.exp(beta0_s + u * beta_u_to_s))
        s = generator.binomial(n=1, p=p_s, size=num_samples)

    beta0_t = 0.3
    beta_u_to_t = 0.4
    beta_s_to_t = 2
    if T in treatments:
        t = np.full(num_samples, treatments[T])
    else:
        loc_t = beta0_t + u * beta_u_to_t + s * beta_s_to_t
        t = generator.normal(loc=loc_t, scale=4, size=num_samples)

    beta0_c = 1.5
    beta_s_to_c = -0.7
    beta_t_to_c = -0.1
    if C in treatments:
        c = np.full(num_samples, treatments[C])
    else:
        p_c = 1 / (1 + np.exp(beta0_c + s * beta_s_to_c + t * beta_t_to_c))
        c = generator.binomial(n=1, p=p_c, size=num_samples)

    return pd.DataFrame({S.name: s, T.name: t, C.name: c})
