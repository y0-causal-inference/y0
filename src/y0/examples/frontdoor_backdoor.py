"""Genrate date for the frontdoor graph."""

import numpy as np
import pandas as pd

from y0.dsl import Variable, W, X, Y, Z

__all__ = [
    "generate_data_for_frontdoor_backdoor",
]


def generate_data_for_frontdoor_backdoor(
    num_samples: int,
    treatments: dict[Variable, float] | None = None,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate testing data for the front-door back_door case study.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names in the front-door back_door example
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)
    if W in treatments:
        w = np.full(num_samples, treatments[W])
    else:
        w = generator.binomial(n=1, p=0.3, size=num_samples)

    beta0_x = 0.1
    beta_w_to_x = 0.5
    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        p_x = beta0_x + w * beta_w_to_x
        x = generator.binomial(n=1, p=p_x, size=num_samples)

    beta0_z = 0.5
    beta_x_to_z = 0.4
    if Z in treatments:
        z = np.full(num_samples, treatments[Z])
    else:
        p_z = beta0_z + x * beta_x_to_z
        z = generator.binomial(n=1, p=p_z, size=num_samples)

    beta0_y = 0.1
    beta_z_to_y = 0.7
    beta_w_to_y = 0.1
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        p_y = beta0_y + z * beta_z_to_y + w * beta_w_to_y
        y = generator.binomial(n=1, p=p_y, size=num_samples)

    return pd.DataFrame({X.name: x, Z.name: z, W.name: w, Y.name: y})
