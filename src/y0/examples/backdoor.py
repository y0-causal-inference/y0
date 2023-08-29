"""Examples for backdoor."""

import numpy as np
import pandas as pd

from y0.dsl import Variable, X, Y, Z

__all__ = [
    "generate_data_for_backdoor",
]


def _r_exp(x):
    return 1 / (1 + np.exp(x))


def generate_data_for_backdoor(
    num_samples: int, treatments: dict[Variable, float] | None = None, *, seed: int | None = None
) -> pd.DataFrame:
    """Generate testing data for the back-door case study.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names in the back-door example
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)

    if Z in treatments:
        z = np.full(num_samples, treatments[Z])
    else:
        z = generator.normal(loc=40.0, scale=10.0, size=num_samples)

    beta0_x = -1
    beta_z_to_x = 0.05

    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        p = _r_exp(-beta0_x - z * beta_z_to_x)
        x = generator.binomial(1, p, size=num_samples)

    beta0_y = -1.8
    beta_z_to_y = 0.05
    beta_x_to_y = 0.06
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        y = generator.normal(
            loc=100 * _r_exp(-beta0_y - z * beta_z_to_y - x * beta_x_to_y),
            scale=1,
            size=num_samples,
        )

    return pd.DataFrame({X.name: x, Z.name: z, Y.name: y})
