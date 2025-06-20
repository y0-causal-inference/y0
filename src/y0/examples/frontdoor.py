"""Examples for front-door."""

from typing import Any

import numpy as np
import pandas as pd

from y0.dsl import Variable, X, Y, Z

__all__ = [
    "generate_data_for_frontdoor",
]


def _r_exp(x: float | np.ndarray[Any, np.dtype[Any]]) -> float:
    return 1 / (1 + np.exp(x))  # type:ignore


def generate_data_for_frontdoor(
    num_samples: int, treatments: dict[Variable, float] | None = None, *, seed: int | None = None
) -> pd.DataFrame:
    """Generate testing data for the front-door case study.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names in the front-door example
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)

    u = generator.normal(loc=40.0, scale=10.0, size=num_samples)

    beta0_x = -1
    beta_u_to_x = 0.05
    x: np.ndarray[Any, np.dtype[Any]]
    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        p = _r_exp(-beta0_x - u * beta_u_to_x)
        x = generator.binomial(1, p, size=num_samples)

    beta0_z = -1.9
    beta_x_to_z = 0.04
    z: np.ndarray[Any, np.dtype[Any]]
    if Z in treatments:
        z = np.full(num_samples, treatments[Z])
    else:
        z = generator.normal(
            loc=100 * _r_exp(-beta0_z - x * beta_x_to_z), scale=1, size=num_samples
        )

    beta0_y = -1.8
    beta_z_to_y = 0.05
    beta_u_to_y = 0.06
    y: np.ndarray[Any, np.dtype[Any]]
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        y = generator.normal(
            loc=100 * _r_exp(-beta0_y - z * beta_z_to_y - u * beta_u_to_y),
            scale=1,
            size=num_samples,
        )

    return pd.DataFrame({X.name: x, Z.name: z, Y.name: y})
