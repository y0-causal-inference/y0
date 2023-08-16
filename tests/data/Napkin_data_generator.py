import numpy as np
import pandas as pd
from ananke.estimation import CausalEffect
from y0.examples import napkin


# Generate observational data for napkin
def generate_obs_data_for_napkin(
    seed: int, num_samples: int, treatment_assignment: int | None = None
) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    # U1 is the latent variable that is a common cause of W and X
    u1 = generator.normal(loc=3, scale=1, size=num_samples)
    # U2 is the latent variable that is a common cause of W and Y
    u2 = generator.normal(loc=5, scale=1, size=num_samples)
    w = generator.gamma(
        shape=1 / (1 * (u1 * 0.3 + 0.5 * u2) ** 2),
        scale=5 * (u1 * 0.3 + 0.5 * u2),
        size=num_samples,
    )
    r = generator.normal(loc=w * 0.7, scale=6, size=num_samples)
    if treatment_assignment:
        x = np.full(num_samples, treatment_assignment)
    else:
        x = generator.binomial(n=1, p=1 / (1 + np.exp(-2 - 0.23 * u1 - 0.1 * r)), size=num_samples)
    y = generator.normal(loc=u2 * 0.5 + x * 3, scale=6)
    return pd.DataFrame({"W": w, "R": r, "X": x, "Y": y})


def main():
    seed = 1
    num_samples = 1000

    # Compute the real ACE value
    intv_data_X1 = generate_obs_data_for_napkin(
        seed=seed, num_samples=num_samples, treatment_assignment=1
    )
    intv_data_X0 = generate_obs_data_for_napkin(
        seed=seed, num_samples=num_samples, treatment_assignment=0
    )
    real_ace = np.mean(intv_data_X1["Y"]) - np.mean(intv_data_X0["Y"])

    # Compute the ACE estimated with ananke
    obs_data = generate_obs_data_for_napkin(seed=seed, num_samples=num_samples)
    ace_obj_2 = CausalEffect(graph=napkin.to_admg(), treatment="X", outcome="Y")
    ace_anipw = ace_obj_2.compute_effect(obs_data, "anipw")


if __name__ == "__main__":
    main()
