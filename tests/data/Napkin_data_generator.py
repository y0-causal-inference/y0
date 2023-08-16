import numpy as np
import pandas as pd
from ananke.estimation import CausalEffect
from ananke.graphs import ADMG


# Generate observational data for napkin
def generate_obs_data_for_napkin(seed, num_samples, treatment_assignment=None):
    generator = np.random.default_rng(seed)
    # U1 is the latent variable that is a common cause of W and X
    U1 = generator.normal(loc=3, scale=1, size=num_samples)
    # U2 is the latent variable that is a common cause of W and Y
    U2 = generator.normal(loc=5, scale=1, size=num_samples)
    W = generator.gamma(
        shape=1 / (1 * (U1 * 0.3 + 0.5 * U2) ** 2),
        scale=5 * (U1 * 0.3 + 0.5 * U2),
        size=num_samples,
    )
    R = generator.normal(loc=W * 0.7, scale=6, size=num_samples)
    if treatment_assignment:
        X = np.full(num_samples, treatment_assignment)
    else:
        X = generator.binomial(n=1, p=1 / (1 + np.exp(-2 - 0.23 * U1 - 0.1 * R)), size=num_samples)
    Y = generator.normal(loc=U2 * 0.5 + X * 3, scale=6)
    return pd.DataFrame({"W": W, "R": R, "X": X, "Y": Y})


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
    # napkin model
    vertices = ["W", "R", "X", "Y"]
    di_edges = [("W", "R"), ("R", "X"), ("X", "Y")]
    bi_edges = [("W", "X"), ("W", "Y")]
    graph_napkin = ADMG(vertices, di_edges, bi_edges)
    graph_napkin.draw(direction="LR")

    obs_data = generate_obs_data_for_napkin(seed=seed, num_samples=num_samples)
    ace_obj_2 = CausalEffect(graph=graph_napkin, treatment="X", outcome="Y")
    ace_anipw = ace_obj_2.compute_effect(obs_data, "anipw")


if __name__ == "__main__":
    main()
