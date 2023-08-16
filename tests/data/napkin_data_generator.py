import warnings
from contextlib import redirect_stdout
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ananke.estimation import CausalEffect
from tqdm import trange

from y0.examples import napkin

warnings.simplefilter(action="ignore", category=FutureWarning)

HERE = Path(__file__).parent.resolve()
DELTAS_PATH = HERE.joinpath("deltas.png")


# Generate observational data for napkin
def generate_napkin_data(
    *, seed: int | None = None, num_samples: int, treatment_assignment: int | None = None
) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    # U1 is the latent variable that is a common cause of W and X
    u1 = generator.normal(loc=3, scale=1, size=num_samples)
    # U2 is the latent variable that is a common cause of W and Y
    u2 = generator.normal(loc=5, scale=1, size=num_samples)
    z2 = generator.gamma(
        shape=1 / (1 * (u1 * 0.3 + 0.5 * u2) ** 2),
        scale=5 * (u1 * 0.3 + 0.5 * u2),
        size=num_samples,
    )
    z1 = generator.normal(loc=z2 * 0.7, scale=6, size=num_samples)
    if treatment_assignment is not None:
        x = np.full(num_samples, treatment_assignment)
    else:
        x = generator.binomial(n=1, p=1 / (1 + np.exp(-2 - 0.23 * u1 - 0.1 * z1)), size=num_samples)
    y = generator.normal(loc=u2 * 0.5 + x * 3, scale=6)
    return pd.DataFrame({"Z2": z2, "Z1": z1, "X": x, "Y": y})


@click.command()
def main():
    seed = 1
    num_samples = 1000

    # Compute the real ACE value
    df_treat_x_1 = generate_napkin_data(seed=seed, num_samples=num_samples, treatment_assignment=1)
    df_treat_x_1.to_csv(HERE.joinpath("napkin_treat_x_1.tsv"), sep="\t", index=False)
    df_treat_x_0 = generate_napkin_data(seed=seed, num_samples=num_samples, treatment_assignment=0)
    df_treat_x_0.to_csv(HERE.joinpath("napkin_treat_x_0.tsv"), sep="\t", index=False)
    real_ace = np.mean(df_treat_x_1["Y"]) - np.mean(df_treat_x_0["Y"])

    with redirect_stdout(None):
        ace_obj_2 = CausalEffect(graph=napkin.to_admg(), treatment="X", outcome="Y")

    df = generate_napkin_data(num_samples=num_samples)
    df.to_csv(HERE.joinpath("napkin_observational.tsv"), sep="\t", index=False)
    # TODO fix output
    # grid = sns.pairplot(data=df, aspect=1.65, height=1.2, plot_kws=dict(alpha=0.7))
    # grid.savefig(HERE.joinpath("napkin_observational.png"), dpi=300)

    deltas = []
    for _ in trange(500):
        df = generate_napkin_data(num_samples=num_samples)
        ace_anipw = ace_obj_2.compute_effect(df, "anipw")
        delta = ace_anipw - real_ace
        deltas.append(delta)

    fig, ax = plt.subplots(figsize=(4, 2.3))
    sns.histplot(deltas, ax=ax)
    ax.set_title("Deviation from actual ACE")
    fig.savefig(DELTAS_PATH, dpi=300)
    click.echo(f"output histogram to {DELTAS_PATH}")


if __name__ == "__main__":
    main()
