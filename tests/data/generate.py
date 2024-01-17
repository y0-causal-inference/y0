"""Generate example data.

This automatically works for all instances of :clas:`y0.examples.Example`
appearing in :mod:`y0.examples`. If you make a new generator, make sure
you annotate it to the correct graph and add a corresponding example
query.

This file can be run as a script directly with ``python generate.py``.
"""

import warnings
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate
from tqdm import tqdm, trange

from y0.algorithm.estimation import estimate_ace
from y0.examples import examples

warnings.simplefilter(action="ignore", category=FutureWarning)


HERE = Path(__file__).parent.resolve()


def main(seed: int = 1, num_samples: int = 1000, bootstraps: int = 500):
    """Generate example data."""
    aces = []
    for example in examples:
        if example.generate_data is None:
            continue
        directory = HERE.joinpath(example.name.lower().replace(" ", "-").replace("_", "-"))
        directory.mkdir(exist_ok=True, parents=True)

        df = example.generate_data(num_samples)
        df.to_csv(directory.joinpath("observational.tsv"), sep="\t", index=False)
        # TODO fix output
        # grid = sns.pairplot(data=df, aspect=1.65, height=1.2, plot_kws=dict(alpha=0.7))
        # grid.savefig(HERE.joinpath("napkin_observational.png"), dpi=300)

        if not example.example_queries:
            raise ValueError(f"{example.name} should have at least one example query")

        queries = [q for q in example.example_queries if 1 == len(q.treatments) == len(q.outcomes)]
        if not queries:
            raise ValueError(
                f"{example.name} should have at least one example query with a single treatment and outcome"
            )

        query = queries[0]
        treatment = list(query.treatments)[0]
        outcome = list(query.outcomes)[0]
        treatment_name = treatment.name.lower().replace("-", "").replace("_", "")

        df_treat_1 = example.generate_data(num_samples, seed=seed, treatments={treatment: 1})
        df_treat_1.to_csv(
            directory.joinpath(f"treat_{treatment_name}_1.tsv"), sep="\t", index=False
        )
        df_treat_0 = example.generate_data(num_samples, seed=seed, treatments={treatment: 0})
        df_treat_0.to_csv(
            directory.joinpath(f"treat_{treatment_name}_0.tsv"), sep="\t", index=False
        )
        actual_ace = np.mean(df_treat_1[outcome.name]) - np.mean(df_treat_0[outcome.name])
        aces.append((example.name, actual_ace))

        ace_deltas = []
        for _ in trange(bootstraps, desc=f"ACE {example.name}"):
            df = example.generate_data(num_samples)
            estimated_ace = estimate_ace(
                graph=example.graph,
                treatments=treatment,
                outcomes=outcome,
                data=df,
            )
            delta = estimated_ace - actual_ace
            ace_deltas.append(delta)

        fig, ax = plt.subplots(figsize=(4, 2.3))
        sns.histplot(ace_deltas, ax=ax)
        ax.set_title("Deviation from actual ACE")
        deviation_path = directory.joinpath("deltas.png")
        fig.savefig(deviation_path, dpi=300)
        tqdm.write(f"Wrote {example.name} deviations chart to {deviation_path}")

    click.echo(tabulate(aces, headers=["Name", "ACE"], tablefmt="github"))


if __name__ == "__main__":
    main()
