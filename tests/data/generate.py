"""Generate example data."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange

from y0.algorithm.estimation import estimate_ate
from y0.dsl import X
from y0.examples import examples

warnings.simplefilter(action="ignore", category=FutureWarning)


HERE = Path(__file__).parent.resolve()


def main(seed: int = 1, num_samples: int = 1000, bootstraps: int = 500):
    """Generate example data."""
    for example in examples:
        if example.generate_data is None:
            continue
        directory = HERE.joinpath(example.name.lower())
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

        df_treat_x_1 = example.generate_data(num_samples, seed=seed, treatments={X: 1})
        df_treat_x_1.to_csv(directory.joinpath("treat_x_1.tsv"), sep="\t", index=False)
        df_treat_x_0 = example.generate_data(num_samples, seed=seed, treatments={X: 0})
        df_treat_x_0.to_csv(directory.joinpath("treat_x_0.tsv"), sep="\t", index=False)
        actual_ace = np.mean(df_treat_x_1[outcome.name]) - np.mean(df_treat_x_0[outcome.name])

        ace_deltas = []
        for _ in trange(bootstraps, desc=f"ACE {example.name}"):
            df = example.generate_data(num_samples)
            estimated_ace = estimate_ate(
                graph=example.graph,
                treatment=treatment,
                outcome=outcome,
                data=df,
            )
            delta = estimated_ace - actual_ace
            ace_deltas.append(delta)

        fig, ax = plt.subplots(figsize=(4, 2.3))
        sns.histplot(ace_deltas, ax=ax)
        ax.set_title("Deviation from actual ACE")
        fig.savefig(directory.joinpath("deltas.png"), dpi=300)


if __name__ == "__main__":
    main()
