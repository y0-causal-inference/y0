import warnings
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ananke.estimation import CausalEffect
from tqdm import trange

from y0.algorithm.identify import Query
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

        if example.example_queries:
            query: Query = example.example_queries[0]
            if len(query.treatments) != 1 or len(query.outcomes) != 1:
                raise NotImplementedError

            treatment = list(query.treatments)[0]
            outcome = list(query.outcomes)[0]

            df_treat_1 = example.generate_data(num_samples, seed=seed, treatment_assignment=1)
            df_treat_1.to_csv(
                directory.joinpath(f"treat_{treatment.name}_1.tsv"), sep="\t", index=False
            )
            df_treat_0 = example.generate_data(num_samples, seed=seed, treatment_assignment=0)
            df_treat_0.to_csv(
                directory.joinpath(f"treat_{treatment.name}_0.tsv"), sep="\t", index=False
            )

            with redirect_stdout(None):
                ace_obj = CausalEffect(
                    graph=example.graph.to_admg(), treatment=treatment.name, outcome=outcome.name
                )

            actual_ace = np.mean(df_treat_1[outcome.name]) - np.mean(df_treat_0[outcome.name])

            ace_deltas = []
            for _ in trange(bootstraps, desc=f"ATE {example.name}"):
                df = example.generate_data(num_samples)
                estimated_ace = ace_obj.compute_effect(df, "anipw")
                delta = estimated_ace - actual_ace
                ace_deltas.append(delta)

            fig, ax = plt.subplots(figsize=(4, 2.3))
            sns.histplot(ace_deltas, ax=ax)
            ax.set_title("Deviation from actual ACE")
            fig.savefig(directory.joinpath("deltas.png"), dpi=300)


if __name__ == "__main__":
    main()
