"""Utilities for structural causal models (SCMs)."""

from statistics import fmean

import pandas as pd
from sklearn.linear_model import LinearRegression

from y0.dsl import Variable
from y0.graph import NxMixedGraph

__all__ = [
    "get_single_door",
]


def get_single_door(
    graph: NxMixedGraph, data: pd.DataFrame
) -> dict[tuple[Variable, Variable], float]:
    """Estimate parameter values for a linear SCM using backdoor adjustment."""
    inference = graph.to_pgmpy_causal_inference()
    rv = {}
    for source, target in graph.directed.edges():
        try:
            adjustment_sets = inference.get_all_backdoor_adjustment_sets(source.name, target.name)
        except ValueError:
            continue
        if not adjustment_sets:
            continue
        coefficients = []
        for adjustment_set in adjustment_sets:
            variables = sorted(adjustment_set | {source.name})
            idx = variables.index(source.name)
            model = LinearRegression()
            model.fit(data[variables], data[target.name])
            coefficients.append(model.coef_[idx])
        rv[source, target] = fmean(coefficients)
    return rv


def _main():
    from y0.examples import examples

    for example in examples:
        if example.generate_data is None:
            continue
        data = example.generate_data(500)
        rv = get_single_door(example.graph, data)
        print(example.name, rv)  # noqa:T201


if __name__ == "__main__":
    _main()
