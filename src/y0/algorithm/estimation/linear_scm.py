"""Utilities for structural causal models (SCMs)."""

from statistics import fmean

import pandas as pd
import sympy
from sklearn.linear_model import LinearRegression

from y0.dsl import Variable
from y0.graph import NxMixedGraph, sympy_nested

__all__ = [
    "get_single_door",
    "evaluate_admg",
    "evaluate_lscm",
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
            # There are no valid adjustment sets.
            continue
        if not adjustment_sets:
            # There is a valid adjustment set, and it is the empty set, so just regress the target on the source.
            adjustment_sets = frozenset([frozenset([])])
        coefficients = []
        for adjustment_set in adjustment_sets:
            variables = sorted(adjustment_set | {source.name})
            idx = variables.index(source.name)
            model = LinearRegression()
            model.fit(data[variables], data[target.name])
            coefficients.append(model.coef_[idx])
        rv[source, target] = fmean(coefficients)
    return rv


def evaluate_admg(graph, data: pd.DataFrame):
    """Evaluate an acyclic directed mixed graph (ADMG)."""
    params = {sympy_nested("\\beta", l, r): v for (l, r), v in get_single_door(graph, data).items()}
    lscm = graph.to_linear_scm_sympy()
    return evaluate_lscm(lscm, params)


def evaluate_lscm(
    linear_scm: dict[Variable, sympy.Expr], params: dict[sympy.Symbol, float]
) -> dict[sympy.Symbol, sympy.core.numbers.Rational]:
    """Assign values to the parameters and return variable assignments dictionary."""
    expressions: dict[sympy.Symbol, sympy.Expr] = {
        variable.to_sympy(): expression for variable, expression in linear_scm.items()
    }
    eqns = [sympy.Eq(lhs.subs(params), rhs.subs(params)) for lhs, rhs in expressions.items()]
    return sympy.solve(eqns, list(expressions))


def _main():
    import warnings

    from y0.examples import examples

    warnings.filterwarnings("ignore")

    for example in examples:
        if example.generate_data is None:
            continue
        data = example.generate_data(500)
        rv = get_single_door(example.graph, data)
        print(example.name, rv)  # noqa:T201

        s = evaluate_admg(example.graph, data)
        print(s)  # noqa:T201


if __name__ == "__main__":
    _main()
