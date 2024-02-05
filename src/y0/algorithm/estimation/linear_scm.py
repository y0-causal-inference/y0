"""Utilities for structural causal models (SCMs)."""

from statistics import fmean

import pandas as pd
from sklearn.linear_model import LinearRegression

from y0.dsl import Variable
from y0.graph import NxMixedGraph
import networkx as nx
import sympy as sy

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


def evaluate_admg(graph, data: pd.DataFrame):
    params = {_get_beta(l, r): v for (l, r), v in get_single_door(graph, data).items()}
    # do something
    lscm = generate_lscm_from_mixed_graph(graph)
    return evaluate_lscm(lscm, params)


def evaluate_lscm(
    LSCM: dict[sy.Symbol, sy.Expr], params: dict[sy.Symbol, float]
) -> dict[sy.Symbol, sy.core.numbers.Rational]:
    """given an LSCM, assign values to the parameters (i.e. beta, epsilon, gamma terms), and return variable assignments dictionary"""
    # solve set of simulateous linear equations in sympy
    eqns = [sy.Eq(lhs.subs(params), rhs.subs(params)) for lhs, rhs in LSCM.items()]
    print(eqns)
    return sy.solve(eqns, list(LSCM))


def _get_beta(left: Variable, right: Variable) -> sy.Symbol:
    return sy.Symbol(f"beta_{left.name}_->{right.name}")


def generate_lscm_from_mixed_graph(graph: NxMixedGraph) -> dict:
    equations = {}
    for node in graph.topological_sort():
        node_sym = sy.Symbol(node.name)  # fix name prop
        expression_terms = []

        # Add parent edges
        for parent in graph.directed.predecessors(node):
            expression_terms.append(_get_beta(parent, node) * sy.Symbol(f"{parent.name}"))

        # Add noise term
        epsilon_sym = sy.Symbol(f"epsilon_{node.name}")
        expression_terms.append(epsilon_sym)

        # get bidirected edges
        for u, v in graph.undirected.edges(node):
            u, v = sorted([u, v])
            temp_gamma_sym = sy.Symbol(f"gamma_{u}_<->{v}")
            expression_terms.append(temp_gamma_sym)

        equations[node_sym] = sum(expression_terms)
    return equations


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
