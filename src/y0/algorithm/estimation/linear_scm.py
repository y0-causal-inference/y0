"""Utilities for structural causal models (SCMs)."""

from collections.abc import Iterable
from statistics import fmean
from typing import cast

import pandas as pd
import sympy
from sklearn.linear_model import LinearRegression

from y0.algorithm.separation.sigma_single_door import find_sigma_single_door_sets
from y0.dsl import Variable
from y0.graph import NxMixedGraph, sympy_nested

__all__ = [
    "evaluate_admg",
    "evaluate_lscm",
    "get_sigma_single_door",
    "get_single_door",
]

EvalRv = dict[sympy.Symbol, sympy.core.numbers.Rational]


def _estimate_edge(
    data: pd.DataFrame,
    source: Variable,
    target: Variable,
    adjustment_sets: Iterable[frozenset[str]],
) -> float:
    """Estimate the direct-effect coefficient for source → target via OLS.

    For each adjustment set Z, regresses *target* on {source} ∪ Z using
    :class:`sklearn.linear_model.LinearRegression` and extracts the
    coefficient for *source*.  Returns the mean across all adjustment sets.

    :param data: Observed data; columns must include source.name, target.name,
        and all variable names appearing in any adjustment set.
    :param source: The cause variable.
    :param target: The effect variable.
    :param adjustment_sets: An iterable of adjustment sets, each a frozenset of
        *string* column names (matching ``data`` columns).
    :returns: The mean OLS coefficient for *source* across all adjustment sets.
    """
    coefficients = []
    for adjustment_set in adjustment_sets:
        variables = sorted(adjustment_set | {source.name})
        idx = variables.index(source.name)
        model = LinearRegression()
        model.fit(data[variables], data[target.name])
        coefficients.append(model.coef_[idx])
    return fmean(coefficients)


def get_single_door(
    graph: NxMixedGraph, data: pd.DataFrame
) -> dict[tuple[Variable, Variable], float]:
    """Estimate parameter values for a linear SCM using backdoor adjustment.

    Restricted to **DAGs** (uses pgmpy's d-separation back-door enumeration).
    For directed graphs with cycles, use :func:`get_sigma_single_door`.

    :param graph: An acyclic directed mixed graph (ADMG / DAG).
    :param data: Observed data; columns must match node names in *graph*.
    :returns: A mapping from each directed edge (source, target) to the
        estimated linear coefficient β(source → target).
    """
    inference = graph.to_pgmpy_causal_inference()
    rv = {}
    for source, target in graph.directed.edges():
        try:
            adjustment_sets = inference.get_all_backdoor_adjustment_sets(source.name, target.name)
        except ValueError:
            # There are no valid adjustment sets.
            continue
        if not adjustment_sets:
            # There is a valid adjustment set, and it is the empty set, so just
            # regress the target on the source.
            adjustment_sets = frozenset([frozenset([])])
        rv[source, target] = _estimate_edge(data, source, target, adjustment_sets)
    return rv


def get_sigma_single_door(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    *,
    max_conditions: int | None = None,
) -> dict[tuple[Variable, Variable], float]:
    """Estimate linear-SCM coefficients via the σ-separation single-door criterion.

    Cyclic generalization of :func:`get_single_door`: replaces d-separation
    back-door adjustment (DAG-only, via pgmpy) with σ-separation admissible
    sets (valid for directed graphs with cycles, per Forré & Mooij 2018).

    For each directed edge source → target, enumerates σ-separation admissible
    adjustment sets Z using :func:`~y0.algorithm.separation.sigma_single_door.\
find_sigma_single_door_sets`, regresses *target* on {source} ∪ Z, and
    averages the source coefficient across all admissible sets.  Edges for
    which no admissible set exists are **omitted** from the result (they are
    not identifiable via the single-door criterion).

    :param graph: A directed mixed graph (may contain cycles).
    :param data: Observed data; columns must match node names in *graph*.
    :param max_conditions: Maximum size of adjustment sets to consider.
        ``None`` (default) means unbounded enumeration over non-descendants.
        **Warning:** enumeration is O(2^k) in the candidate pool size; set a
        cap for large graphs.
    :returns: A mapping from each identifiable directed edge (source, target)
        to the estimated linear coefficient β(source → target).

    .. note::
        On acyclic graphs, results should numerically agree with
        :func:`get_single_door` (up to sampling noise), since σ-separation
        reduces to d-separation on DAGs.

    References
    ----------
    P. Forré, J. M. Mooij (2018). Constraint-based Causal Discovery for
    Non-linear Structural Causal Models with Cycles and Latent Confounders.
    """
    rv = {}
    for source, target in graph.directed.edges():
        # Collect all σ-admissible adjustment sets (as frozensets of Variable)
        admissible_variable_sets = list(
            find_sigma_single_door_sets(graph, source, target, max_conditions=max_conditions)
        )
        if not admissible_variable_sets:
            # Edge is not identifiable via the single-door criterion — skip it.
            continue
        # Convert Variable frozensets → string frozensets for _estimate_edge
        admissible_str_sets: list[frozenset[str]] = [
            frozenset(v.name for v in z) for z in admissible_variable_sets
        ]
        rv[source, target] = _estimate_edge(data, source, target, admissible_str_sets)
    return rv


def evaluate_admg(graph: NxMixedGraph, data: pd.DataFrame) -> EvalRv:
    """Evaluate an acyclic directed mixed graph (ADMG)."""
    params = {
        sympy_nested("\\beta", source, target): mean
        for (source, target), mean in get_single_door(graph, data).items()
    }
    lscm = graph.to_linear_scm_sympy()
    return evaluate_lscm(lscm, params)


def evaluate_lscm(
    linear_scm: dict[Variable, sympy.Expr], params: dict[sympy.Symbol, float]
) -> EvalRv:
    """Assign values to the parameters and return variable assignments dictionary."""
    expressions: dict[sympy.Symbol, sympy.Expr] = {
        variable.to_sympy(): expression for variable, expression in linear_scm.items()
    }
    eqns = [sympy.Eq(lhs.subs(params), rhs.subs(params)) for lhs, rhs in expressions.items()]
    return cast(EvalRv, sympy.solve(eqns, list(expressions)))


def _main() -> None:
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
