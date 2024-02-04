# -*- coding: utf-8 -*-

"""An implementation to get conditional independencies of an ADMG from [pearl2009]_."""

from functools import partial
from itertools import combinations, groupby
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx
import pandas as pd
from tqdm.auto import tqdm

from ..dsl import Variable
from ..graph import NxMixedGraph
from ..struct import (
    DEFAULT_SIGNIFICANCE,
    CITest,
    CITestTuple,
    DSeparationJudgement,
    _ensure_method,
)
from ..util.combinatorics import powerset

__all__ = [
    "are_d_separated",
    "minimal",
    "get_conditional_independencies",
    "test_conditional_independencies",
    "add_ci_undirected_edges",
]


def add_ci_undirected_edges(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    *,
    method: Optional[CITest] = None,
    significance_level: Optional[float] = None,
) -> NxMixedGraph:
    """Add undirected edges between d-separated nodes that fail a data-driven conditional independency test.

    Inspired by [taheri2024]_.

    :param graph: An acyclic directed mixed graph
    :param data: observational data corresponding to the graph
    :param method:
        The conditional independency test to use. If None, defaults to
        :data:`y0.struct.DEFAULT_CONTINUOUS_CI_TEST` for continuous data
        or :data:`y0.struct.DEFAULT_DISCRETE_CI_TEST` for discrete data.
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.05.
    :returns: A copy of the input graph potentially with new undirected edges added
    """
    rv = graph.copy()
    for judgement, result in test_conditional_independencies(
        graph=graph, data=data, method=method, boolean=True, significance_level=significance_level
    ):
        if not result:
            rv.add_undirected_edge(judgement.left, judgement.right)
    return rv


def test_conditional_independencies(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    *,
    method: Optional[CITest] = None,
    boolean: bool = False,
    significance_level: Optional[float] = None,
    _method_checked: bool = False,
) -> List[Tuple[DSeparationJudgement, Union[bool, CITestTuple]]]:
    """Gets CIs with :func:`get_conditional_independencies` then tests them against data.

    :param graph: An acyclic directed mixed graph
    :param data: observational data corresponding to the graph
    :param method:
        The conditional independency test to use. If None, defaults to
        :data:`y0.struct.DEFAULT_CONTINUOUS_CI_TEST` for continuous data
        or :data:`y0.struct.DEFAULT_DISCRETE_CI_TEST` for discrete data.
    :param boolean:
        If set to true, switches the test return type to be a pre-computed
        boolean based on the significance level (see parameter below)
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.05.
    :returns: A copy of the input graph potentially with new undirected edges added
    """
    if significance_level is None:
        significance_level = DEFAULT_SIGNIFICANCE
    method = _ensure_method(method, data, skip=_method_checked)
    return [
        (
            judgement,
            judgement.test(
                data,
                boolean=boolean,
                method=method,
                significance_level=significance_level,
                _method_checked=True,
            ),
        )
        for judgement in get_conditional_independencies(graph)
    ]


def get_conditional_independencies(
    graph: NxMixedGraph,
    *,
    policy=None,
    **kwargs,
) -> Set[DSeparationJudgement]:
    """Get the conditional independencies from the given ADMG.

    Conditional independencies is the minimal set of d-separation judgements to cover
    the unique left/right combinations in all valid d-separation.

    :param graph: An acyclic directed mixed graph
    :param policy: Retention policy when more than one conditional independency option exists (see minimal for details)
    :param kwargs: Other keyword arguments are passed to :func:`d_separations`
    :return: A set of conditional dependencies

    .. seealso:: Original issue https://github.com/y0-causal-inference/y0/issues/24
    """
    if policy is None:
        policy = get_topological_policy(graph)
    return minimal(
        d_separations(graph, **kwargs),
        policy=policy,
    )


def minimal(judgements: Iterable[DSeparationJudgement], policy=None) -> Set[DSeparationJudgement]:
    r"""Given some d-separations, reduces to a 'minimal' collection.

    For independencies of the form $A \perp B | {C_1, C_2, ...}$, the minimal collection will

    - Have only one independency with the same A/B nodes.
    - If there are multiples sets of C-nodes, the kept d-separation will be the first/minimal
      element in the group sorted according to `policy` argument.

    The default policy is to sort by the shortest set of conditions & then lexicographic.

    :param judgements: Collection of judgements to minimize
    :param policy: Function from d-separation to a representation suitable for sorting.
    :return: A set of judgements that is minimal (as described above)
    """
    if policy is None:
        policy = _len_lex
    judgements = sorted(judgements, key=_judgement_grouper)
    return {min(vs, key=policy) for k, vs in groupby(judgements, _judgement_grouper)}


def get_topological_policy(
    graph: NxMixedGraph,
) -> Callable[[DSeparationJudgement], Tuple[int, int]]:
    """Sort d-separations by condition length and topological order.

    This policy prefers small collections, and collections with variables earlier
    in topological order for collections of the same size.

    :param graph: a mixed graph
    :return: A function suitable for use as a sort key on d-separations
    """
    order = list(graph.topological_sort())
    return partial(_topological_policy, order=order)


def _topological_policy(
    judgement: DSeparationJudgement, order: Sequence[Variable]
) -> Tuple[int, int]:
    return (
        len(judgement.conditions),
        sum((order.index(v) for v in judgement.conditions)),
    )


def _judgement_grouper(judgement: DSeparationJudgement) -> Tuple[Variable, Variable]:
    """Simplify d-separation to just left & right element (for grouping left/right pairs)."""
    return judgement.left, judgement.right


def _len_lex(judgement: DSeparationJudgement) -> Tuple[int, str]:
    """Sort by length of conditions & the lexicography a d-separation."""
    return len(judgement.conditions), ",".join(c.name for c in judgement.conditions)


def are_d_separated(
    graph: NxMixedGraph,
    a: Variable,
    b: Variable,
    *,
    conditions: Optional[Iterable[Variable]] = None,
) -> DSeparationJudgement:
    """Test if nodes named by a & b are d-separated in G as described in [pearl2009]_.

    a & b can be provided in either order and the order of conditions does not matter.
    However, DSeparationJudgement may put things in canonical order.

    :param graph: Graph to test
    :param a: A node in the graph
    :param b: A node in the graph
    :param conditions: A collection of graph nodes
    :return: T/F and the final graph (as evidence)
    :raises TypeError: if the left/right arguments or any conditions are
        not Variable instances
    :raises KeyError: if the left/right arguments or any conditions are
        not in the graph

    .. seealso:: NetworkX implementation :func:`nx.d_separated`
    """
    if conditions is None:
        conditions = set()
    conditions = set(conditions)
    if not isinstance(a, Variable):
        raise TypeError(f"left argument is not given as a Variable: {type(a)}: {a}")
    if not isinstance(b, Variable):
        raise TypeError(f"right argument is not given as a Variable: {type(b)}: {b}")
    if not all(isinstance(c, Variable) for c in conditions):
        raise TypeError(f"some conditions are not variables: {conditions}")
    if a not in graph:
        raise KeyError(f"left argument is not in graph: {a}")
    if b not in graph:
        raise KeyError(f"right argument is not in graph: {b}")
    missing_conditions = {condition for condition in conditions if condition not in graph}
    if missing_conditions:
        raise KeyError(f"conditions missing from graph: {missing_conditions}")

    named = {a, b}.union(conditions)

    # Filter to ancestors
    keep = graph.ancestors_inclusive(named)
    evidence_graph = graph.subgraph(keep).moralize().disorient()

    keep = set(evidence_graph.nodes) - set(conditions)
    evidence_graph = evidence_graph.subgraph(keep)

    # check for path....
    separated = not nx.has_path(evidence_graph, a, b)  # If no path, then d-separated!

    return DSeparationJudgement.create(left=a, right=b, conditions=conditions, separated=separated)


def d_separations(
    graph: NxMixedGraph,
    *,
    max_conditions: Optional[int] = None,
    verbose: Optional[bool] = False,
    return_all: Optional[bool] = False,
) -> Iterable[DSeparationJudgement]:
    """Generate d-separations in the provided graph.

    :param graph: Graph to search for d-separations.
    :param max_conditions: Longest set of conditions to investigate
    :param return_all: If false (default) only returns the first d-separation per left/right pair.
    :param verbose: If true, prints extra output with tqdm
    :yields: True d-separation judgements
    """
    vertices = set(graph.nodes())
    for a, b in tqdm(
        combinations(vertices, 2),
        disable=not verbose,
        desc="Checking d-separations",
        unit="pair",
        total=len(vertices) * (len(vertices) - 1) // 2,
    ):
        for conditions in powerset(vertices - {a, b}, stop=max_conditions):
            judgement = are_d_separated(graph, a, b, conditions=conditions)
            if judgement.separated:
                yield judgement
                if not return_all:
                    break
