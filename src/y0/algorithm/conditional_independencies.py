"""An implementation to get conditional independencies of an ADMG from [pearl2009]_."""

from collections.abc import Callable, Collection, Iterable, Sequence
from functools import partial
from itertools import combinations, groupby
from typing import Any, Literal, NamedTuple, Self, TypeAlias, overload

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
    "Policy",
    "add_ci_undirected_edges",
    "are_d_separated",
    "get_conditional_independencies",
    "minimal",
    "test_conditional_independencies",
]


def add_ci_undirected_edges(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    *,
    method: CITest | None = None,
    significance_level: float | None = None,
    max_conditions: int | None = None,
) -> NxMixedGraph:
    """Add undirected edges between d-separated nodes that fail a data-driven conditional independency test.

    Inspired by [taheri2024]_.

    :param graph: An acyclic directed mixed graph
    :param data: observational data corresponding to the graph
    :param method: The conditional independency test to use. If None, defaults to
        :data:`y0.struct.DEFAULT_CONTINUOUS_CI_TEST` for continuous data or
        :data:`y0.struct.DEFAULT_DISCRETE_CI_TEST` for discrete data.
    :param significance_level: The statistical tests employ this value for comparison
        with the p-value of the test to determine the independence of the tested
        variables. If none, defaults to 0.05.
    :param max_conditions: Longest set of conditions to investigate

    :returns: A copy of the input graph potentially with new undirected edges added
    """
    rv = graph.copy()
    for judgement, result in test_conditional_independencies(
        graph=graph,
        data=data,
        method=method,
        boolean=True,
        significance_level=significance_level,
        max_conditions=max_conditions,
    ):
        if not result:
            rv.add_undirected_edge(judgement.left, judgement.right)
    return rv


# docstr-coverage:excused `overload`
@overload
def test_conditional_independencies(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    *,
    method: CITest | None = ...,
    boolean: Literal[True] = ...,
    significance_level: float | None = ...,
    _method_checked: bool = ...,
    max_conditions: int | None = ...,
) -> list[tuple[DSeparationJudgement, bool]]: ...


# docstr-coverage:excused `overload`
@overload
def test_conditional_independencies(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    *,
    method: CITest | None = ...,
    boolean: Literal[False] = ...,
    significance_level: float | None = ...,
    _method_checked: bool = ...,
    max_conditions: int | None = ...,
) -> list[tuple[DSeparationJudgement, CITestTuple]]: ...


def test_conditional_independencies(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    *,
    method: CITest | None = None,
    boolean: bool = False,
    significance_level: float | None = None,
    _method_checked: bool = False,
    max_conditions: int | None = None,
) -> list[tuple[DSeparationJudgement, CITestTuple]] | list[tuple[DSeparationJudgement, bool]]:
    """Gets CIs with :func:`get_conditional_independencies` then tests them against data.

    :param graph: An acyclic directed mixed graph
    :param data: observational data corresponding to the graph
    :param method: The conditional independency test to use. If None, defaults to
        :data:`y0.struct.DEFAULT_CONTINUOUS_CI_TEST` for continuous data or
        :data:`y0.struct.DEFAULT_DISCRETE_CI_TEST` for discrete data.
    :param boolean: If set to true, switches the test return type to be a pre-computed
        boolean based on the significance level (see parameter below)
    :param significance_level: The statistical tests employ this value for comparison
        with the p-value of the test to determine the independence of the tested
        variables. If none, defaults to 0.05.
    :param max_conditions: Longest set of conditions to investigate

    :returns: A copy of the input graph potentially with new undirected edges added
    """
    if significance_level is None:
        significance_level = DEFAULT_SIGNIFICANCE
    method = _ensure_method(method, data, skip=_method_checked)
    return [
        (
            judgement,
            judgement.test(  # type:ignore[call-overload]
                data,
                boolean=boolean,
                method=method,
                significance_level=significance_level,
                _method_checked=True,
            ),
        )
        for judgement in get_conditional_independencies(graph, max_conditions=max_conditions)
    ]


#: A policy for reducing conditional independencies
Policy: TypeAlias = Callable[[DSeparationJudgement], Any]


class PairSeparationContext(NamedTuple):
    """Pair-specific graph state reused across conditioning sets."""

    evidence_graph: nx.Graph
    condition_candidates: Sequence[Variable]

    @classmethod
    def prepare(
        cls,
        graph: NxMixedGraph,
        a: Variable,
        b: Variable,
    ) -> Self:
        named = {a, b}
        keep = graph.ancestors_inclusive(named)
        evidence_graph = graph.subgraph(keep).moralize().disorient()
        condition_candidates = _order_condition_candidates(
            evidence_graph,
            a,
            b,
            candidates=set(evidence_graph.nodes) - named,
        )
        return cls(
            evidence_graph=evidence_graph,
            condition_candidates=condition_candidates,
        )

    def is_d_separated(
        self,
        a: Variable,
        b: Variable,
        *,
        conditions: Collection[Variable] | None = None,
    ) -> bool:
        """Check if the variables are d-separated."""
        if conditions:
            evidence_graph = self.evidence_graph.subgraph(
                set(self.evidence_graph.nodes).difference(conditions)
            )
        else:
            evidence_graph = self.evidence_graph
        return not nx.has_path(evidence_graph, a, b)


def _order_condition_candidates(
    evidence_graph: nx.Graph,
    a: Variable,
    b: Variable,
    *,
    candidates: Iterable[Variable],
) -> Sequence[Variable]:
    """Prioritize nodes closer to the queried pair for earlier separator discovery."""
    a_distances = nx.single_source_shortest_path_length(evidence_graph, a)
    b_distances = nx.single_source_shortest_path_length(evidence_graph, b)
    condition_candidates = sorted(candidates, key=_get_key(a_distances, b_distances))
    return condition_candidates


def _get_key(a_distances, b_distances) -> Callable[[Variable], tuple[float, float, float, str]]:
    def key(node: Variable) -> tuple[float, float, float, str]:
        """Rank condition candidates by closeness to the queried pair."""
        a_distance = a_distances.get(node, float("inf"))
        b_distance = b_distances.get(node, float("inf"))
        # TODO what is the meaning of this key?
        return (a_distance + b_distance, min(a_distance, b_distance), a_distance, str(node))

    return key


def get_conditional_independencies(
    graph: NxMixedGraph,
    *,
    policy: Policy | None = None,
    max_conditions: int | None = None,
    return_all: bool | None = False,
    verbose: bool = False,
) -> set[DSeparationJudgement]:
    """Get the conditional independencies from the given ADMG.

    Conditional independencies is the minimal set of d-separation judgements to cover
    the unique left/right combinations in all valid d-separation.

    :param graph: An acyclic directed mixed graph
    :param policy: Retention policy when more than one conditional independency option
        exists (see minimal for details)
    :param max_conditions: Longest set of conditions to investigate
    :param return_all: If false (default) only returns the first d-separation per
        left/right pair.
    :param verbose: Should a progress bar be shown?

    :returns: A set of conditional dependencies

    .. seealso::

        Original issue https://github.com/y0-causal-inference/y0/issues/24
    """
    if not return_all:
        return set(
            d_separations(
                graph,
                max_conditions=max_conditions,
                return_all=return_all,
                verbose=verbose,
            )
        )
    if policy is None:
        policy = get_topological_policy(graph)
    return minimal(
        d_separations(graph, max_conditions=max_conditions, return_all=return_all, verbose=verbose),
        policy=policy,
    )


def minimal(
    judgements: Iterable[DSeparationJudgement], policy: Policy | None = None
) -> set[DSeparationJudgement]:
    r"""Given some d-separations, reduces to a 'minimal' collection.

    For independencies of the form $A \perp B | {C_1, C_2, ...}$, the minimal collection
    will

    - Have only one independency with the same A/B nodes.
    - If there are multiples sets of C-nodes, the kept d-separation will be the
      first/minimal element in the group sorted according to `policy` argument.

    The default policy is to sort by the shortest set of conditions & then
    lexicographic.

    :param judgements: Collection of judgements to minimize
    :param policy: Function from d-separation to a representation suitable for sorting.

    :returns: A set of judgements that is minimal (as described above)
    """
    if policy is None:
        policy = _len_lex
    judgements = sorted(judgements, key=_judgement_grouper)
    return {min(vs, key=policy) for k, vs in groupby(judgements, _judgement_grouper)}


def get_topological_policy(
    graph: NxMixedGraph,
) -> Callable[[DSeparationJudgement], tuple[int, int]]:
    """Sort d-separations by condition length and topological order.

    This policy prefers small collections, and collections with variables earlier in
    topological order for collections of the same size.

    :param graph: a mixed graph

    :returns: A function suitable for use as a sort key on d-separations
    """
    order = list(graph.topological_sort())
    return partial(_topological_policy, order=order)


def _topological_policy(
    judgement: DSeparationJudgement, order: Sequence[Variable]
) -> tuple[int, int]:
    return (
        len(judgement.conditions),
        sum(order.index(v) for v in judgement.conditions),
    )


def _judgement_grouper(judgement: DSeparationJudgement) -> tuple[Variable, Variable]:
    """Simplify d-separation to just left & right element (for grouping left/right pairs)."""
    return judgement.left, judgement.right


def _len_lex(judgement: DSeparationJudgement) -> tuple[int, str]:
    """Sort by length of conditions & the lexicography a d-separation."""
    return len(judgement.conditions), ",".join(c.name for c in judgement.conditions)


def are_d_separated(
    graph: NxMixedGraph,
    a: Variable,
    b: Variable,
    *,
    conditions: Iterable[Variable] | None = None,
) -> DSeparationJudgement:
    """Test if nodes named by a & b are d-separated in G as described in [pearl2009]_.

    a & b can be provided in either order and the order of conditions does not matter.
    However, DSeparationJudgement may put things in canonical order.

    :param graph: Graph to test
    :param a: A node in the graph
    :param b: A node in the graph
    :param conditions: A collection of graph nodes

    :returns: T/F and the final graph (as evidence)

    :raises TypeError: if the left/right arguments or any conditions are not Variable
        instances
    :raises KeyError: if the left/right arguments or any conditions are not in the graph

    .. seealso::

        NetworkX implementation :func:`nx.d_separated`
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
    context = PairSeparationContext(evidence_graph=evidence_graph, condition_candidates=())
    separated = context.is_d_separated(a, b, conditions=conditions)

    return DSeparationJudgement.create(left=a, right=b, conditions=conditions, separated=separated)


def d_separations(
    graph: NxMixedGraph,
    *,
    max_conditions: int | None = None,
    verbose: bool | None = False,
    return_all: bool | None = False,
) -> Iterable[DSeparationJudgement]:
    """Generate d-separations in the provided graph.

    :param graph: Graph to search for d-separations.
    :param max_conditions: Longest set of conditions to investigate
    :param return_all: If false (default) only returns the first d-separation per
        left/right pair.
    :param verbose: If true, prints extra output with tqdm

    :yields: True d-separation judgments
    """
    vertices = tuple(sorted(graph.nodes(), key=str))
    for a, b in tqdm(
        combinations(vertices, 2),
        disable=not verbose,
        desc="Checking d-separations",
        unit="pair",
        total=len(vertices) * (len(vertices) - 1) // 2,
    ):
        context = PairSeparationContext.prepare(graph, a, b)
        for conditions in powerset(context.condition_candidates, stop=max_conditions):
            if context.is_d_separated(a, b, conditions=conditions):
                yield DSeparationJudgement.create(
                    left=a, right=b, conditions=conditions, separated=True
                )
                if not return_all:
                    break
