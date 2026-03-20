"""An implementation to get conditional independencies of an ADMG from [pearl2009]_."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from itertools import combinations, groupby, islice
from typing import Any, NamedTuple

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


def test_conditional_independencies(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    *,
    method: CITest | None = None,
    boolean: bool = False,
    significance_level: float | None = None,
    _method_checked: bool = False,
    max_conditions: int | None = None,
) -> list[tuple[DSeparationJudgement, bool | CITestTuple]]:
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
            judgement.test(
                data,
                boolean=boolean,
                method=method,
                significance_level=significance_level,
                _method_checked=True,
            ),
        )
        for judgement in get_conditional_independencies(graph, max_conditions=max_conditions)
    ]


Policy = Callable[[DSeparationJudgement], Any]


class _PairSeparationContext(NamedTuple):
    """Pair-specific graph state reused across conditioning sets."""

    evidence_graph: nx.Graph
    condition_candidates: tuple[Variable, ...]


class _PairBatchTask(NamedTuple):
    """A batch of pair searches to evaluate in one worker."""

    graph: NxMixedGraph
    pairs: tuple[tuple[Variable, Variable], ...]
    max_conditions: int | None
    return_all: bool


def get_conditional_independencies(
    graph: NxMixedGraph,
    *,
    policy: Policy | None = None,
    max_conditions: int | None = None,
    n_jobs: int | None = None,
    batch_size: int | None = None,
    **kwargs: Any,
) -> set[DSeparationJudgement]:
    """Get the conditional independencies from the given ADMG.

    Conditional independencies is the minimal set of d-separation judgements to cover
    the unique left/right combinations in all valid d-separation.

    :param graph: An acyclic directed mixed graph
    :param policy: Retention policy when more than one conditional independency option
        exists (see minimal for details)
    :param max_conditions: Longest set of conditions to investigate
    :param kwargs: Other keyword arguments are passed to :func:`d_separations`

    :returns: A set of conditional dependencies

    .. seealso::

        Original issue https://github.com/y0-causal-inference/y0/issues/24
    """
    if not kwargs.get("return_all", False):
        return set(
            d_separations(
                graph,
                max_conditions=max_conditions,
                n_jobs=n_jobs,
                batch_size=batch_size,
                **kwargs,
            )
        )
    if policy is None:
        policy = get_topological_policy(graph)
    return minimal(
        d_separations(
            graph,
            max_conditions=max_conditions,
            n_jobs=n_jobs,
            batch_size=batch_size,
            **kwargs,
        ),
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


def _prepare_pair_separation_context(
    graph: NxMixedGraph,
    a: Variable,
    b: Variable,
) -> _PairSeparationContext:
    named = {a, b}
    keep = graph.ancestors_inclusive(named)
    evidence_graph = graph.subgraph(keep).moralize().disorient()
    condition_candidates = _order_condition_candidates(
        evidence_graph,
        a,
        b,
        candidates=set(evidence_graph.nodes) - named,
    )
    return _PairSeparationContext(
        evidence_graph=evidence_graph,
        condition_candidates=condition_candidates,
    )


def _order_condition_candidates(
    evidence_graph: nx.Graph,
    a: Variable,
    b: Variable,
    *,
    candidates: Iterable[Variable],
) -> tuple[Variable, ...]:
    """Prioritize nodes closer to the queried pair for earlier separator discovery."""
    a_distances = nx.single_source_shortest_path_length(evidence_graph, a)
    b_distances = nx.single_source_shortest_path_length(evidence_graph, b)

    def key(node: Variable) -> tuple[float, float, float, str]:
        a_distance = a_distances.get(node, float("inf"))
        b_distance = b_distances.get(node, float("inf"))
        return (a_distance + b_distance, min(a_distance, b_distance), a_distance, str(node))

    return tuple(sorted(candidates, key=key))


def _is_d_separated_in_context(
    context: _PairSeparationContext,
    a: Variable,
    b: Variable,
    *,
    conditions: Iterable[Variable],
) -> bool:
    conditions = set(conditions)
    if conditions:
        evidence_graph = context.evidence_graph.subgraph(set(context.evidence_graph.nodes) - conditions)
    else:
        evidence_graph = context.evidence_graph
    return not nx.has_path(evidence_graph, a, b)


def _chunked(
    iterable: Iterable[tuple[Variable, Variable]],
    size: int,
) -> Iterable[tuple[tuple[Variable, Variable], ...]]:
    iterator = iter(iterable)
    while chunk := tuple(islice(iterator, size)):
        yield chunk


def _get_pair_batch_size(pair_count: int, *, n_jobs: int) -> int:
    return max(1, pair_count // (n_jobs * 4))


def _search_pair_batch(task: _PairBatchTask) -> list[DSeparationJudgement]:
    judgements = []
    for a, b in task.pairs:
        context = _prepare_pair_separation_context(task.graph, a, b)
        for conditions in powerset(context.condition_candidates, stop=task.max_conditions):
            if _is_d_separated_in_context(context, a, b, conditions=conditions):
                judgements.append(
                    DSeparationJudgement.create(left=a, right=b, conditions=conditions, separated=True)
                )
                if not task.return_all:
                    break
    return judgements


def _parallel_search_pair_batches(
    tasks: Sequence[_PairBatchTask],
    *,
    n_jobs: int,
) -> Iterable[list[DSeparationJudgement]]:
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            yield from executor.map(_search_pair_batch, tasks)
            return
    except PermissionError:
        pass
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        yield from executor.map(_search_pair_batch, tasks)


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
    context = _PairSeparationContext(evidence_graph=evidence_graph, condition_candidates=())
    separated = _is_d_separated_in_context(context, a, b, conditions=conditions)

    return DSeparationJudgement.create(left=a, right=b, conditions=conditions, separated=separated)


def d_separations(
    graph: NxMixedGraph,
    *,
    max_conditions: int | None = None,
    verbose: bool | None = False,
    return_all: bool | None = False,
    n_jobs: int | None = None,
    batch_size: int | None = None,
) -> Iterable[DSeparationJudgement]:
    """Generate d-separations in the provided graph.

    :param graph: Graph to search for d-separations.
    :param max_conditions: Longest set of conditions to investigate
    :param return_all: If false (default) only returns the first d-separation per
        left/right pair.
    :param verbose: If true, prints extra output with tqdm
    :param n_jobs: Number of worker processes. Defaults to serial execution.
    :param batch_size: Number of pairs to submit per worker task.

    :yields: True d-separation judgements
    """
    vertices = tuple(sorted(graph.nodes(), key=str))
    pairs = tuple(combinations(vertices, 2))
    if n_jobs is None or n_jobs <= 1:
        for a, b in tqdm(
            pairs,
            disable=not verbose,
            desc="Checking d-separations",
            unit="pair",
            total=len(pairs),
        ):
            context = _prepare_pair_separation_context(graph, a, b)
            for conditions in powerset(context.condition_candidates, stop=max_conditions):
                if _is_d_separated_in_context(context, a, b, conditions=conditions):
                    yield DSeparationJudgement.create(left=a, right=b, conditions=conditions, separated=True)
                    if not return_all:
                        break
        return

    if batch_size is None:
        batch_size = _get_pair_batch_size(len(pairs), n_jobs=n_jobs)
    tasks = [
        _PairBatchTask(
            graph=graph,
            pairs=pair_batch,
            max_conditions=max_conditions,
            return_all=bool(return_all),
        )
        for pair_batch in _chunked(pairs, batch_size)
    ]
    iterator = tasks
    if verbose:
        iterator = tqdm(
            tasks,
            disable=not verbose,
            desc="Checking d-separations",
            unit="batch",
            total=len(tasks),
        )
    for judgements in _parallel_search_pair_batches(tuple(iterator), n_jobs=n_jobs):
        yield from judgements
