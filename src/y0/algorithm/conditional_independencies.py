# -*- coding: utf-8 -*-

"""An implementation to get conditional independencies of an ADMG."""

import copy
import warnings
from functools import partial
from itertools import chain, combinations, groupby
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
from tqdm.auto import tqdm

from ..dsl import Variable
from ..graph import NxMixedGraph
from ..struct import DSeparationJudgement
from ..util.combinatorics import powerset

__all__ = [
    "are_d_separated",
    "minimal",
    "get_conditional_independencies",
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
    """Given some d-separations, reduces to a 'minimal' collection.

    For independencies of the form A _||_ B | {C1, C2, ...} the minimal collection will::

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

    This policy will prefers small collections, and collections with variables earlier
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


def disorient(graph: NxMixedGraph) -> nx.Graph:
    """Convert an ADMG into a undirected (networkx) graph.

    :param graph: An ADMG
    :returns: A disoriented graph
    """
    rv = nx.Graph()
    rv.add_nodes_from(graph.nodes())
    rv.add_edges_from(graph.directed.edges())
    rv.add_edges_from(graph.undirected.edges())
    return rv


def get_moral_links(graph: NxMixedGraph) -> List[Tuple[Variable, Variable]]:
    """Generate links to ensure all co-parents in a graph are linked.

    May generate links that already exist as we assume we are not working on a multi-graph.

    :param graph: Graph to process
    :return: An collection of edges to add.
    """
    parents = [graph.directed.predecessors(node) for node in graph.nodes()]
    moral_links = [*chain(*[combinations(nodes, 2) for nodes in parents if len(parents) > 1])]
    return moral_links


def are_d_separated(
    graph: NxMixedGraph,
    a: Variable,
    b: Variable,
    *,
    conditions: Optional[Iterable[Variable]] = None,
) -> DSeparationJudgement:
    """Test if nodes named by a & b are d-separated in G.

    a & b can be provided in either order and the order of conditions does not matter.
    However DSeparationJudgement may put things in canonical order.

    :param graph: Graph to test
    :param a: A node in the graph
    :param b: A node in the graph
    :param conditions: A collection of graph nodes
    :return: T/F and the final graph (as evidence)
    :raises TypeError: if the left/right arguments or any conditions are
        not Variable instances
    """
    if conditions is None:
        conditions = set()
    if not isinstance(a, Variable):
        raise TypeError(f"left argument is not given as a Variable: {type(a)}: {a}")
    if not isinstance(b, Variable):
        raise TypeError(f"right argument is not given as a Variable: {type(b)}: {b}")
    if not all(isinstance(c, Variable) for c in conditions):
        raise TypeError(f"some conditions are not variables: {conditions}")
    if a not in graph or b not in graph:
        # If either a or b is not in the graph, raise a warning
        warning_message = f"Variables are d-separted because either '{a}' or '{b}' (or both) is not present in the graph."
        warnings.warn(warning_message)
        return DSeparationJudgement.create(left=a, right=b, conditions=conditions, separated=True)

    condition_names = {c for c in conditions}
    named = {a, b}.union(condition_names)

    # Filter to ancestors
    keep = graph.ancestors_inclusive(named)
    sg = copy.deepcopy(graph.subgraph(keep))

    # Moralize (link parents of mentioned nodes)
    for u, v in get_moral_links(sg):
        sg.add_undirected_edge(u, v)

    # disorient & remove conditions
    evidence_graph = disorient(sg)

    keep = set(evidence_graph.nodes) - set(condition_names)
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
