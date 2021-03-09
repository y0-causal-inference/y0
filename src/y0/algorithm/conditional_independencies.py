# -*- coding: utf-8 -*-

"""An implementation to get conditional independencies of an ADMG."""

import copy
from functools import partial
from itertools import chain, combinations, groupby
from typing import Iterable, Optional, Set, Tuple, Union

import networkx as nx
from ananke.graphs import ADMG, SG
from tqdm import tqdm

from ..graph import NxMixedGraph
from ..struct import DSeparationJudgement
from ..util.combinatorics import powerset

__all__ = [
    'are_d_separated',
    'minimal',
    'get_conditional_independencies',
]


def get_conditional_independencies(
    graph: Union[NxMixedGraph, SG],
    *,
    policy=None,
    max_conditions: Optional[int] = None,
    verbose: bool = False,
) -> Set[DSeparationJudgement]:
    """Get the conditional independencies from the given ADMG.

    Conditional independencies is the minmal set of d-separation judgements to cover
    the unique left/right combinations in all valid d-separation.

    :param graph: An acyclic directed mixed graph
    :param policy: Retention policy when more than one conditional independency option exists (see minimal for details)
    :param max_conditions: Maximum number of variable conditions (see d_separations)
    :param verbose: Use verbose output when generating d-separations
    :return: A set of conditional dependencies

    .. seealso:: Original issue https://github.com/y0-causal-inference/y0/issues/24
    """
    if isinstance(graph, NxMixedGraph):
        graph = graph.to_admg()
    if policy is None:
        policy = topological_policy(graph)
    return minimal(
        d_separations(graph, max_conditions=max_conditions, verbose=verbose),
        policy=policy,
    )


def minimal(judgements: Iterable[DSeparationJudgement], policy=None) -> Set[DSeparationJudgement]:
    """Given some d-separations, reduces to a 'minimal' collection.

    For indepdencies of the form A _||_ B | {C1, C2, ...} the minimal collection will::

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
    return {
        min(vs, key=policy)
        for k, vs in groupby(judgements, _judgement_grouper)
    }


def topological_policy(graph: ADMG):
    """Sort d-separations by condition length and topological order.

    This policy will prefers small collections, and collections with variables earlier
    in topological order for collections of the same size.

    :param graph: ADMG
    :return: A function suitable for use as a sort key on d-separations
    """
    order = graph.topological_sort()
    return partial(_topological_policy, order=order)


def _topological_policy(judgement: DSeparationJudgement, order):
    return (
        len(judgement.conditions),
        sum((order.index(v) for v in judgement.conditions)),
    )


def _judgement_grouper(judgement: DSeparationJudgement) -> Tuple[str, str]:
    """Simplify d-separation to just left & right element (for grouping left/right pairs)."""
    return judgement.left, judgement.right


def _len_lex(judgement: DSeparationJudgement) -> Tuple[int, str]:
    """Sort by length of conditions & the lexicography a d-separation."""
    return len(judgement.conditions), ",".join(judgement.conditions)


def disorient(graph: SG) -> nx.Graph:
    """Convert an :mod:`ananke` mixed directed/undirected into a undirected (networkx) graph."""
    rv = nx.Graph()
    rv.add_nodes_from(graph.vertices)
    rv.add_edges_from(chain(graph.di_edges, graph.ud_edges, graph.bi_edges))
    return rv


def get_moral_links(graph: SG):
    """Generate links to ensure all co-parents in a graph are linked.

    May generate links that already exist as we assume we are not working on a multi-graph.

    :param graph: Graph to process
    :return: An collection of edges to add.
    """
    parents = [graph.parents([v]) for v in graph.vertices]
    moral_links = [*chain(*[combinations(nodes, 2) for nodes in parents if len(parents) > 1])]
    return moral_links


def are_d_separated(graph: SG, a: str, b: str, *, conditions: Optional[Iterable[str]] = None) -> DSeparationJudgement:
    """Test if nodes named by a & b are d-separated in G.

    a & b can be provided in either order and the order of conditions does not matter.
    However DSeparationJudgement may put things in canonical order.

    :param graph: Graph to test
    :param a: A node in the graph
    :param b: A node in the graph
    :param conditions: A collection of graph nodes
    :return: T/F and the final graph (as evidence)
    """
    conditions = set(conditions) if conditions else set()
    named = {a, b}.union(conditions)

    # Filter to ancestors
    keep = graph.ancestors(named)
    graph = copy.deepcopy(graph.subgraph(keep))

    # Moralize (link parents of mentioned nodes)
    for u, v in get_moral_links(graph):
        graph.add_udedge(u, v)

    # disorient & remove conditions
    evidence_graph = disorient(graph)

    keep = set(evidence_graph.nodes) - set(conditions)
    evidence_graph = evidence_graph.subgraph(keep)

    # check for path....
    separated = not nx.has_path(evidence_graph, a, b)  # If no path, then d-separated!

    return DSeparationJudgement.create(left=a, right=b, conditions=conditions, separated=separated)


def d_separations(
    graph: Union[NxMixedGraph, SG],
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
    if isinstance(graph, NxMixedGraph):
        graph = graph.to_admg()

    vertices = set(graph.vertices)
    for a, b in tqdm(combinations(vertices, 2), disable=not verbose, desc="d-separation check"):
        for conditions in powerset(vertices - {a, b}, stop=max_conditions):
            judgement = are_d_separated(graph, a, b, conditions=conditions)
            if judgement.separated:
                yield judgement
                if not return_all:
                    break
