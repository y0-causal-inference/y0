# -*- coding: utf-8 -*-

"""An implementation to get conditional independencies of an ADMG."""

import copy
from itertools import chain, combinations, groupby
from typing import Iterable, Optional, Set, Tuple, Union

import networkx as nx
from ananke.graphs import SG
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
    :return: A set of conditional dependencies

    .. seealso:: Original issue https://github.com/y0-causal-inference/y0/issues/24
    """
    if isinstance(graph, NxMixedGraph):
        graph = graph.to_admg()
    return minimal(d_separations(graph, max_conditions=max_conditions, verbose=verbose),
                   policy=topological_policy(graph))


def minimal(judgements: Iterable[DSeparationJudgement], policy=None) -> Set[DSeparationJudgement]:
    """Given some d-separations, reduces to a 'minimal' collection.

    For indepdencies of the form A _||_ B | {C1, C2, ...} the minmal collection will:
         * Have only one indepdency with the same A/B nodes.
         * For sets of C nodes, replacement is made according to the 'policy' argument

    'policy' -- Function from dseparation to representation suitable for sorting
                (used it is used as the 'key' function in python's 'sorted').
                The kept d-separation in an A/B pair will be the first/minimal
                element in the group sorted according to policy.

    The default replacement policy is shortest set of conditions & then lexicographic.

    TODO: Investigate a shortest + topological policy. If the model is incomplete
          and there are unobserved links, a topological policy might be less-senstiive
          to such model errors.
    """
    if policy is None:
        policy = _len_lex
    judgements = sorted(judgements, key=_judgement_grouper)
    return {
        min(vs, key=policy)
        for k, vs in groupby(judgements, _judgement_grouper)
    }


def topological_policy(graph):
    """
    :param graph: ADMG
    """
    order = graph.topological_sort()
    return lambda dsep: (len(dsep.conditions),
                         sum((order.index(v) for v in dsep.conditions)))


def _judgement_grouper(judgement: DSeparationJudgement) -> Tuple[str, str]:
    """Returns a tuple of left & right side of a d-separation."""
    return judgement.left, judgement.right


def _len_lex(judgement: DSeparationJudgement) -> Tuple[int, str]:
    """Sort by length of conditions & the lexicography a d-separation"""
    return len(judgement.conditions), ",".join(judgement.conditions)


def disorient(graph: SG) -> nx.Graph:
    """Disorient the :mod:`ananke` segregated graph to a simple networkx graph."""
    rv = nx.Graph()
    rv.add_nodes_from(graph.vertices)
    rv.add_edges_from(chain(graph.di_edges, graph.ud_edges, graph.bi_edges))
    return rv


def get_moral_links(graph: SG):
    """
    If a node in the graph has more than one parent BUT not a link between them,
    generates that link.  Returns all the edges to add.
    """
    parents = [graph.parents([v]) for v in graph.vertices]
    augments = [*chain(*[combinations(nodes, 2) for nodes in parents if len(parents) > 1])]
    return augments


def are_d_separated(graph: SG, a, b, *, conditions: Optional[Iterable[str]] = None) -> DSeparationJudgement:
    """Tests if nodes named by a & b are d-separated in G.

    Additional conditions can be provided with the optional 'conditions' parameter.
    returns T/F and the final graph (as evidence)
    """
    conditions = set(conditions) if conditions else set()
    named = {a, b}.union(conditions)

    # Filter to ancestors
    keep = graph.ancestors(named)
    graph = copy.deepcopy(graph.subgraph(keep))

    # Moralize (link parents of mentioned nodes)
    for u, v in get_moral_links(graph):
        graph.add_udedge(u, v)

    # disorient & remove conditionss
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
    verbose: Optional[bool] = False
) -> Iterable[DSeparationJudgement]:
    """
    Returns an iterator of all of the d-separations in the provided graph.

    graph -- Graph to search for d-separations.
    max_conditions -- Longest set of conditions to investigate
    verbose -- If true, prints extra output with tqdm
    """
    if isinstance(graph, NxMixedGraph):
        graph = graph.to_admg()
    vertices = set(graph.vertices)
    for a, b in tqdm(combinations(vertices, 2), disable=not verbose, desc="d-separation check"):
        for conditions in powerset(vertices - {a, b}, stop=max_conditions):
            judgement = are_d_separated(graph, a, b, conditions=conditions)
            if judgement.separated:
                yield judgement
