# -*- coding: utf-8 -*-

"""An implementation to get conditional independencies of an ADMG."""

import copy
from itertools import combinations, chain, groupby
from typing import Set, Optional, Iterable, TypeVar, Collection

import networkx as nx
from tqdm import tqdm
from ananke.graphs import ADMG, SG

from ..struct import DSeparationJudgement

X = TypeVar('X')

__all__ = [
    'get_conditional_independencies',
]


def get_conditional_independencies(graph: ADMG,
                                   *,
                                   max_conditions: Optional[int] = None,
                                   verbose: bool = False) -> Set[DSeparationJudgement]:
    """Get the conditional independencies from the given ADMG.

    Conditional independencies is the minmal set of d-separation judgements to cover
    the unique left/right combinations in all valid d-separation.

    :param graph: An acyclic directed mixed graph
    :return: A set of conditional dependencies

    .. seealso:: Original issue https://github.com/y0-causal-inference/y0/issues/24
    """

    return minimal(d_separations(graph, max_conditions=max_conditions, verbose=verbose))


def minimal(dseps, policy=None):
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
    def _grouper(dsep):
        """Returns a tuple of left & right side of a d-separation."""
        return (dsep.left, dsep.right)

    def _len_lex(dsep):
        """Sort by length of conditions & the lexicography a d-separation"""
        return (len(dsep.conditions), ",".join(dsep.conditions))

    policy = _len_lex if policy is None else policy

    dseps = sorted(dseps, key=_grouper)
    instances = {k: min(vs, key=policy)
                 for k, vs in groupby(dseps, _grouper)}
    return instances.values()


def powerset(iterable: Iterable[X],
             start: int = 0,
             stop: Optional[int] = None) -> Iterable[Collection[X]]:
    """Get successively longer combinations of the source.

    :param iterable: List to get combinations from
    :param start: smallest combination to get (default 0)
    :param stop: Largest combination to get (None means length of the list and is the default)

    .. seealso: :func:`more_iterools.powerset` for a non-constrainable implementation
    """
    s = list(iterable)
    if stop is None:
        stop = len(s) + 1
    return chain.from_iterable(combinations(s, r) for r in range(start, stop))


def disorient(graph: SG) -> nx.Graph:
    """Disorient the ananke segregated graph to a simple networkx graph."""
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


def are_d_separated(graph: SG, a, b, *, conditions=frozenset()) -> DSeparationJudgement:
    """Tests if nodes named by a & b are d-separated in G.

    Additional conditions can be provided with the optional 'conditions' parameter.
    returns T/F and the final graph (as evidence)
    """
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

    return DSeparationJudgement.create(a, b,
                                       conditions=conditions,
                                       separated=separated)


def d_separations(graph: SG,
                  *,
                  max_conditions: Optional[int] = None,
                  verbose: bool = False) -> Iterable[DSeparationJudgement]:
    """
    Returns an iterator of all of the d-separations in the provided graph.

    graph -- Graph to search for d-separations.
    max_conditions -- Longest set of conditionss to investigate
    verbose -- If true, prints extra output with tqdm
    """

    verticies = set(graph.vertices)
    for a, b in tqdm(combinations(verticies, 2), disable=not verbose, desc="d-separation check"):
        for conditions in powerset(verticies - {a, b}, stop=max_conditions):
            rslt = are_d_separated(graph, a, b, conditions=conditions)
            if rslt:
                yield rslt
