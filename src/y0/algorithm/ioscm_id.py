# -*- coding: utf-8 -*-

"""Implementation of the ID algorithm for input-output structural causal models (ioSCMs).

.. [forré20a] http://proceedings.mlr.press/v115/forre20a/forre20a.pdf.
.. [forré20b] http://proceedings.mlr.press/v115/forre20a/forre20a-supp.pdf
"""

import logging
from typing import Collection

import networkx as nx

from y0.dsl import Variable  # P,; R,; W,; X,; Y,; Z,
from y0.graph import NxMixedGraph

__all__ = [
    # TODO do a proper audit of which of these a user should ever have to import
    # Utilities
    "get_strongly_connected_component",
    "get_vertex_consolidated_district",
    "get_consolidated_district",
    "get_apt_order",
    "is_apt_order",
    # TODO add functions/classes/variables you want to appear in the docs and be exposed to the user in this list
    #  Run tox -e docs then `open docs/build/html/index.html` to see docs
]

logger = logging.getLogger(__name__)


def get_strongly_connected_component(graph: NxMixedGraph, v: Variable) -> set[Variable]:
    r"""Return the strongly-connected component within which a graph vertex lies.

    :math: The strongly connected component of $v$ in $G$ is defined to be:
    $\text{Sc}^{G}(v):= \text{Anc}^{G}(v)\cap \text{Desc}^{G}(v)$.

    :param graph:
        The corresponding graph.
    :param v:
        The vertex for which the strongly connected component is to be retrieved.
    :returns:
        The set of variables comprising the strongly connected component $\text{Sc}^{G}(v)$.
    """
    #ancestors = graph.ancestors_inclusive(v)
    #descendents = graph.descendants_inclusive(v)
    #logger.warning(f"In get_strongly_connected_component: ancestors = {str(ancestors)}")
    #logger.warning(f"In get_strongly_connected_component: descendents = {str(descendents)}")
    #result = ancestors.intersection(descendents)
    #logger.warning(f"In get_strongly_connected_component: returning {str(result)}")
    # TODO: It might be faster to use strongly_connected_components:
    #return result
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.strongly_connected_components.html#networkx.algorithms.components.strongly_connected_components
    return graph.ancestors_inclusive(v).intersection(graph.descendants_inclusive(v))


def get_vertex_consolidated_district(graph: NxMixedGraph, v: Variable) -> frozenset[Variable]:
    r"""Return the consolidated district for a single vertex in a graph.

    See Definition 9.1 of [forré20a].

    :math: Let $G$ be a directed mixed graph (DMG) with set of nodes $V$. Let $v \in V$. The
    consolidated district $\text{Cd}^{G}(v)$ of $v$ in $G$ is given by all nodes $w \in V$ for which
    there exist $k \ge 1$ nodes $(v_1,\dots,v_k)$ in $G$ such that $v_1 = v, v_k = w$ and for
    $i = 2,\dots\,k$ we have that the bidirected edge $v_{i-1} \leftrightarrow v_i$ is in $G$
    or that $v_i \in \text{Sc}^{G}(v_{i-1})$. For $B \subseteq V$ we write
    $\text{Cd}^{G}(B) := \bigcup_{v\in B}\text{Cd}^{G}(v)$. Let
    $\mathcal{CD}(G)$ be the set of consolidated districts of $G$.

    (This function retrieves the consolidated district for $v$, not $B$.)

    :param graph:
        The corresponding graph.
    :param v:
        The vertex for which the consolidated district is to be retrieved.
    :returns:
        The set of variables comprising $\text{Cd}^{G}(v)$.
    """
    # Strategy:
    # 1. Get the strongly-connected component of every vertex in the graph, using the networkx function.
    # 2. Create a new graph that replaces every directed edge in a strongly-connected component with a bidirected edge.
    # 3. Get the district for the new graph that contains the target vertex in question using get_district().
    # 4. Return the resulting set.
    raise NotImplementedError


def get_consolidated_district(
    graph: NxMixedGraph, v: Collection[Variable]
) -> frozenset[frozenset[Variable]]:
    r"""Return the consolidated districts for one or more vertices in a graph.

    See Definition 9.1 of [forré20a].

    :math: Let $G$ be a directed mixed graph (DMG) with set of nodes $V$. Let $v \in V$. The
    consolidated district $\text{Cd}^{G}(v)$ of $v$ in $G$ is given by all nodes $w \in V$ for which
    there exist $k \ge 1$ nodes $(v_1,\dots,v_k)$ in $G$ such that $v_1 = v, v_k = w$ and for
    $i = 2,\dots\,k$ we have that the bidirected edge $v_{i-1} \leftrightarrow v_i$ is in $G$
    or that $v_i \in \text{Sc}^{G}(v_{i-1})$. For $B \subseteq V$ we write
    $\text{Cd}^{G}(B) := \bigcup_{v\in B}\text{Cd}^{G}(v)$. Let
    $\mathcal{CD}(G)$ be the set of consolidated districts of $G$.

    :param graph:
        The corresponding graph.
    :param v:
        The vertex for which the consolidated district is to be retrieved.
    :returns:
        The set of consolidated districts for the variables in $B$.
    """
    # 1. Get the strongly-connected component of every vertex in the graph, using the networkx function.
    # 2. Create a new graph that replaces every directed edge in a strongly-connected component with a bidirected edge.
    # 3. Get all the consolidated districts.
    # 4. Return the resulting set.
    raise NotImplementedError


# Utility function
def _convert_strongly_connected_components(graph: NxMixedGraph) -> NxMixedGraph:
    r"""Replace every edge in a strongly-connected component with a bidirected edge."""
    # undirected: nx.Graph = field(default_factory=nx.Graph)

    sccs = nx.connected_components(graph.undirected)
    logger.warning(f"In _convert_strongly_connected_components: graph = {str(graph)}")
    logger.warning(f"In _convert_strongly_connected_components: sccs = {str(sccs)}")
    return graph  # TODO: Implement the function for real


def get_apt_order(graph: NxMixedGraph) -> list[Variable]:
    r"""Return one possible assembling pseudo-topological order ("apt-order") for the vertices in a graph.

    See Definition 9.2 of [forré20a].

    :math: Let $G$ be a directed mixed graph (DMG) with set of nodes $V$. An assembling
    pseudo-topological order (apt-order) of $G$ is a total order $\lt$ on $V$ with the following two properties:

    1. For every $v, w \in V$ we have:

       $w \in \text{Anc}^{G}(v) \backslash \text{Sc}^{G}(v) \Longrightarrow w \lt v$.

    2. For every $v_1, v_2, w \in V$ we have:

      $v_2 \in \text{Sc}^{G}(v_1) \land(v_1 \le w \le v_2) \Longrightarrow w \in \text{Sc}^{G}(v_1)$.

    :param graph:
        The corresponding graph.
    :returns:
        An apt-order for the vertices in $G$.
    """
    raise NotImplementedError


def is_apt_order(order: list[Variable], graph: NxMixedGraph) -> bool:
    r"""Verify that a list of vertices is a possible assembling pseudo-topological order ("apt-order") for a graph.

    See Definition 9.2 of [forré20a].

    :math: Let $G$ be a directed mixed graph (DMG) with set of nodes $V$. An assembling
    pseudo-topological order (apt-order) of $G$ is a total order $\lt$ on $V$ with the following two properties:

    1. For every $v, w \in V$ we have:

       $w \in \text{Anc}^{G}(v) \backslash \text{Sc}^{G}(v) \Longrightarrow w \lt v$.

    2. For every $v_1, v_2, w \in V$ we have:

      $v_2 \in \text{Sc}^{G}(v_1) \land(v_1 \le w \le v_2) \Longrightarrow w \in \text{Sc}^{G}(v_1)$.

    :param order:
        The candidate apt-order.
    :param graph:
        The corresponding graph.
    :returns:
        True if the candidate apt-order is a possible apt-order for the graph, False otherwise.
    """
    raise NotImplementedError
