# -*- coding: utf-8 -*-

"""Implementation of the ID algorithm for input-output structural causal models (ioSCMs).

.. [forré20a] http://proceedings.mlr.press/v115/forre20a/forre20a.pdf.
.. [forré20b] http://proceedings.mlr.press/v115/forre20a/forre20a-supp.pdf
"""

from typing import Collection

from y0.dsl import Variable  # P,; R,; W,; X,; Y,; Z,
from y0.graph import NxMixedGraph


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
    raise NotImplementedError


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
    raise NotImplementedError


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
