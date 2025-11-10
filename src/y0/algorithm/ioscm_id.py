"""Implementation of the ID algorithm for input-output structural causal models (ioSCMs).

.. [forré20a] http://proceedings.mlr.press/v115/forre20a/forre20a.pdf.
.. [forré20b] http://proceedings.mlr.press/v115/forre20a/forre20a-supp.pdf
"""

import copy
import logging
from collections.abc import Callable, Collection, Iterable

import networkx as nx

from y0.algorithm.separation.sigma_separation import get_equivalence_classes
from y0.dsl import Variable
from y0.graph import NxMixedGraph

__all__ = [
    # TODO do a proper audit of which of these a user should ever have to import
    "get_strongly_connected_components",
    "get_vertex_consolidated_district",
    "get_consolidated_district",
    "get_graph_consolidated_districts",
    "get_apt_order",
    "is_apt_order",
]

logger = logging.getLogger(__name__)

#: Variable to component mapping
NodeToComponent = dict[Variable, frozenset[Variable]]
ComponentToNode = dict[frozenset[Variable], Variable]


def get_strongly_connected_components(graph: NxMixedGraph) -> set[frozenset[Variable]]:
    r"""Return the strongly-connected components for a graph.

    :math: The strongly connected component of $v$ in $G$ is defined to be:
    $\text{Sc}^{G}(v):= \text{Anc}^{G}(v)\cap \text{Desc}^{G}(v)$.

    :param graph:
        The corresponding graph.
    :returns:
        A set of frozen sets of variables comprising $\text{Sc}^{G}(v)$ for all vertices $v$.
    """
    return set(
        frozenset(component) for component in nx.strongly_connected_components(graph.directed)
    )


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
    # Strategy: (O(N^2))
    # 1. Get the strongly-connected component of every vertex in the graph, using the networkx function.
    # 2. Create a new graph that replaces every directed edge in a strongly-connected component with a bidirected edge.
    # 3. Get the district for the new graph that contains the target vertex in question using get_district().
    # 4. Return the resulting set.
    converted_graph = scc_to_bidirected(graph)
    result = converted_graph.get_district(v)
    return result


def get_consolidated_district(graph: NxMixedGraph, vertices: Collection[Variable]) -> set[Variable]:
    r"""Return the consolidated districts for one or more vertices in a graph.

    See Definition 9.1 of [forré20a].

    :math: Let $G$ be a directed mixed graph (DMG) with set of nodes $V$. Let $v \in V$. The
    consolidated district $\text{Cd}^{G}(v)$ of $v$ in $G$ is given by all nodes $w \in V$ for which
    there exist $k \ge 1$ nodes $(v_1,\dots,v_k)$ in $G$ such that $v_1 = v, v_k = w$ and for
    $i = 2,\dots\,k$ we have that the bidirected edge $v_{i-1} \leftrightarrow v_i$ is in $G$
    or that $v_i \in \text{Sc}^{G}(v_{i-1})$. For $B \subseteq V$ we write
    $\text{Cd}^{G}(B) := \bigcup_{v\in B}\text{Cd}^{G}(v)$. Let
    $\mathcal{CD}(G)$ be the set of consolidated districts of $G$.

    Note: it's not entirely clear from the text whether the return value is meant to be
    a set of sets of vertices or just a set of vertices. We return a set of vertices in order
    for the notation to be consistent with Notation 9.4 part 3 and Remark 9.5: in Remark 9.5,
    the function $\text{Anc}^{G}$ only makes sense when called on a set of vertices, not a
    set of sets of vertices.

    :param graph:
        The corresponding graph.
    :param vertices:
        The vertex for which the consolidated district is to be retrieved.
    :returns:
        The set of consolidated districts for the variables in $B$.
    """
    # 1. Get the strongly-connected component of every vertex in the graph, using the networkx function.
    # 2. Create a new graph that replaces every directed edge in a strongly-connected component with a bidirected edge.
    # 3. Get all the consolidated districts.
    # 4. Return the union of all of them as one set.
    converted_graph = scc_to_bidirected(graph)
    result: set[Variable] = set()
    for vertex in vertices:
        district = converted_graph.get_district(vertex)
        if vertex not in result:
            result.update(district)
    return result


def get_graph_consolidated_districts(graph: NxMixedGraph) -> set[frozenset[Variable]]:
    r"""Return the set of all consolidated districts in a graph.

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
    :returns:
        The set of consolidated districts for the graph.
    """
    # 1. Get the strongly-connected component of every vertex in the graph, using the networkx function.
    # 2. Create a new graph that replaces every directed edge in a strongly-connected component with a bidirected edge.
    # 3. Get each consolidated district as a frozen set.
    # 4. Return all of them as a set of frozen sets.
    converted_graph = scc_to_bidirected(graph)
    result: set[frozenset[Variable]] = set()
    for node in graph.nodes():
        if node not in result:
            result.add(converted_graph.get_district(node))
    return result


def scc_to_bidirected(graph: NxMixedGraph) -> NxMixedGraph:
    """Replace every edge in a strongly-connected component with a bidirected edge."""
    new_graph = copy.deepcopy(graph)

    node_to_component = get_equivalence_classes(graph)

    edges_to_convert = {
        (ego, alter)
        for ego, alter in new_graph.directed.edges
        if node_to_component[ego] == node_to_component[alter]
    }

    for ego, alter in edges_to_convert:
        new_graph.directed.remove_edge(ego, alter)
        new_graph.undirected.add_edge(ego, alter)

    logger.warning(f"In _convert_strongly_connected_components: graph = {new_graph!s}")
    return new_graph


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
    # Strategy:
    # 1. Get the strongly-connected components and replace each one with a single vertex.
    #    An edge going into or out of the strongly-connected component becomes an edge going
    #    into or out of the representative vertex
    # 2. Topologically sort the resulting graph
    # 3. For the vertices in the topologically sorted list associated with strongly-connected components,
    #    replace each one with a list of vertices in the strongly-connected component in any order
    # 4. Flatten the resulting list (e.g., [A, B, [C, D], E] -> [A, B, C, D, E])
    # JZ: Consider not even bothering to add an edge into or out of a strongly connected component
    #     once it's already been added.

    # 1.
    new_graph, node_to_component = _simplify_strongly_connected_components(graph)
    components = [sorted(node_to_component[v]) for v in new_graph.topological_sort()]
    logger.warning(f"In get_apt_order: original vertex list, not flattened = {components!s}")
    nodes = [node for component in components for node in component]
    logger.warning(f"In get_apt_order: flattened output vertex list = {nodes!s}")
    return nodes


def _min_from_component(component: Iterable[Variable]) -> Variable:
    return min(component)


def _simplify_strongly_connected_components(
    graph: NxMixedGraph, _get_rep_node: Callable[[Iterable[Variable]], Variable] | None = None
) -> tuple[NxMixedGraph, dict[Variable, frozenset[Variable]]]:
    r"""Reduce each strongly-connected component in a directed graph to a single vertex.

    This is a helper function for generating the assembling pseudo-topological order for a graph.

    Get the strongly-connected components and replace each one with a single vertex. An edge going into or out
    of the strongly-connected component becomes an edge going into or out of the representative vertex.

    :param graph:
        The input graph.
    :returns:
        The simplified graph and a dictionary mapping vertices representing strongly-connected components
        in the new graph to the vertices in each strongly-connected component in the original graph.
    """
    comp_to_rep_node: ComponentToNode = {}
    node_to_component: NodeToComponent = {}
    representative_node_to_component: NodeToComponent = {}

    if _get_rep_node is None:
        _get_rep_node = _min_from_component

    for component in get_strongly_connected_components(graph):
        representative_node = _get_rep_node(component)
        representative_node_to_component[representative_node] = component
        comp_to_rep_node[component] = representative_node
        for node in component:
            node_to_component[node] = component

    directed: set[tuple[Variable, Variable]] = set()
    # O(V^2), sorting is just to make testing predictable
    for ego, alter in sorted(graph.directed.edges):
        u_component = node_to_component[ego]
        v_component = node_to_component[alter]
        if u_component == v_component:
            continue
        directed.add((comp_to_rep_node[u_component], comp_to_rep_node[v_component]))

    # The undirected edges don't affect the topological ordering, but they may indicate the presence
    # of vertices otherwise not included in the graph. Such vertices must be their own strongly-connected
    # components since they're not present in any directed edges. And they're therefore their own representative
    # vertices. So, replacing the "ego" (first vertex) in each edge in graph.undirected.edges with the
    # representative vertex for the strongly-connected component corresponding to the ego, and doing the
    # same for the "alter" (slight abuse of naming), will maintain the vertices not connected to other
    # strongly-connected components in the resulting graph, while getting rid of undirected edges between
    # vertices within strongly-connected components. And thus topological_sort will work on the result.
    undirected: set[tuple[Variable, Variable]] = set()
    for ego, alter in sorted(graph.undirected.edges):
        u_component = node_to_component[ego]
        v_component = node_to_component[alter]
        if u_component == v_component:
            continue
        undirected.add((comp_to_rep_node[u_component], comp_to_rep_node[v_component]))
        # If we add both (u,v) and (v,u), that will go away when the actual graph gets
        # produced, so there's no need for a test

    new_graph = NxMixedGraph.from_edges(directed=directed, undirected=undirected)
    return new_graph, representative_node_to_component


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
    # TODO: Confirm we need the function
    # Strategy (not sure this is optimal yet):
    # 1. Get the strongly-connected components
    # 2. For each strongly-connected component, flag the associated vertices in the input list
    #    and make sure the vertices are consecutive in the 'order' param
    # 3. Replace the vertices in 'order' associated with a single strongly-connected component
    #    by one vertex in that component (with a dictionary mapping the vertex name to the
    #    set of vertices in the strongly-connected component). An edge going into or out of the
    #    strongly-connected component becomes an edge going into or out of the representative vertex
    # 4. Test whether the result is in topologically sorted order
