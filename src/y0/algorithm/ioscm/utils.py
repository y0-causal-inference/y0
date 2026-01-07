"""Utilities supporting operations on the ioSCM data structure."""

import copy
import logging
from collections.abc import Collection, Iterable
from typing import Annotated, TypeAlias

import networkx as nx

from ..separation.sigma_separation import get_equivalence_classes
from ...dsl import Variable
from ...graph import NxMixedGraph
from ...util import InPaperAs

__all__ = [
    "get_apt_order",
    "get_consolidated_district",
    "get_graph_consolidated_districts",
    "get_strongly_connected_components",
    "get_vertex_consolidated_district",
    "is_apt_order",
    "scc_to_bidirected",
    "simplify_strongly_connected_components",
]

logger = logging.getLogger(__name__)

#: Variable to component mapping
NodeToComponent: TypeAlias = dict[Variable, frozenset[Variable]]
ComponentToNode: TypeAlias = dict[frozenset[Variable], Variable]


def get_strongly_connected_components(graph: NxMixedGraph) -> set[frozenset[Variable]]:
    r"""Return the strongly-connected components for a graph.

    The strongly connected component of $v$ in $G$ is defined to be: $\text{Sc}^{G}(v):=
    \text{Anc}^{G}(v)\cap \text{Desc}^{G}(v)$.

    :param graph: The corresponding graph.

    :returns: A set of frozen sets of variables comprising $\text{Sc}^{G}(v)$ for all
        vertices $v$.
    """
    return {frozenset(component) for component in nx.strongly_connected_components(graph.directed)}


def get_vertex_consolidated_district(
    graph: Annotated[NxMixedGraph, InPaperAs("G")], vertex: Annotated[Variable, InPaperAs("v")]
) -> Annotated[frozenset[Variable], InPaperAs(r"\text{Cd}^{G}(v)")]:
    r"""Return the consolidated district for a single vertex in a graph.

    See Definition 9.1 of [forré20a]_.

    Let $G$ be a directed mixed graph (DMG) with set of nodes $V$. Let $v \in V$. The
    consolidated district $\text{Cd}^{G}(v)$ of $v$ in $G$ is given by all nodes $w \in
    V$ for which there exist $k \ge 1$ nodes $(v_1,\dots,v_k)$ in $G$ such that $v_1 =
    v, v_k = w$ and for $i = 2,\dots\,k$ we have that the bidirected edge $v_{i-1}
    \leftrightarrow v_i$ is in $G$ or that $v_i \in \text{Sc}^{G}(v_{i-1})$. For $B
    \subseteq V$ we write $\text{Cd}^{G}(B) := \bigcup_{v\in B}\text{Cd}^{G}(v)$. Let
    $\mathcal{CD}(G)$ be the set of consolidated districts of $G$.

    (This function retrieves the consolidated district for $v$, not $B$.)

    :param graph: The corresponding graph.
    :param vertex: The vertex for which the consolidated district is to be retrieved.

    :returns: The set of variables comprising $\text{Cd}^{G}(v)$.
    """
    # Strategy: (O(N^2))
    # 1. Get the strongly-connected component of every vertex in the graph, using the networkx function.
    # 2. Create a new graph that replaces every directed edge in a strongly-connected component with a bidirected edge.
    # 3. Get the district for the new graph that contains the target vertex in question using get_district().
    # 4. Return the resulting set.
    converted_graph = scc_to_bidirected(graph)
    result = converted_graph.get_district(vertex)
    return result


def get_consolidated_district(
    graph: Annotated[NxMixedGraph, InPaperAs("G")],
    vertices: Annotated[Collection[Variable], InPaperAs("B")],
) -> Annotated[set[Variable], InPaperAs(r"\text{Cd}^{G}(B)")]:
    r"""Return the consolidated districts for one or more vertices in a graph.

    See Definition 9.1 of [forré20a]_.

    Let $G$ be a directed mixed graph (DMG) with set of nodes $V$. Let $v \in V$. The
    consolidated district $\text{Cd}^{G}(v)$ of $v$ in $G$ is given by all nodes $w \in
    V$ for which there exist $k \ge 1$ nodes $(v_1,\dots,v_k)$ in $G$ such that $v_1 =
    v, v_k = w$ and for $i = 2,\dots\,k$ we have that the bidirected edge $v_{i-1}
    \leftrightarrow v_i$ is in $G$ or that $v_i \in \text{Sc}^{G}(v_{i-1})$. For $B
    \subseteq V$ we write $\text{Cd}^{G}(B) := \bigcup_{v\in B}\text{Cd}^{G}(v)$. Let
    $\mathcal{CD}(G)$ be the set of consolidated districts of $G$.

    Note: it's not entirely clear from the text whether the return value is meant to be
    a set of sets of vertices or just a set of vertices. We return a set of vertices in
    order for the notation to be consistent with Notation 9.4 part 3 and Remark 9.5: in
    Remark 9.5, the function $\text{Anc}^{G}$ only makes sense when called on a set of
    vertices, not a set of sets of vertices.

    :param graph: The corresponding graph.
    :param vertices: The vertices for which the consolidated district is to be
        retrieved.

    :returns: The set of consolidated districts for the variables in $B$.
    """
    return {node for district in get_unique_districts(graph, vertices) for node in district}


def get_unique_districts(
    graph: NxMixedGraph, vertices: Collection[Variable]
) -> set[frozenset[Variable]]:
    """Get all unique districts for the given vertex (some might overlap)."""
    # 1. Get the strongly-connected component of every vertex in the graph, using the networkx function.
    # 2. Create a new graph that replaces every directed edge in a strongly-connected component with a bidirected edge.
    # 3. Get all the consolidated districts.
    # 4. Return the union of all of them as one set.
    converted_graph = scc_to_bidirected(graph)
    districts: set[frozenset[Variable]] = set()
    for vertex in vertices:
        district = converted_graph.get_district(vertex)
        districts.add(district)
    return districts


def get_graph_consolidated_districts(
    graph: Annotated[NxMixedGraph, InPaperAs("G")],
) -> Annotated[set[frozenset[Variable]], InPaperAs(r"\mathcal{CD}(G)")]:
    r"""Return the set of all consolidated districts in a graph.

    See Definition 9.1 of [forré20a]_.

    Let $G$ be a directed mixed graph (DMG) with set of nodes $V$. Let $v \in V$. The
    consolidated district $\text{Cd}^{G}(v)$ of $v$ in $G$ is given by all nodes $w \in
    V$ for which there exist $k \ge 1$ nodes $(v_1,\dots,v_k)$ in $G$ such that $v_1 =
    v, v_k = w$ and for $i = 2,\dots\,k$ we have that the bidirected edge $v_{i-1}
    \leftrightarrow v_i$ is in $G$ or that $v_i \in \text{Sc}^{G}(v_{i-1})$. For $B
    \subseteq V$ we write $\text{Cd}^{G}(B) := \bigcup_{v\in B}\text{Cd}^{G}(v)$. Let
    $\mathcal{CD}(G)$ be the set of consolidated districts of $G$.

    :param graph: The corresponding graph.

    :returns: The set of consolidated districts for the graph.
    """
    # 1. Get the strongly-connected component of every vertex in the graph, using the networkx function.
    # 2. Create a new graph that replaces every directed edge in a strongly-connected component with a bidirected edge.
    # 3. Get each consolidated district as a frozen set.
    # 4. Return all of them as a set of frozen sets.
    return get_unique_districts(graph, graph.nodes())


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

    return new_graph


def get_apt_order(graph: NxMixedGraph) -> list[Variable]:
    r"""Return one possible assembling pseudo-topological order ("apt-order") for the vertices in a graph.

    See Definition 9.2 of [forré20a]_.

    Let $G$ be a directed mixed graph (DMG) with set of nodes $V$. An assembling
    pseudo-topological order (apt-order) of $G$ is a total order $\lt$ on $V$ with the
    following two properties:

    1. For every $v, w \in V$ we have:

       $w \in \text{Anc}^{G}(v) \backslash \text{Sc}^{G}(v) \Longrightarrow w \lt v$.

    2. For every $v_1, v_2, w \in V$ we have:

           $v_2 \in \text{Sc}^{G}(v_1) \land(v_1 \le w \le v_2) \Longrightarrow w \in
           \text{Sc}^{G}(v_1)$.

    :param graph: The corresponding graph.

    :returns: An apt-order for the vertices in $G$.
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
    new_graph, node_to_component = simplify_strongly_connected_components(graph)
    components = [sorted(node_to_component[v]) for v in new_graph.topological_sort()]
    nodes = [node for component in components for node in component]
    return nodes


def simplify_strongly_connected_components(
    graph: NxMixedGraph,
) -> tuple[NxMixedGraph, dict[Variable, frozenset[Variable]]]:
    r"""Reduce each strongly-connected component in a directed graph to a single vertex.

    This is a helper function for generating the assembling pseudo-topological order for
    a graph.

    Get the strongly-connected components and replace each one with a single vertex. An
    edge going into or out of the strongly-connected component becomes an edge going
    into or out of the representative vertex.

    :param graph: The input graph.

    :returns: The simplified graph and a dictionary mapping vertices representing
        strongly-connected components in the new graph to the vertices in each
        strongly-connected component in the original graph.
    """
    comp_to_rep_node: ComponentToNode = {}
    node_to_component: NodeToComponent = {}
    representative_node_to_component: NodeToComponent = {}

    for component in get_strongly_connected_components(graph):
        representative_node = min(component)
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
            # this happens when two nodes in the same strongly connected component have
            # an undirected edge between them and the edge is skipped during simplification.
            continue
        undirected.add((comp_to_rep_node[u_component], comp_to_rep_node[v_component]))
        # If we add both (u,v) and (v,u), that will go away when the actual graph gets
        # produced, so there's no need for a test

    new_graph = NxMixedGraph.from_edges(directed=directed, undirected=undirected)
    return new_graph, representative_node_to_component


def is_apt_order(candidate_order: list[Variable], graph: NxMixedGraph) -> bool:
    r"""Verify that a list of vertices is a possible assembling pseudo-topological order ("apt-order") for a graph.

    See Definition 9.2 of [forré20a]_.

    Let $G$ be a directed mixed graph (DMG) with set of nodes $V$. An assembling
    pseudo-topological order (apt-order) of $G$ is a total order $\lt$ on $V$ with the
    following two properties:

    1. For every $v, w \in V$ we have:

       $w \in \text{Anc}^{G}(v) \backslash \text{Sc}^{G}(v) \Longrightarrow w \lt v$.

    2. For every $v_1, v_2, w \in V$ we have:

           $v_2 \in \text{Sc}^{G}(v_1) \land(v_1 \le w \le v_2) \Longrightarrow w \in
           \text{Sc}^{G}(v_1)$.

    :param candidate_order: The candidate apt-order.
    :param graph: The corresponding graph.

    :returns: True if the candidate apt-order is a possible apt-order for the graph,
        False otherwise.
    """
    # first check - validate inputs
    _validate_apt_order_inputs(candidate_order, graph)

    # second check - get the strongly connected components
    sccs = get_strongly_connected_components(graph)

    # third check - check both conditions from Definition 9.2
    return _check_ancestors_are_prior_to_non_scc_descendants(
        candidate_order, graph, sccs
    ) and _check_members_of_scc_are_consecutive(candidate_order, sccs)


def _validate_apt_order_inputs(candidate_order: list[Variable], graph: NxMixedGraph) -> None:
    r"""Validate inputs for is_apt_order function.

    Definition 9.2 requires apt-order to be a total order $\lt$ on $V$. This function
    ensures the input satisfies the basic requirements of a total order: all vertices
    present exactly once.

    This function checks:

    1. All nodes in order exist in the graph.
    2. All nodes in the graph are present in order.
    3. No duplicate nodes in order.

    :param candidate_order: The candidate apt-order.
    :param graph: The corresponding graph.

    :raises ValueError: If order is invalid.
    """
    order_set = set(candidate_order)  # converting to a set for easier checking
    graph_nodes = set(graph.nodes())  # set of nodes in the graph

    # checking for nodes in order but not in the graph
    extra_nodes = order_set - graph_nodes
    if extra_nodes:
        raise ValueError(f"Order contains nodes that are not in graph: {extra_nodes}")

    # check for nodes in graph but not in order
    missing_nodes = graph_nodes - order_set
    if missing_nodes:
        raise ValueError(f"Order is missing nodes from the graph: {missing_nodes}")

    # checking to see if there are duplicates in order
    if len(candidate_order) != len(order_set):
        raise ValueError("Order contains duplicate nodes.")


def _check_ancestors_are_prior_to_non_scc_descendants(
    candidate_order: list[Variable], graph: NxMixedGraph, components: set[frozenset[Variable]]
) -> bool:
    r"""Check Condition 1 from Definition 9.2 of [forré20a]_.

    For every v, w ∈ V: w ∈ Anc^G(v) Sc^G(v) ⟹ w < v

    This verifies that ancestors outside a node's SCC appear before that node in the
    order. In other words: you can't have a node appear before its non-SCC ancestors.

    :param candidate_order: The candidate apt-order (list of variables).
    :param graph: The corresponding graph.
    :param components: Set of strongly connected components (each is a frozenset of
        variables).

    :returns: True if the ancestry constraint is satisfied, False otherwise.
    """
    node_to_index = {node: index for index, node in enumerate(candidate_order)}
    node_to_component = {node: component for component in components for node in component}

    return not any(
        (
            # Check if w is in Anc^G(v) \ Sc^G(v) - which would mean w is an ancestor but not in the same SCC
            node_to_component[w] != node_to_component[v]
            # then the constraint requires w < v in the order
            # in the order, this means index of w < index of v
            and node_to_index[w] >= node_to_index[v]
        )
        for v in graph.nodes()
        for w in graph.ancestors_inclusive(v)
    )


def _check_members_of_scc_are_consecutive(
    candidate_order: list[Variable], components: set[frozenset[Variable]]
) -> bool:
    r"""Check Condition 2 from Definition 9.2 of [forré20a]_.

    For every v₁, v₂, w ∈ V: v₂ ∈ Sc^G(v₁) ∧ (v₁ ≤ w ≤ v₂) ⟹ w ∈ Sc^G(v₁)

    Translation: If v₂ is in same SCC as v₁, and w is between them in the order, then w
    must also be in that SCC.

    In other words: Nodes in the same SCC (feedback loop) must appear consecutively in
    the order with no nodes from other SCCs in between.

    :param candidate_order: The order to validate as a potential apt-order.
    :param components: Set of strongly connected components (each is a frozenset of
        variables).

    :returns: True if all SCCs are consecutive, False otherwise.
    """

    def _get_target_nodes(component: frozenset[Variable]) -> Iterable[Variable]:
        # find where each node in this SCC appears in the order
        positions = {candidate_order.index(node) for node in component}

        # find the first and last occurrence of nodes from this SCC in the order
        min_pos = min(positions)
        max_pos = max(positions)

        # check all positions between min_pos and max_pos which is inclusive
        for pos in range(min_pos, max_pos + 1):
            yield candidate_order[pos]

    return not any(
        node not in component
        for component in components
        if len(component) > 1
        for node in _get_target_nodes(component)
    )
