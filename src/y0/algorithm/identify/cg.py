# -*- coding: utf-8 -*-

"""Utilities for parallel world graphs and counterfactual graphs."""

from itertools import combinations
from typing import Collection, FrozenSet, Iterable, List, Optional, Set, Tuple

from y0.dsl import CounterfactualVariable, Event, Intervention, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "has_same_function",
    "has_same_parents",
    "extract_interventions",
    "is_pw_equivalent",
    "merge_pw",
    "make_counterfactual_graph",
    "make_parallel_worlds_graph",
    "combine_worlds",
]


class World(FrozenSet[Intervention]):
    """A set of interventions corresponding to a "world"."""

    def __contains__(self, item) -> bool:
        if not isinstance(item, Intervention):
            raise TypeError(
                f"can not check if non-intervention is in a world: ({type(item)}) {item}"
            )
        return super().__contains__(item)


Worlds = Set[World]


def has_same_parents(graph: NxMixedGraph, a: Variable, b: Variable) -> bool:
    """Check if all parents of the two nodes are the same.

    :param graph: An ADMG
    :param a: A variable in the ADMG
    :param b: Another variable in the ADMG
    :returns:
        True if the set of directed parents are the same and either there
        exists a bidirected edge between the two nodes or there exists no bidirected
        edges for either node.
    """
    return (set(graph.directed.predecessors(a)) == set(graph.directed.predecessors(b))) and (
        graph.undirected.has_edge(a, b)
        or (0 == len(graph.undirected.edges(a)) == len(graph.undirected.edges(b)))
    )


def has_same_function(node1: Variable, node2: Variable) -> bool:
    """Check if the two nodes have the same name."""
    # TODO maybe check if the get_base() is the same instead?
    return node1.name == node2.name


def extract_interventions(variables: Iterable[Variable]) -> Worlds:
    """Extract the set of interventions for each counterfactual variable that corresponds to a world."""
    # is sorting necessary? why not just return a set/frozenset?
    # Yes, because otherwise different counterfactual graphs will be created each time.
    return set(
        World(variable.interventions)
        for variable in variables
        if isinstance(variable, CounterfactualVariable)
    )


def is_pw_equivalent(graph: NxMixedGraph, node1: Variable, node2: Variable) -> bool:
    r"""Check if two nodes in a parallel worlds graph are equivalent.

    :param graph: A parallel worlds graph
    :param node1: A node in the graph
    :param node2: Another node in the graph
    :returns: If the two nodes are equivalent under the parallel worlds assumption

    Let :math:`M` be a model inducing :math:`G` containing variables
    :math:`\alpha`, :math:`\beta` with the following properties:

    * :math:`\alpha` and :math:`\beta` have the same domain of values.
    * There is a bijection :math:`f` from :math:`Pa(\alpha)` to :math:`Pa(\beta)`
      such that a parent  :math:`\gamma` and :math:`f(\gamma)` have the same domain
      of values.
    * The functional mechanisms of :math:`\alpha` and :math:`\beta` are the same
      (except whenever the function for :math:`\alpha` uses the parent :math:`\gamma`,
      the corresponding function for :math:`\beta` uses :math:`f(\gamma)`).

    Assume an observable variable set :math:`\mathbf{Z}` was observed to attain values
    :math:`z` in :math:`M_\mathbf{x}` , the submodel obtained from :math:`M` by forcing
    another observable variable set :math:`\mathbf{X}` to attain values :math:`\mathbf{x}`.
    Assume further that for each  :math:`\gamma \in Pa(\alpha)`, either
    :math:`f(\gamma) =  \gamma`, or  :math:`\gamma` and :math:`f(\gamma)` attain the
    same values (whether by observation or intervention). Then :math:`\alpha` and
    :math:`\beta` are the same random variable in :math:`M_\mathbf{x}` with observations
    :math:`\mathbf{z}`.
    """
    # Rather than all n choose 2 combinations, we can restrict ourselves to the original
    # graph variables and their counterfactual versions
    return has_same_function(node1, node2) and has_same_parents(graph, node1, node2)


def merge_pw(graph: NxMixedGraph, node1: Variable, node2: Variable) -> NxMixedGraph:
    r"""Merge node1 and node2 and return the reduced graph and query.

    Let :math:`M_\mathbf{x}` be a submodel derived from :math:`M` with set :math:`\mathbf{Z}`
    observed to attain values :math:`\mathbf{z}`, such that Lemma 24 holds for :math:`\alpha`;
    :math:`\beta`. Let :math:`M'` be a causal model obtained from :math:`M` by merging
    :math:`\alpha`; :math:`\beta` into a new node :math:`\omega`, which inherits all parents
    and the functional mechanism of :math:`\alpha`. All children of
    :math:`\alpha`; :math:`\beta` in :math:`M'` become children of :math:`\omega`. Then
    :math:`M_\mathbf{x},  M'_\mathbf{x} agree on any distribution consistent with :math:`z`
    being observed.

    :param graph: A parallel worlds graph
    :param node1: A node in the graph
    :param node2: Another node in the graph
    :returns: A reduced graph
    """
    # If a we are going to merge two nodes, we want to keep the factual variable.
    if isinstance(node1, CounterfactualVariable) and not isinstance(node2, CounterfactualVariable):
        node1, node2 = node2, node1
    elif not isinstance(node1, CounterfactualVariable) and isinstance(
        node2, CounterfactualVariable
    ):
        pass
    else:  # both are counterfactual or both are factual, so keep the variable with the lower name
        node1, node2 = sorted([node1, node2])
    directed = [(u, v) for u, v in graph.directed.edges() if node2 not in (u, v)]
    directed += [(node1, v) for u, v in graph.directed.edges() if node2 == u]
    # directed += [(u, node1) for u, v in graph.directed.edges() if node2 == v]
    undirected = [frozenset({u, v}) for u, v in graph.undirected.edges() if node2 not in (u, v)]
    undirected += [
        frozenset({node1, v}) for u, v in graph.undirected.edges() if node2 == u and node1 != v
    ]
    undirected += [
        frozenset({u, node1}) for u, v in graph.undirected.edges() if node2 == v and node1 != u
    ]
    return NxMixedGraph.from_edges(
        nodes=[node for node in graph.nodes() if node != node2],
        directed=list(set(directed)),
        undirected=[(u, v) for u, v in set(undirected)],
    )


def make_counterfactual_graph(
    graph: NxMixedGraph, event: Event
) -> Tuple[NxMixedGraph, Optional[Event]]:
    """Make counterfactual graph."""
    worlds = extract_interventions(event)
    pw_graph = make_parallel_worlds_graph(graph, worlds)
    new_event = dict(event)
    cf_graph = NxMixedGraph.from_edges(
        nodes=pw_graph.nodes(),
        directed=pw_graph.directed.edges(),
        undirected=pw_graph.undirected.edges(),
    )
    for node in graph.topological_sort():
        for world in worlds:
            node_at_interventions = node @ world
            if (
                (node in cf_graph.nodes())
                and (node_at_interventions in cf_graph.nodes())
                and is_pw_equivalent(cf_graph, node, node_at_interventions)
            ):
                cf_graph = merge_pw(cf_graph, node, node_at_interventions)
                if (
                    (node in new_event)
                    and (node_at_interventions in new_event)
                    and (new_event[node] != new_event[node_at_interventions])
                ):
                    return cf_graph, None
                if node_at_interventions in new_event:
                    new_event[node] = new_event[node_at_interventions]
                    new_event.pop(node_at_interventions, None)

        if len(worlds) > 1:
            for intervention1, intervention2 in combinations(worlds, 2):
                node_at_intervention1 = node @ intervention1
                node_at_intervention2 = node @ intervention2

                if (
                    (node_at_intervention1 in cf_graph.nodes())
                    and (node_at_intervention2 in cf_graph.nodes())
                    and is_pw_equivalent(cf_graph, node_at_intervention1, node_at_intervention2)
                ):
                    cf_graph = merge_pw(cf_graph, node_at_intervention1, node_at_intervention2)
                    if (
                        (node_at_intervention1 in new_event)
                        and (node_at_intervention2 in new_event)
                        and (new_event[node_at_intervention1] != new_event[node_at_intervention2])
                    ):
                        # TODO needs test case
                        return cf_graph, None

                    if node_at_intervention2 in new_event:
                        # TODO needs test case
                        new_event[node_at_intervention1] = new_event[node_at_intervention2]
                        new_event.pop(node_at_intervention2, None)
    rv_graph = cf_graph.subgraph(cf_graph.ancestors_inclusive(new_event))
    # rv_graph = rv_graph.remove_nodes_from(
    #    node for node in rv_graph.nodes() if is_self_intervened(node)
    # )
    # rv_graph = remove_redundant_interventions(rv_graph)
    return rv_graph, new_event


def make_parallel_worlds_graph(graph: NxMixedGraph, worlds: Worlds) -> NxMixedGraph:
    """Make a parallel worlds graph.

    :param graph: A normal graph
    :param worlds: A set of sets of treatments
    :returns: A combine parallel world graph
    """
    world_graphs: List[NxMixedGraph] = [graph.intervene(world) for world in worlds]
    return combine_worlds(graph, world_graphs, worlds)


def _check_world(*, world: World, node: Variable) -> bool:
    return (node not in world) and (~node not in world)


def _stitch_1(
    graph: NxMixedGraph, worlds: Worlds
) -> List[Tuple[CounterfactualVariable, CounterfactualVariable]]:
    # TODO high level documentation
    return [
        (node, node @ world)
        for world in worlds
        for node in graph.nodes()
        # Don't add an edge if a variable is intervened on
        # TODO make this check part of the World class during pair coding
        if _check_world(world=world, node=node)
    ]


def _stitch_2(
    graph: NxMixedGraph, worlds: Worlds
) -> List[Tuple[CounterfactualVariable, CounterfactualVariable]]:
    # TODO high level documentation
    return [
        (u, v @ world)
        for world in worlds
        for u in graph.nodes()
        for v in graph.undirected.neighbors(u)
        # Don't add an edge if a variable is intervened on
        if _check_world(world=world, node=v)
    ]


def _stitch_3(
    graph: NxMixedGraph, worlds: Worlds
) -> List[Tuple[CounterfactualVariable, CounterfactualVariable]]:
    return [
        (u @ world_1, u @ world_2)
        for world_1, world_2 in combinations(worlds, 2)
        for u in graph.nodes()
        # Don't add an edge if a variable is intervened on in either world.
        if _check_world(world=world_1, node=u) and _check_world(world=world_2, node=u)
    ]


def _stitch_4(
    graph: NxMixedGraph, worlds: Worlds
) -> List[Tuple[CounterfactualVariable, CounterfactualVariable]]:
    return [
        (u @ world_1, v @ world_2)
        for world_1, world_2 in combinations(worlds, 2)
        for u in graph.nodes()
        for v in graph.undirected.neighbors(u)
        # Don't add an edge if a variable is intervened on in either world.
        if _check_world(node=u, world=world_1) and _check_world(node=v, world=world_2)
    ]


def combine_worlds(
    graph: NxMixedGraph,
    world_graphs: Collection[NxMixedGraph],
    worlds: Worlds,
) -> NxMixedGraph:
    """Stitch together parallel worlds through the magic of bidirected edges."""
    # get all the undirected edges in all the parallel worlds
    undirected = [(u, v) for world_graph in world_graphs for u, v in world_graph.undirected.edges()]

    # Stitch together counterfactual variables with observed variables
    undirected.extend(_stitch_1(graph, worlds))
    undirected.extend(_stitch_2(graph, worlds))

    # Stitch together variables from different counterfactual worlds
    if len(worlds) > 1:
        undirected.extend(_stitch_3(graph, worlds))
        undirected.extend(_stitch_4(graph, worlds))

    nodes = [
        *graph.nodes(),
        *(node for world_graph in world_graphs for node in world_graph.nodes()),
    ]
    directed = [
        *graph.directed.edges(),
        *((u, v) for world_graph in world_graphs for u, v in world_graph.directed.edges()),
    ]
    return NxMixedGraph.from_edges(
        nodes=nodes,
        directed=directed,
        undirected=list(graph.undirected.edges()) + undirected,
    )
