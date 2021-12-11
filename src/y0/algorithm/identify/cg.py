from itertools import combinations
from typing import Collection, Tuple

from y0.dsl import (
    CounterfactualVariable,
    Intervention,
    P,
    Probability,
    Variable,
    _get_treatment_variables,
)
from y0.graph import NxMixedGraph

__all__ = ["make_counterfactual_graph", "has_same_function", "forget_value"]


def has_same_parents(graph: NxMixedGraph, a: Variable, b: Variable) -> bool:
    """Check if all parents of the two nodes are the same.

    This is true if the set of directed parents are the same and either there
    exists a bidirected edge between the two nodes or there exists no bidirected
    edges for either node.
    """
    return (set(graph.directed.predecessors(a)) == set(graph.directed.predecessors(b))) and (
        graph.undirected.has_edge(a, b)
        or (0 == len(graph.undirected.edges(a)) == len(graph.undirected.edges(b)))
    )


def has_same_function(node1: Variable, node2: Variable) -> bool:
    return node1.name == node2.name


def forget_value(node: Variable) -> Variable:
    if isinstance(node, Intervention):
        raise TypeError
    if isinstance(node, CounterfactualVariable):
        return CounterfactualVariable(name=node.name, interventions=node.interventions, star=None)
    else:
        return Variable(name=node.name, star=None)


def get_worlds(query: Probability) -> Collection[Collection[Variable]]:
    # is sorting necessary? why not just return a set/frozenset?
    return sorted(
        sorted(_get_treatment_variables(variable.get_variables()), key=str)
        for variable in query.get_variables()
        if isinstance(variable, CounterfactualVariable)
    )


def lemma_24(pw_graph: NxMixedGraph, node1: Variable, node2: Variable) -> bool:
    r"""Check if two nodes in a parallel worlds graph are equivalent.

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
    return has_same_function(node1, node2) and has_same_parents(pw_graph, node1, node2)


def lemma_25(graph: NxMixedGraph, node1: Variable, node2: Variable) -> NxMixedGraph:
    r"""Merge node1 and node2 and return the reduced graph and query.

    Let :math:`M_\mathbf{x}` be a submodel derived from :math:`M` with set :math:`\mathbf{Z}`
    observed to attain values :math:`\mathbf{z}`, such that Lemma 24 holds for :math:`\alpha`;
    :math:`\beta`. Let :math:`M'` be a causal model obtained from :math:`M` by merging
    :math:`\alpha`; :math:`\beta` into a new node :math:`\omega`, which inherits all parents
    and the functional mechanism of :math:`\alpha`. All children of
    :math:`\alpha`; :math:`\beta` in :math:`M'` become children of :math:`\omega`. Then
    :math:`M_\mathbf{x},  M'_\mathbf{x} agree on any distribution consistent with :math:`z`
    being observed.
    """
    if isinstance(node1, CounterfactualVariable) and not isinstance(node2, CounterfactualVariable):
        node1, node2 = node2, node1
    elif not isinstance(node1, CounterfactualVariable) and isinstance(
        node2, CounterfactualVariable
    ):
        pass
    else:
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
    graph: NxMixedGraph, query: Probability
) -> Tuple[NxMixedGraph, Probability]:
    """Make counterfactual graph"""
    worlds = get_worlds(query)
    pw_graph = make_parallel_worlds_graph(graph, worlds)
    new_query_variables = set(query.children)
    cf_graph = NxMixedGraph.from_edges(
        nodes=pw_graph.nodes(),
        directed=pw_graph.directed.edges(),
        undirected=pw_graph.undirected.edges(),
    )
    for node in graph.topological_sort():
        for intervention in worlds:
            node_at_intervention = node @ intervention
            if (
                (node in cf_graph.nodes())
                and (node_at_intervention in cf_graph.nodes())
                and lemma_24(cf_graph, node, node @ intervention)
            ):
                cf_graph = lemma_25(cf_graph, node, node_at_intervention)
                if node_at_intervention in new_query_variables:
                    new_query_variables = (new_query_variables - {node_at_intervention}) | {node}

        if len(worlds) > 1:
            for intervention1, intervention2 in combinations(worlds, 2):
                if (
                    (node @ intervention1 in cf_graph.nodes())
                    and (node @ intervention2 in cf_graph.nodes())
                    and lemma_24(cf_graph, node @ intervention1, node @ intervention2)
                ):
                    cf_graph = lemma_25(cf_graph, node @ intervention1, node @ intervention2)
                    if node @ intervention2 in new_query_variables:
                        new_query_variables = new_query_variables - {node @ intervention2} | {
                            node @ intervention1
                        }
    return (cf_graph.subgraph(cf_graph.ancestors_inclusive([forget_value(v) for v in new_query_variables])),
            P(new_query_variables))


def make_parallel_worlds_graph(
    graph: NxMixedGraph, worlds: Collection[Collection[Variable]]
) -> NxMixedGraph:
    """Make Parallel worlds graph"""
    combined_worlds = [make_world_graph(graph, world) for world in worlds]
    return combine_worlds(graph, combined_worlds, worlds)


def make_world_graph(graph: NxMixedGraph, treatments: Collection[Variable]) -> NxMixedGraph:
    """Make one parallel world based on interventions specified"""
    treatment_variables = [Variable(t.name) for t in treatments]
    world_graph = graph.remove_in_edges(treatment_variables)
    return NxMixedGraph.from_edges(
        nodes=[node.intervene(treatments) for node in world_graph.nodes()],
        directed=[
            (u.intervene(treatments), v.intervene(treatments))
            for u, v in world_graph.directed.edges()
        ],
        undirected=[
            (u.intervene(treatments), v.intervene(treatments))
            for u, v in world_graph.undirected.edges()
        ],
    )


def combine_worlds(
    graph: NxMixedGraph,
    combined_worlds: Collection[NxMixedGraph],
    worlds: Collection[Collection[Variable]],
) -> NxMixedGraph:
    """Stitch together parallel worlds through the magic of bidirected edges"""
    # get all the undirected edges in all the parallel worlds
    undirected = [(u, v) for world in combined_worlds for u, v in world.undirected.edges()]
    # Stitch together counterfactual variables with observed variables
    undirected += [
        (u, u @ treatments)
        for treatments in worlds
        for u in graph.nodes()
        # Don't add an edge if a variable is intervened on
        if (u not in treatments) and (~u not in treatments)
    ]
    undirected += [
        (u, v @ treatments)
        for treatments in worlds
        for u in graph.nodes()
        for v in graph.undirected.neighbors(u)
        # Don't add an edge if a variable is intervened on
        if (v not in treatments) and (~v not in treatments)
    ]
    # Stitch together variables from different counterfactual worlds
    if len(worlds) > 1:
        undirected += [
            (u @ treatments_from_world_1, u @ treatments_from_world_2)
            for treatments_from_world_1, treatments_from_world_2 in combinations(worlds, 2)
            for u in graph.nodes()
            # Don't add an edge if a variable is intervened on in either world.
            if (u not in treatments_from_world_1)
            and (u not in treatments_from_world_2)
            and (~u not in treatments_from_world_1)
            and (~u not in treatments_from_world_2)
        ]
        undirected += [
            (u @ treatments_from_world_1, v @ treatments_from_world_2)
            for treatments_from_world_1, treatments_from_world_2 in combinations(worlds, 2)
            for u in graph.nodes()
            for v in graph.undirected.neighbors(u)
            # Don't add an edge if a variable is intervened on in either world.
            if (u not in treatments_from_world_1)
            and (v not in treatments_from_world_2)
            and (~u not in treatments_from_world_1)
            and (~v not in treatments_from_world_2)
        ]
    return NxMixedGraph.from_edges(
        nodes=list(graph.nodes())
        + [node for pw_graph in combined_worlds for node in pw_graph.nodes()],
        directed=list(graph.directed.edges())
        + [(u, v) for pw_graph in combined_worlds for u, v in pw_graph.directed.edges()],
        undirected=list(graph.undirected.edges()) + undirected,
    )
