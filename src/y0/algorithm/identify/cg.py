# -*- coding: utf-8 -*-

"""Utilities for parallel world graphs and counterfactual graphs."""

from itertools import combinations
from typing import FrozenSet, Iterable, Optional, Set, Tuple, cast

from y0.dsl import (
    CounterfactualVariable,
    Event,
    Intervention,
    Variable,
    _variable_sort_key,
)
from y0.graph import NxMixedGraph

__all__ = [
    "has_same_function",
    "extract_interventions",
    "is_pw_equivalent",
    "merge_pw",
    "make_counterfactual_graph",
    "make_parallel_worlds_graph",
    "is_not_self_intervened",
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


def has_same_confounders(graph: NxMixedGraph, a: Variable, b: Variable) -> bool:
    """Check if all confounders of the two nodes are the same."""
    no_undirected_edges = (
        0 == len(list(graph.undirected.edges(a))) == len(list(graph.undirected.edges(b)))
    )
    return graph.undirected.has_edge(a, b) or no_undirected_edges


def has_same_function(node1: Variable, node2: Variable) -> bool:
    """Check if the two nodes have the same functional mechanism."""
    return node1.get_base() == node2.get_base() and is_not_self_intervened(
        node1
    ) == is_not_self_intervened(node2)


def nodes_attain_same_value(graph: NxMixedGraph, event: Event, a: Variable, b: Variable) -> bool:
    """Check if the two nodes attain the same value."""
    if a == b:
        return True
    if not has_same_confounders(graph, a, b):
        return False
    elif a.get_base() != b.get_base():
        return False
    elif a in event and b in event:
        if event[a] != event[b]:
            return False  # D and D @ -d  events = {D: -d}
        else:
            return True
    elif a in event:
        if not isinstance(b, CounterfactualVariable) or event[a] not in b.interventions:
            return False
        else:
            return True
    elif b in event:
        if not isinstance(a, CounterfactualVariable) or event[b] not in a.interventions:
            return False
        else:
            return True
    elif isinstance(a, CounterfactualVariable) or isinstance(b, CounterfactualVariable):
        return False
    return True


def parents_attain_same_values(graph: NxMixedGraph, event: Event, a: Variable, b: Variable) -> bool:
    """Check if the parents of the nodes attain the same value."""
    if not has_same_confounders(graph, a, b):
        return False
    parents_a, parents_b = set(graph.directed.predecessors(a)), set(graph.directed.predecessors(b))
    if parents_a == parents_b:
        return True
    remainder_a, remainder_b = parents_a - parents_b, parents_b - parents_a
    if len(remainder_a) != len(remainder_b):
        return False
    return all(
        nodes_attain_same_value(graph, event, parent_a, parent_b)
        for parent_a, parent_b in zip(
            sorted(remainder_a, key=lambda x: x.get_base()),
            sorted(remainder_b, key=lambda x: x.get_base()),
        )
    )


def is_not_self_intervened(node: Variable) -> bool:
    """Check if the node is not self-intervened."""
    return not isinstance(node, CounterfactualVariable) or (
        +(node.get_base()) not in node.interventions
        and -(node.get_base()) not in node.interventions
    )


def extract_interventions(variables: Iterable[Variable]) -> Worlds:
    """Extract the set of interventions for each counterfactual variable that corresponds to a world."""
    return set(
        World(variable.interventions)
        for variable in variables
        if isinstance(variable, CounterfactualVariable)
    )


def is_pw_equivalent(graph: NxMixedGraph, event: Event, node1: Variable, node2: Variable) -> bool:
    r"""Check if two nodes in a parallel worlds graph are equivalent.

    :param graph: A parallel worlds graph
    :param event: A dictionary of variables and variables values
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

    Lemma 24 from Ilya Shpitser and Judea Pearl. 2008.
    Complete Identification Methods for the Causal Hierarchy.
    Journal of Machine Learning Research (2008).
    """
    # Rather than all n choose 2 combinations, we can restrict ourselves to the original
    # graph variables and their counterfactual versions
    assert (node1 in graph.nodes()) and (node2 in graph.nodes()), "Nodes must be in the graph"
    return (
        has_same_function(node1, node2)
        and parents_attain_same_values(graph, event, node1, node2)
        and nodes_have_same_domain_of_values(graph, event, node1, node2)
    )


def nodes_have_same_domain_of_values(
    graph: NxMixedGraph, event: Event, a: Variable, b: Variable
) -> bool:
    """Check if the nodes have the same domain of values."""
    if not has_same_confounders(graph, a, b):
        return False
    if a.get_base() != b.get_base():
        return False
    if is_not_self_intervened(a) and is_not_self_intervened(b):
        return True
    if is_not_self_intervened(a) or is_not_self_intervened(b):
        return False
    if value_of_self_intervention(a) == value_of_self_intervention(b):
        return True
    return False


def value_of_self_intervention(a: Variable) -> Optional[Intervention]:
    """Get the value of the self-intervention."""
    if not isinstance(a, CounterfactualVariable):
        return None
    base = a.get_base()
    if +base in a.interventions:
        return cast(Intervention, +base)
    elif -base in a.interventions:
        return cast(Intervention, -base)
    return None


def merge_pw(
    graph: NxMixedGraph, node1: Variable, node2: Variable
) -> Tuple[NxMixedGraph, Variable, Variable]:
    r"""Merge node1 and node2 and return the reduced graph and query.

    :param graph: A parallel worlds graph
    :param node1: A node in the graph
    :param node2: Another node in the graph
    :returns: A reduced graph

    Let :math:`M_\mathbf{x}` be a submodel derived from :math:`M` with set :math:`\mathbf{Z}`
    observed to attain values :math:`\mathbf{z}`, such that Lemma 24 holds for :math:`\alpha`;
    :math:`\beta`. Let :math:`M'` be a causal model obtained from :math:`M` by merging
    :math:`\alpha`; :math:`\beta` into a new node :math:`\omega`, which inherits all parents
    and the functional mechanism of :math:`\alpha`. All children of
    :math:`\alpha`; :math:`\beta` in :math:`M'` become children of :math:`\omega`. Then
    :math:`M_\mathbf{x},  M'_\mathbf{x} agree on any distribution consistent with :math:`z`
    being observed.

    Lemma 25 from Ilya Shpitser and Judea Pearl. 2008.
    Complete Identification Methods for the Causal Hierarchy.
    Journal of Machine Learning Research (2008).
    """
    # If we are going to merge two nodes, we want to keep the factual variable.
    if isinstance(node1, CounterfactualVariable) and not isinstance(node2, CounterfactualVariable):
        node1, node2 = node2, node1
    elif not isinstance(node1, CounterfactualVariable) and isinstance(
        node2, CounterfactualVariable
    ):
        pass
    else:  # both are counterfactual or both are factual, so keep the variable with the lower name
        node1, node2 = sorted([node1, node2], key=_variable_sort_key)
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
    parents_of_node1 = [u for u, v in graph.directed.edges() if v == node1]
    parents_of_node2_not_node1 = [
        u for u, v in graph.directed.edges() if v == node2 and u not in parents_of_node1
    ]
    return (
        NxMixedGraph.from_edges(
            nodes=[
                node
                for node in graph.nodes()
                if node != node2 and node not in parents_of_node2_not_node1
            ],
            directed=list(set(directed)),
            undirected=[(u, v) for u, v in set(undirected)],
        ),
        node1,
        node2,
    )


def lemma_24_holds(
    cf_graph: NxMixedGraph, event: Event, node: Variable, node_at_interventions: Variable
) -> bool:
    r"""Check if Lemma 24 holds for the given nodes."""
    return (
        (node in cf_graph.nodes())
        and (node_at_interventions in cf_graph.nodes())
        and is_pw_equivalent(cf_graph, event, node, node_at_interventions)
    )


def is_inconsistent(event: Event, node: Variable, node_at_interventions: Variable) -> bool:
    r"""Check if the equivalant nodes (according to lemma 25) have been assigned different values in the event."""
    return (
        (node in event)
        and (node_at_interventions in event)
        and (event[node] != event[node_at_interventions])
    )


def update_event(event: Event, preferred_node: Variable, eliminated_node: Variable) -> Event:
    r"""Update the event to reflect the fact that the preferred node has been merged into the eliminated node."""
    if eliminated_node in event:
        event[preferred_node] = event[eliminated_node]
        del event[eliminated_node]
    return event


def make_counterfactual_graph(
    graph: NxMixedGraph, event: Event
) -> Tuple[NxMixedGraph, Optional[Event]]:
    r"""Make counterfactual graph.

    :param graph: A causal graph :math:`G`
    :param event: A conjunction of counterfactual events :math:`\gamma`
    :returns:
        A counterfactual graph and either a set of new events :math:`\gamma'`
        such that :math:`P(\gamma') = P(\gamma)` or None

    * Construct a submodel graph :math:`G_{\mathbf{x}_i}` for each action
      :math:`do(\mathbf{x}_i)` mentioned in the event :math:`\gamma`. Construct the
      parallel worlds graph :math:`G'` by having all such submodel graphs share their
      corresponding :math:`U` nodes
    * Let :math:`\pi` be a topological ordering of nodes in :math:`G'`, let :math:`\gamma' := \gamma`
    * Apply Lemmas 24 and 25, in order :math:`\pi`, to each observable pair of nodes :math:`\alpha, \beta` in :math:`G`.
      For each :math:`\alpha, \beta` that are the same, do:
        * Let :math:`G'` be modified as specified in Lemma 25
        * Modify :math:`\gamma'` by renaming all occurrences of :math:`\beta` to :math:`\alpha`.
        * If :math:`\mathbf{\alpha} \ne :math:`\mathbf{\beta}`, return :math:`G'`, None
    * Return :math:`(G', \gamma')`, where :math:`An(\gamma')` is the set of nodes in :math:`G'`
      ancestral to nodes corresponding to variables mentioned in :math:`\gamma'`.
    """
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
            if lemma_24_holds(cf_graph, new_event, node, node_at_interventions):
                cf_graph, preferred_node, eliminated_node = merge_pw(
                    cf_graph, node, node_at_interventions
                )
                if is_inconsistent(new_event, preferred_node, eliminated_node):
                    return cf_graph, None
                new_event = update_event(new_event, preferred_node, eliminated_node)
        if len(worlds) > 1:
            for intervention1, intervention2 in combinations(worlds, 2):
                node_at_intervention1 = node @ intervention1
                node_at_intervention2 = node @ intervention2

                if lemma_24_holds(
                    cf_graph, new_event, node_at_intervention1, node_at_intervention2
                ):
                    cf_graph, preferred_node, eliminated_node = merge_pw(
                        cf_graph, node_at_intervention1, node_at_intervention2
                    )
                    if is_inconsistent(new_event, node_at_intervention1, node_at_intervention2):
                        return cf_graph, None
                    new_event = update_event(new_event, preferred_node, eliminated_node)

    ancestors = cf_graph.ancestors_inclusive(new_event)
    rv_graph = cf_graph.subgraph(ancestors)
    return rv_graph, new_event


def node_not_an_intervention_in_world(world: World, node: Variable) -> bool:
    """Confirm that node is not an intervention in a given world."""
    if isinstance(node, (Intervention, CounterfactualVariable)):
        raise TypeError(
            "this shouldn't happen since the graph should not have interventions as nodes"
        )
    return (+node not in world) and (-node not in world)


def stitch_factual_and_dopplegangers(
    graph: NxMixedGraph, worlds: Worlds
) -> Set[Tuple[Variable, CounterfactualVariable]]:
    """Stitch together a node and its counterfactual doppleganger in each world."""
    return {
        (u, u @ world)
        for world in worlds
        for u in graph.nodes()
        if node_not_an_intervention_in_world(world, u)
    }


def stitch_factual_and_doppleganger_neighbors(
    graph: NxMixedGraph, worlds: Worlds
) -> Set[Tuple[Variable, CounterfactualVariable]]:
    """Stitch together a node with the dopplegangers of its neighbors in each world."""
    return {
        (u, v @ world)
        for world in worlds
        for u in graph.nodes()
        for v in graph.undirected.neighbors(u)
        # Don't add an edge if a variable is intervened on
        if node_not_an_intervention_in_world(world, v)
    }


def stitch_counterfactual_and_dopplegangers(
    graph: NxMixedGraph, worlds: Worlds
) -> Set[Tuple[CounterfactualVariable, CounterfactualVariable]]:
    """Stitch together a counterfactual variable with its doppelganger.

    Unless the counterfactual is intervened upon in one of the worlds.

    :param graph: A NxMixedGraph
    :param worlds: A set of frozensets of interventions
    :returns: A set of undirected edges
    """
    rv = {
        (u @ world_1, u @ world_2)
        for world_1, world_2 in combinations(worlds, 2)
        for u in graph.nodes()
        # Don't add an edge if a variable is intervened on in either world.
        if node_not_an_intervention_in_world(world_1, u)
        and node_not_an_intervention_in_world(world_2, u)
    }
    return _both_ways(rv)


def _both_ways(s):
    rv = set()
    for a, b in s:
        rv.add((b, a))
    return rv


def stitch_counterfactual_and_doppleganger_neighbors(
    graph: NxMixedGraph, worlds: Worlds
) -> Set[Tuple[CounterfactualVariable, CounterfactualVariable]]:
    """Stitch together a counterfactual variable with the dopplegangers of its neighbors in each world."""
    rv = {
        frozenset({u @ world_1, v @ world_2})
        for world_1, world_2 in combinations(worlds, 2)
        for u in graph.nodes()
        for v in graph.undirected.neighbors(u)
        # Don't add an edge if a variable is intervened on in either world.
        if node_not_an_intervention_in_world(world_1, u)
        and node_not_an_intervention_in_world(world_2, v)
    }
    return _both_ways(rv)


def stitch_counterfactual_and_neighbors(
    graph: NxMixedGraph, worlds: Worlds
) -> Set[Tuple[CounterfactualVariable, CounterfactualVariable]]:
    """Stitch together a counterfactual variable with its neighbors in each world."""
    rv = {
        (u @ world, v @ world)
        for world in worlds
        for u in graph.nodes()
        for v in graph.undirected.neighbors(u)
        # Don't add an edge if a variable is intervened on in either world.
        if node_not_an_intervention_in_world(world, u)
        and node_not_an_intervention_in_world(world, v)
    }
    return _both_ways(rv)


def _get_directed_edges(
    graph: NxMixedGraph, worlds: Worlds
) -> Set[Tuple[CounterfactualVariable, CounterfactualVariable]]:
    """Get the directed edges in the parallel worlds graph.

    Except for those where the target node was intervened upon.

    :param graph: A NxMixedGraph
    :param worlds: A set of frozensets of interventions
    :returns: A set of directed edges
    """
    return {
        (u @ world, v @ world)
        for world in worlds
        for u, v in graph.directed.edges()
        if node_not_an_intervention_in_world(world, v)
    }


def make_parallel_worlds_graph(
    graph: NxMixedGraph,
    worlds: Worlds,
) -> NxMixedGraph:
    """Make a parallel worlds graph.

    :param graph: A normal graph
    :param worlds: A set of sets of treatments
    :returns: A combine parallel world graph
    """
    # Get the undirected edges
    undirected: Set[Tuple[Variable, Variable]] = set()
    # get all the undirected edges in all the parallel worlds
    undirected |= stitch_counterfactual_and_neighbors(graph, worlds)
    # Stitch together factual variables with their dopplegangers in other worlds
    undirected |= stitch_factual_and_dopplegangers(graph, worlds)
    # Stitch together factual variables with the dopplegangers of their neighbors in other worlds
    undirected |= stitch_factual_and_doppleganger_neighbors(graph, worlds)

    # Stitch together counterfactual variables with their dopplegangers in other worlds
    if len(worlds) > 1:
        undirected |= stitch_counterfactual_and_dopplegangers(graph, worlds)
        # Stitch together counterfactual variables with the dopplegangers of their neighbors in other worlds
        undirected |= stitch_counterfactual_and_doppleganger_neighbors(graph, worlds)

    nodes = [
        *graph.nodes(),
        *(node @ world for world in worlds for node in graph.nodes()),
    ]
    directed = [*graph.directed.edges(), *_get_directed_edges(graph, worlds)]
    return NxMixedGraph.from_edges(
        nodes=nodes,
        directed=directed,
        undirected=set(graph.undirected.edges()) | undirected,
    )
