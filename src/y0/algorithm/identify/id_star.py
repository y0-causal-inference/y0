# -*- coding: utf-8 -*-

"""Implementation of the IDC algorithm."""

from .id_std import identify
from .utils import Identification
from ..conditional_independencies import are_d_separated
from ...dsl import Expression, Variable, P, Sum, Product

__all__ = [
    "make_parallel_worlds_graph",
    "combine_parallel_worlds",
    "make_parallel_world_graph",
    "lemma_24",
    "lemma_25",
]


from y0.graph import NxMixedGraph
from y0.dsl import Variable, CounterfactualVariable, _get_treatment_variables, Intervention
from typing import Set, Collection
from itertools import combinations
from y0.graph import NxMixedGraph
from y0.dsl import Variable, Probability, Intervention
from itertools import combinations
from typing import Collection, Tuple
from networkx.algorithms.dag import topological_sort
from networkx import DiGraph


def id_star(graph: NxMixedGraph[Variable], query: Probability) -> Expression:

    # Line 0
    if len(query.distribution.parents) != 0:
        raise ValueError(f"Query {query} must be unconditional")
    gamma = set(query.distribution.children)
    # Line 1
    if len(gamma) == 0:
        return P(1)
    for counterfactual in gamma:
        if isinstance(counterfactual, CounterfactualVariable):
            for intervention in counterfactual.interventions:
                if intervention.name == counterfactual.name:
                    # Line 2: This violates the Axiom of Effectiveness
                    if intervention.star:
                        return 0
                    else:
                        # Line 3: This is a tautological event and can be removed without affecting the probability
                        return id_star(graph, P(gamma - {counterfactual}))
    # Line 4:
    try:
        new_graph, new_query = make_counterfactual_graph(graph, query)
        vertices = set(new_graph.nodes())
        new_gamma = set(new_query.distribution.children)
    # Line 5:
    except Inconsistent(f"query {query} is inconsistent"):
        return P(0)
    # Line 6:
    if not new_graph.is_connected():
        return Sum[vertices - new_gamma](
            Product(
                [
                    id_star(new_graph, P[vertices - district](district))
                    for district in new_graph.get_c_components()
                ]
            )
        )
    # Line 7:
    else:
        # Line 8 is syntactically impossible with the dsl
        interventions = set()
        for counterfactual in vertices:
            if isinstance(counterfactual, CounterfactualVariable):
                interventions |= counterfactual.interventions
        return P[interventions](Variable(v.name) for v in vertices)


def idc_star(graph: NxMixedGraph[Variable], query: Probability) -> Expression:
    gamma = set(query.distribution.children)
    delta = set(query.distribution.parents)
    if len(delta) == 0:
        raise ValueError(f"Query {query} must be conditional")
    # Line 1:
    if id_star(graph, P(delta)) == 0:
        return Undefined(f"Query {query} is undefined")
    # Line 2:
    try:
        new_graph, new_query = make_counterfactual_graph(graph, P(gamma.union(delta)))
        new_gamma = {g for g in gamma if g in new_query.distribution.children}
        new_delta = {d for d in delta if d in new_query.distribution.children}
        vertices = set(new_graph.nodes())
    # Line 3:
    except Inconsistent(f"query {gamma.union(delta)} is inconsistent"):
        return 0
    # Line 4:
    for counterfactual in new_delta:
        if are_d_separated(
            new_graph.remove_outgoing_edges_from(counterfactual), counterfactual, new_gamma
        ):
            counterfactual_value = Variable(counterfactual.name)
            parents = new_delta - {counterfactual}
            children = {g.intervene(counterfactual_value) for g in new_gamma}
            return idc_star(graph, P(children | parents))
    # Line 5:
    estimand = id_star(graph, new_query)
    return estimand / Sum[vertices - delta](estimand)


def has_same_parents(graph: NxMixedGraph[Variable], node1: Variable, node2: Variable) -> bool:
    """Check if all parents of the two nodes are the same

    This is true if the set of directed parents are the same and either there exists a bidirected edge between the two nodes or there exists no bidirected edges for either node.
    """
    return (
        set(graph.directed.predecessors(node1)) == set(graph.directed.predecessors(node2))
    ) and (
        graph.undirected.has_edge(node1, node2)
        or ((len(graph.undirected.edges(node1)) == 0) and len(graph.undirected.edges(node2)) == 0)
    )


def has_same_domain_of_values(node1: Variable, node2: Variable) -> bool:
    if isinstance(node1, CounterfactualVariable) and isinstance(node2, CounterfactualVariable):
        treatment1, treatment2 = _get_treatment_variables(node1), _get_treatment_variables(node2)


def has_same_function(node1: Variable, node2: Variable) -> bool:
    return node1.name == node2.name


def get_worlds(query: Probability) -> Collection[Collection[Variable]]:
    return [
        _get_treatment_variables(var.get_variables())
        for var in query.get_variables()
        if isinstance(var, CounterfactualVariable)
    ]


def lemma_24(pw_graph: NxMixedGraph[Variable], node1, node2) -> bool:
    r"""Check if two nodes in a parallel worlds graph are equivalent

    Let :math:`M` be a model inducing :math:`G` containing variables :math:`\alpha`, :math:`\beta` with the following properties:

    * :math:`\alpha` and :math:`\beta` have the same domain of values.
    * There is a bijection :math:`f` from :math:`Pa(\alpha)` to :math:`Pa(\beta)` such that a parent  :math:`\gamma` and :math:`f(\gamma)` have the same domain of values.
    *  The functional mechanisms of :math:`\alpha` and :math:`\beta` are the same (except whenever the function for :math:`\alpha` uses the parent  :math:`\gamma`, the corresponding function for :math:`\beta` uses :math:`f(\gamma)`).

    Assume an observable variable set :math:`\mathbf{Z}` was observed to attain values :math:`z` in :math:`M_\mathbf{x}` , the submodel obtained from :math:`M` by forcing another observable variable set :math:`\mathbf{X}` to attain values :math:`\mathbf{x}`. Assume further that for each  :math:`\gamma \in Pa(\alpha)`, either :math:`f(\gamma) =  \gamma`, or  :math:`\gamma` and :math:`f(\gamma)` attain the same values (whether by observation or intervention). Then :math:`\alpha` and :math:`\beta` are the same random variable in :math:`M_\mathbf{x}` with observations :math:`\mathbf{z}`

    """
    # Rather than all n choose 2 combinations, we can restrict ourselves to the original graph variables and their counterfactual versions
    return has_same_function(node1, node2) and has_same_parents(pw_graph, node1, node2)


def lemma_25(
    graph: NxMixedGraph[Variable], node1: Variable, node2: Variable
) -> NxMixedGraph[Variable]:
    r"""Merge node1 and node2 and return the reduced graph and query

    Let :math:`M_\mathbf{x}` be a submodel derived from :math:`M` with set :math:`\mathbf{Z}` obse to attain values :math:`\mathbf{z}`, such that Lemma 24 holds for :math:`\alpha`; :math:`\beta`. Let :math:`M'` be a causal model obtained from :math:`M` by merging :math:`\alpha`; :math:`\beta` into a new node :math:`\omega`, which inherits all parents and the functional mechanism of :math:`\alpha`. All children of :math:`\alpha`; :math:`\beta` in :math:`M'` become children of :math:`\omega`. Then :math:`M_\mathbf{x},  M'_\mathbf{x} agree on any distribution consistent with :math:`z` being observed.

    """
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
    graph: NxMixedGraph[Variable], query: Probability
) -> Tuple[NxMixedGraph[Variable], Probability]:
    """Make counterfactual graph"""
    worlds = get_worlds(query)
    pw_graph = make_parallel_worlds_graph(graph, worlds)
    new_query_variables = set(query.distribution.children)
    cf_graph = NxMixedGraph.from_edges(
        nodes=pw_graph.nodes(),
        directed=pw_graph.directed.edges(),
        undirected=pw_graph.undirected.edges(),
    )
    for node in graph.topological_sort():
        for intervention in worlds:
            if (
                (node in cf_graph.nodes())
                and (node @ intervention in cf_graph.nodes())
                and lemma_24(cf_graph, node, node @ intervention)
            ):
                cf_graph = lemma_25(cf_graph, node, node @ intervention)
                if node @ intervention in new_query_variables:
                    new_query_variables = new_query_variables - {node @ intervention} | {node}

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
    return cf_graph.subgraph(cf_graph.ancestors_inclusive(new_query_variables)), P(
        new_query_variables
    )


def make_parallel_worlds_graph(
    graph: NxMixedGraph[Variable], worlds: Collection[Collection[Variable]]
) -> NxMixedGraph[Variable]:
    """Make Parallel worlds graph"""
    combined_worlds = [make_parallel_world_graph(graph, world) for world in worlds]
    return combine_parallel_worlds(graph, combined_worlds, worlds)


def make_parallel_world_graph(
    graph: NxMixedGraph[Variable], treatments: Collection[Variable]
) -> NxMixedGraph[Variable]:
    """Make one parallel world based on interventions specified"""
    pw_graph = graph.intervene(treatments)
    return NxMixedGraph.from_edges(
        nodes=[node.intervene(treatments) for node in pw_graph.nodes()],
        directed=[(u.intervene(treatments), v.intervene(treatments)) for u, v in pw_graph.directed.edges()],
        undirected=[
            (u.intervene(treatments), v.intervene(treatments))
            for u, v in pw_graph.undirected.edges()
            if (u not in treatments)
            and (v not in treatments)
            and (~u not in treatments)
            and (~v not in treatments)
        ],
    )


def to_adj(graph: NxMixedGraph):
    nodes = graph.nodes()
    directed = {u: [] for u in nodes}
    undirected = {u: [] for u in nodes}
    for u, v in graph.directed.edges():
        directed[u].append(v)
    for u, v in graph.undirected.edges():
        undirected[u].append(v)
    return nodes, directed, undirected


def combine_parallel_worlds(
    graph: NxMixedGraph[Variable],
    combined_worlds: Collection[NxMixedGraph[Variable]],
    worlds: Collection[Collection[Variable]],
) -> NxMixedGraph[Variable]:
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
