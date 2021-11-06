# -*- coding: utf-8 -*-

"""Implementation of the IDC algorithm."""

from itertools import combinations
from typing import Collection, Optional, Tuple

from ..conditional_independencies import are_d_separated
from ...dsl import (
    CounterfactualVariable,
    Expression,
    Intervention,
    One,
    P,
    Probability,
    Product,
    Sum,
    Variable,
    _get_treatment_variables,
)
from ...graph import NxMixedGraph

__all__ = [
    "make_parallel_worlds_graph",
    "make_world_graph",
    "lemma_24",
    "lemma_25",
    "idc_star_line_2",
    "id_star_line_1",
    "id_star_line_2",
    "id_star_line_3",
    "id_star_line_4",
    "id_star_line_5",
    "id_star_line_6",
    "id_star_line_7",
    "id_star_line_8",
    "id_star_line_9",
]


class Inconsistent(ValueError):
    pass


def id_star(graph: NxMixedGraph, query: Probability) -> Optional[Expression]:
    # Line 0
    if query.is_conditioned():
        raise ValueError(f"Query {query} must be unconditional")
    gamma = set(query.children)
    # Line 1
    if len(gamma) == 0:
        return One()
    for counterfactual in gamma:
        if isinstance(counterfactual, CounterfactualVariable):
            for intervention in counterfactual.interventions:
                if intervention.name == counterfactual.name:
                    # Line 2: This violates the Axiom of Effectiveness
                    if intervention.star:
                        return None
                    else:
                        # Line 3: This is a tautological event and can be removed without affecting the probability
                        return id_star(graph, P(gamma - {counterfactual}))
    # Line 4:
    try:
        new_graph, new_query = make_counterfactual_graph(graph, query)
        vertices = set(new_graph.nodes())
        new_gamma = set(new_query.children)
    # Line 5:
    except Inconsistent:
        return None
    # Line 6:
    if not new_graph.is_connected():
        return Sum[vertices - new_gamma](
            Product.safe(
                id_star(new_graph, P[vertices - district](district))
                for district in new_graph.get_c_components()
            )
        )
    # Line 7:
    else:
        # Line 8 is syntactically impossible with the dsl
        # Line 9
        return id_star_line_9(query)


def get_val(counterfactual: CounterfactualVariable, graph: NxMixedGraph) -> Intervention:
    var = Variable(counterfactual.name)
    for intervention in counterfactual.interventions:
        if Variable(intervention.name) in graph.ancestors_inclusive(var):
            if intervention.star:
                return ~var
    return -var


def id_star_line_1(graph: NxMixedGraph, gamma: Collection[Variable]) -> Expression:
    r"""Run line 1 of the ID* algorithm.

    The first line states that if :math:`\gamma` is an empty conjunction, then its
    probability is 1, by convention.
    """
    if len(gamma) == 0:
        return One()


def id_star_line_2(graph: NxMixedGraph, query: Probability) -> Expression:
    r"""Run line 2 of the ID* algorithm.

    The second line states that if :math:`\gamma` contains a counterfactual
    which violates the Axiom of Effectiveness (Pearl, 2000), then :math:`\gamma`
    is inconsistent, and we return probability 0.
    """
    raise NotImplementedError


def id_star_line_3(graph: NxMixedGraph, query: Probability) -> Expression:
    r"""Run line 3 of the ID* algorithm.

    The third line states that if a counterfactual contains its own value in the subscript,
    then it is a tautological event, and it can be removed from :math:`\gamma` without
    affecting its probability.
    """
    raise NotImplementedError


def id_star_line_4(graph: NxMixedGraph, query: Probability) -> Expression:
    r"""Run line 4 of the ID* algorithm

    Line 4 invokes make-cg to construct a counterfactual graph :math:`G'` , and the
    corresponding relabeled counterfactual :math:`\gamma'`.
    """
    raise NotImplementedError


def id_star_line_5(graph: NxMixedGraph, query: Probability) -> Expression:
    r"""Run line 5 of the ID* algorithm.

    Line 5 returns probability 0 if an inconsistency was found during the construction
    of the counterfactual graph, for example, if two variables found to be the same in
    :math:`\gamma` had different value assignments.
    """
    raise NotImplementedError


def id_star_line_6(graph: NxMixedGraph, query: Probability) -> Collection[Expression]:
    r"""Run line 6 of the ID* algorithm.

    Line 6 is analogous to Line 4 in the ID algorithm, it decomposes the problem into a
    set of subproblems, one for each C-component in the counterfactual graph. In the ID
    algorithm, the term corresponding to a given C-component :math:`S_i` of the causal
    diagram was the effect of all variables not in :math:`S_i` on variables in :math:`S_i` ,
    in other words :math:`P_{\mathbf{v}\backslash s_i (s_i )`, and the outermost summation
    on line 4 was over values of variables not in :math:`\mathbf{Y},\mathbf{X}`. Here, the
    term corresponding to a given C-component :math:`S^i` of the counterfactual graph :math:`G'`
    is the conjunction of counterfactual variables where each variable contains in its
    subscript all variables not in the C-component :math:`S^i` , in other words
    :math:`\mathbf{v}(G' )\backslash s^i` , and the outermost summation is over observable
    variables not in :math:`\gamma'` , that is over :math:`\mathbf{v}(G' ) \backslash \gamma'` ,
    where we interpret :math:`\gamma'` as a set of counterfactuals, rather than a conjunction.
    """
    return [P[vertices - dictrict](district) for district in graph.get_c_components()]


def id_star_line_7(graph: NxMixedGraph, query: Probability) -> Collection[Expression]:
    r"""Run line 7 of the ID* algorithm.

    Line 7 is the base case, where our counterfactual graph has a single C-component
    """
    raise NotImplementedError


def id_star_line_8(graph: NxMixedGraph, query: Probability) -> Collection[Expression]:
    r"""Run line 8 of the ID* algorithm.

    Line 8 says that if :math:`\gamma'` contains a "conflict," that is an inconsistent
    value assignment where at least one value is in the subscript, then we fail.
    """
    raise NotImplementedError


def id_star_line_9(graph: NxMixedGraph, query: Probability) -> Collection[Expression]:
    r"""Run line 9 of the ID* algorithm.

    Line 9 says if there are no conflicts, then its safe to take the union of all
    subscripts in :math:`\gamma'` , and return the effect of the subscripts in :math:`\gamma'`
    on the variables in :math:`\gamma'`."""
    raise NotImplementedError


def idc_star_line_2(graph: NxMixedGraph, query: Probability) -> Expression:
    r"""Run line 2 of the IDC* algorithm.

    The second line states that if :math:`\gamma` contains a counterfactual which violates
    the Axiom of Effectiveness (Pearl, 2000), then :math:`\gamma` is inconsistent, and we
    return probability 0.
    """
    delta = query.parents
    # FIXME this should be set(query.children).union(query.parents)
    gamma_and_delta = P(query.children + query.parents)
    return make_counterfactual_graph(graph, gamma_and_delta)


def idc_star_line_4(graph: NxMixedGraph, query: Probability) -> bool:
    r"""Run line 4 of the IDC* algorithm.

    Line 4 of IDC* is the central line of the algorithm and is
    analogous to line 1 of IDC. In IDC, we moved a value
    assignment :math:`Z = z` from being observed to being fixed if
    there were no back-door paths from :math:`Z` to the outcome
    variables :math:`Y` given the context of the effect of
    :math:`do(\mathbf{x})`. Here in IDC*, we move a counterfactual
    value assignment :math:`Y_\mathbf{x} = y` from being observed (that is being a
    part of :math:`\delta`), to being fixed (that is appearing in every
    subscript of :math:`\gamma'` ) if there are no back-door paths from :math:`Y_\mathbf{x}` to
    the counterfactual of interest :math:`\gamma'` .
    """
    gamma = set(query.children)
    raise NotImplementedError


def idc_star(graph: NxMixedGraph, query: Probability) -> Optional[Expression]:
    r"""Run the IDC* algorithm.

    INPUT:
        G a causal diagram,
        :math:`\gamma` a conjunction of counterfactual outcomes,
        :math:`\delta` a conjunction of counterfactual observations
    :returns: an expression for :math:`P(\gamma | \delta)` in terms of P, FAIL, or UNDEFINED
    """
    delta = set(query.parents)
    if not delta:
        raise ValueError(f"Query {query} must be conditional")
    # Line 1:
    if not id_star(graph, P(delta)):
        raise ValueError(f"Query {query} is undefined")
    gamma = set(query.children)
    # Line 2:
    try:
        new_graph, new_query = make_counterfactual_graph(graph, P(gamma.union(delta)))
        new_gamma = {g for g in gamma if g in new_query.children}
        new_delta = {d for d in delta if d in new_query.children}
        vertices = set(new_graph.nodes())
    # Line 3:
    except Inconsistent(f"query {gamma.union(delta)} is inconsistent"):
        return None
    # Line 4:
    for counterfactual in new_delta:
        # TODO do we need to extend the notion of d-separation from 1-1 to 1-many?
        if are_d_separated(new_graph.remove_out_edges(counterfactual), counterfactual, new_gamma):
            counterfactual_value = Variable(counterfactual.name)
            parents = new_delta - {counterfactual}
            children = {g.remove_in_edges(counterfactual_value) for g in new_gamma}
            return idc_star(graph, P(children | parents))
    # Line 5:
    estimand = id_star(graph, new_query)
    if estimand is None:
        raise NotImplementedError
    return estimand / Sum.safe(estimand, vertices - delta)


def get_varnames(query: Probability) -> Collection[Variable]:
    r"""Return new Variables generated from the names of the outcome variables in the query."""
    return {Variable(outcome.name) for outcome in query.children}


def get_interventions(query: Probability) -> Collection[Variable]:
    r"""Generate new Variables from the subscripts of counterfactual variables in the query."""
    interventions = set()
    for counterfactual in query.children:
        if isinstance(counterfactual, CounterfactualVariable):
            interventions |= set(counterfactual.interventions)
    return sorted(interventions)


def id_star_line_9(query: Probability) -> Expression:
    """Gather all interventions and applies them to the varnames of outcome variables."""
    varnames = get_varnames(query)
    interventions = get_interventions(query)
    return P[interventions](varnames)


def has_same_parents(graph: NxMixedGraph, node1: Variable, node2: Variable) -> bool:
    """Check if all parents of the two nodes are the same.

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
    raise NotImplementedError


def has_same_function(node1: Variable, node2: Variable) -> bool:
    return node1.name == node2.name


def get_worlds(query: Probability) -> Collection[Collection[Variable]]:
    return sorted(
        [
            sorted(_get_treatment_variables(var.get_variables()), key=lambda x: str(x))
            for var in query.get_variables()
            if isinstance(var, CounterfactualVariable)
        ]
    )


def lemma_24(pw_graph: NxMixedGraph, node1, node2) -> bool:
    r"""Check if two nodes in a parallel worlds graph are equivalent.

    Let :math:`M` be a model inducing :math:`G` containing variables :math:`\alpha`, :math:`\beta` with the following properties:

    * :math:`\alpha` and :math:`\beta` have the same domain of values.
    * There is a bijection :math:`f` from :math:`Pa(\alpha)` to :math:`Pa(\beta)` such that a parent  :math:`\gamma` and :math:`f(\gamma)` have the same domain of values.
    *  The functional mechanisms of :math:`\alpha` and :math:`\beta` are the same (except whenever the function for :math:`\alpha` uses the parent  :math:`\gamma`, the corresponding function for :math:`\beta` uses :math:`f(\gamma)`).

    Assume an observable variable set :math:`\mathbf{Z}` was observed to attain values :math:`z` in :math:`M_\mathbf{x}` , the submodel obtained from :math:`M` by forcing another observable variable set :math:`\mathbf{X}` to attain values :math:`\mathbf{x}`. Assume further that for each  :math:`\gamma \in Pa(\alpha)`, either :math:`f(\gamma) =  \gamma`, or  :math:`\gamma` and :math:`f(\gamma)` attain the same values (whether by observation or intervention). Then :math:`\alpha` and :math:`\beta` are the same random variable in :math:`M_\mathbf{x}` with observations :math:`\mathbf{z}`

    """
    # Rather than all n choose 2 combinations, we can restrict ourselves to the original graph variables and their counterfactual versions
    return has_same_function(node1, node2) and has_same_parents(pw_graph, node1, node2)


def lemma_25(graph: NxMixedGraph, node1: Variable, node2: Variable) -> NxMixedGraph:
    r"""Merge node1 and node2 and return the reduced graph and query.

    Let :math:`M_\mathbf{x}` be a submodel derived from :math:`M` with set :math:`\mathbf{Z}` observed to attain values :math:`\mathbf{z}`, such that Lemma 24 holds for :math:`\alpha`; :math:`\beta`. Let :math:`M'` be a causal model obtained from :math:`M` by merging :math:`\alpha`; :math:`\beta` into a new node :math:`\omega`, which inherits all parents and the functional mechanism of :math:`\alpha`. All children of :math:`\alpha`; :math:`\beta` in :math:`M'` become children of :math:`\omega`. Then :math:`M_\mathbf{x},  M'_\mathbf{x} agree on any distribution consistent with :math:`z` being observed.

    """
    if isinstance(node1, CounterfactualVariable) and not isinstance(node2, CounterfactualVariable):
        node1, node2 = node2, node1
    elif (not isinstance(node1, CounterfactualVariable)) and isinstance(
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
            if (
                (node in cf_graph.nodes())
                and (node @ intervention in cf_graph.nodes())
                and lemma_24(cf_graph, node, node @ intervention)
            ):
                cf_graph = lemma_25(cf_graph, node, node @ intervention)
                if node @ intervention in new_query_variables:
                    new_query_variables = (new_query_variables - {node @ intervention}) | {node}

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


def to_adj(graph: NxMixedGraph):
    nodes: list[Variable] = list(graph.nodes())
    directed: dict[Variable, list[Variable]] = {u: [] for u in nodes}
    undirected: dict[Variable, list[Variable]] = {u: [] for u in nodes}
    for u, v in graph.directed.edges():
        directed[u].append(v)
    for u, v in graph.undirected.edges():
        undirected[u].append(v)
    return nodes, directed, undirected


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
