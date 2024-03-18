# -*- coding: utf-8 -*-

"""Implement of surrogate outcomes and transportability from https://arxiv.org/abs/1806.07172.

.. todo::

    high level documentation

    1. What problem are we trying to solve here?
    2. What's the difference between surrogate outcomes nad transportability?
    3. Real world example
"""

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Collection, Dict, FrozenSet, Iterable, Optional, Set, Union, cast

from y0.algorithm.conditional_independencies import are_d_separated
from y0.dsl import (
    TARGET_DOMAIN,
    CounterfactualVariable,
    Distribution,
    Expression,
    Fraction,
    Intervention,
    One,
    Population,
    PopulationProbability,
    Probability,
    Product,
    Sum,
    Variable,
    Zero,
    _upgrade_variables,
)
from y0.graph import NxMixedGraph
from y0.mutate.canonicalize_expr import canonicalize

__all__ = [
    "identify_target_outcomes",
    "trso",
    "TransportQuery",
]

logger = logging.getLogger(__name__)


def get_nodes_to_transport(
    *,
    surrogate_interventions: Union[Set[Variable], Variable],
    surrogate_outcomes: Union[Set[Variable], Variable],
    graph: NxMixedGraph,
) -> Set[Variable]:
    """Identify which nodes the transport nodes should point to.

    :param surrogate_interventions: The interventions performed in an experiment.
    :param surrogate_outcomes: The outcomes observed in an experiment.
    :param graph: The graph of the target domain.
    :returns: A set of variables representing target domain nodes where transportability nodes should be added.
    """
    surrogate_interventions = set(_upgrade_variables(surrogate_interventions))
    surrogate_outcomes = set(_upgrade_variables(surrogate_outcomes))

    # Find the c_component with surrogate_outcomes
    c_component_surrogate_outcomes: Set[Variable] = set()
    for component in graph.districts():
        # Check if surrogate_outcomes is present in the current set
        if surrogate_outcomes.intersection(component):
            c_component_surrogate_outcomes.update(component)

    ancestors_surrogate_outcomes = graph.get_intervened_ancestors(
        surrogate_interventions, surrogate_outcomes
    )

    # Descendants of interventions in graph
    descendants_interventions = graph.descendants_inclusive(surrogate_interventions)

    return (descendants_interventions - surrogate_outcomes).union(
        c_component_surrogate_outcomes - ancestors_surrogate_outcomes
    )


_TRANSPORT_PREFIX = "T_"


def transport_variable(variable: Variable) -> Variable:
    """Create a transport Variable by adding the transport prefix to a variable.

    :param variable: variable that the transport node will point to
    :returns: Variable with _TRANSPORT_PREFIX and variable name
    :raises TypeError: If a non-standard variable is passed
    """
    if isinstance(variable, (CounterfactualVariable, Intervention)):
        raise TypeError
    return Variable(_TRANSPORT_PREFIX + variable.name)


def is_transport_node(node: Variable) -> bool:
    """Check if a Variable is a transport node.

    :param node: A node to evaluate.
    :returns: boolean True if node is a transport node, False otherwise.
    """
    return not isinstance(node, (CounterfactualVariable, Intervention)) and node.name.startswith(
        _TRANSPORT_PREFIX
    )


def get_transport_nodes(graph: NxMixedGraph) -> Set[Variable]:
    """Find all the transport nodes in a graph.

    :param graph: an NxMixedGraph which may have transport nodes
    :returns: Set containing all transport nodes in the graph
    """
    return {node for node in graph.nodes() if is_transport_node(node)}


def get_regular_nodes(graph: NxMixedGraph) -> Set[Variable]:
    """Find all the nodes in a graph which are not transport nodes.

    :param graph: an NxMixedGraph
    :returns: Set containing all nodes which are not transport nodes
    """
    return {node for node in graph.nodes() if not is_transport_node(node)}


def _c14n_safe(expression: Expression | None) -> Expression | None:
    if expression is None:
        return None
    return canonicalize(expression)


def create_transport_diagram(
    *,
    nodes_to_transport: Iterable[Variable],
    graph: NxMixedGraph,
) -> NxMixedGraph:
    """Create a NxMixedGraph identical to graph but with transport nodes added.

    :param nodes_to_transport: nodes which have transport nodes pointing to them.
    :param graph: The graph of the target domain.
    :returns: graph with transport nodes added
    """
    rv = NxMixedGraph()
    for node in graph.nodes():
        rv.add_node(node)
    for u, v in graph.directed.edges():
        rv.add_directed_edge(u, v)
    for u, v in graph.undirected.edges():
        rv.add_undirected_edge(u, v)
    for node in nodes_to_transport:
        transport_node = transport_variable(node)
        rv.add_directed_edge(transport_node, node)
    return rv


@dataclass
class TransportQuery:
    """A query used as output for surrogate_to_transport."""

    target_interventions: Set[Variable]
    target_outcomes: Set[Variable]
    graphs: Dict[Population, NxMixedGraph]
    domains: Set[Population]
    surrogate_interventions: Dict[Population, Set[Variable]]
    target_experiments: Set[Variable]


@dataclass
class TRSOQuery:
    """A query used for TRSO input."""

    target_interventions: Set[Variable]
    target_outcomes: Set[Variable]
    expression: Expression
    active_interventions: Set[Variable]
    domain: Population
    domains: Set[Population]
    graphs: Dict[Population, NxMixedGraph]
    surrogate_interventions: Dict[Population, Set[Variable]]


def surrogate_to_transport(
    *,
    graph: NxMixedGraph,
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    surrogate_outcomes: Dict[Population, Set[Variable]],
    surrogate_interventions: Dict[Population, Set[Variable]],
) -> TransportQuery:
    """Create transportability diagrams and query from a surrogate outcome problem.

    :param target_outcomes: A set of target variables for causal effects.
    :param target_interventions: A set of interventions for the target domain.
    :param graph: The graph of the target domain.
    :param surrogate_outcomes: A dictionary of outcomes in other populations
    :param surrogate_interventions: A dictionary of interventions in other populations
    :returns: An octuple representing the query transformation of a surrogate outcome query.
    :raises ValueError: if surrogate outcomes' and surrogate interventions' keys do not correspond
    """
    if set(surrogate_outcomes) != set(surrogate_interventions):
        raise ValueError("Inconsistent surrogate outcome and intervention domains")

    graphs = {
        domain: create_transport_diagram(
            graph=graph,
            nodes_to_transport=get_nodes_to_transport(
                surrogate_interventions=surrogate_interventions[domain],
                surrogate_outcomes=domain_outcomes,
                graph=graph,
            ),
        )
        for domain, domain_outcomes in surrogate_outcomes.items()
    }
    graphs[TARGET_DOMAIN] = graph

    return TransportQuery(
        target_interventions=target_interventions,
        target_outcomes=target_outcomes,
        graphs=graphs,
        domains=set(surrogate_outcomes),
        surrogate_interventions=surrogate_interventions,
        target_experiments=set(),
    )


def trso_line1(
    target_outcomes: Set[Variable],
    expression: Expression,
    graph: NxMixedGraph,
) -> Expression:
    """Return the probability in the case where no interventions are present.

    :param target_outcomes: A set of nodes that comprise our target outcomes.
    :param expression: The distribution in the current domain.
    :param graph: The graph with transport nodes in this domain.
    :returns: Sum over the probabilities of nodes other than target outcomes.
    """
    return Sum.safe(expression, get_regular_nodes(graph) - target_outcomes)


def trso_line2(
    query: TRSOQuery,
    outcomes_ancestors: Set[Variable],
) -> TRSOQuery:
    """Restrict the interventions and diagram to only include ancestors of target variables.

    :param query: A TRSO query
    :param outcomes_ancestors: the ancestors of target variables in transportability_diagram
    :returns: A TRSO query with modified attributes.
    """
    new_query = deepcopy(query)
    new_query.target_interventions.intersection_update(outcomes_ancestors)

    for domain, graph in query.graphs.items():
        outcome_ancestors_domain = graph.ancestors_inclusive(query.target_outcomes)
        new_query.graphs[domain] = graph.subgraph(outcome_ancestors_domain)

    new_query.expression = Sum.safe(
        query.expression,
        get_regular_nodes(query.graphs[query.domain]) - outcomes_ancestors,
        simplify=True,
    )
    if isinstance(new_query.expression, Probability):
        assert isinstance(new_query.expression, PopulationProbability)
        # it might be the case that these two are not the same, but
        # other parts of the algorithm clean it up. This isn't so
        # satisfying. Sorry!
        # if new_query.expression.population != new_query.domain:
        #     pass
        new_query.expression = PopulationProbability(
            population=new_query.domain,
            distribution=Distribution(
                children=new_query.expression.children,
            ),
        )
    return new_query


def trso_line3(query: TRSOQuery, additional_interventions: Set[Variable]) -> TRSOQuery:
    """Add nodes that will affect the outcome to the interventions of the query.

    :param query: A TRSO query
    :param additional_interventions: interventions to be added to target_interventions
    :returns: A TRSO query with modified attributes.
    """
    new_query = deepcopy(query)
    new_query.target_interventions.update(additional_interventions)
    return new_query


def trso_line4(
    query: TRSOQuery,
    components: Iterable[FrozenSet[Variable]],
) -> Dict[FrozenSet[Variable], TRSOQuery]:
    """Find the trso inputs for each C-component.

    :param query: A TRSO query
    :param components: Set of c_components of transportability_diagram without target_interventions
    :returns: Dictionary with components as keys TRSOQuery objects as values
    """
    graph = query.graphs[query.domain]
    rv = {}
    for component in components:
        new_query = deepcopy(query)
        new_query.target_outcomes = set(component)
        new_query.target_interventions = get_regular_nodes(graph) - component
        rv[component] = new_query
    return rv


def trso_line6(query: TRSOQuery) -> Dict[Population, TRSOQuery]:
    """Find the active interventions for each domain, remove available experiments from interventions.

    :param query: A TRSO query
    :returns: Dictionary with domains as keys TRSOQuery objects as values
    """
    expressions = {}
    for domain, graph in query.graphs.items():
        if domain == TARGET_DOMAIN:
            continue
        new_query = _line_6_helper(query, domain, graph)
        if new_query is not None:
            expressions[domain] = new_query
    return expressions


def _line_6_helper(
    query: TRSOQuery, domain: Population, graph: NxMixedGraph
) -> Optional[TRSOQuery]:
    """Perform d-separation check and then modify query active interventions.

    :param query: A TRSO query
    :param domain: A given population
    :param graph: A NxMixedGraph
    :returns: A TRSO query or None
    """
    surrogate_interventions = query.surrogate_interventions[domain]
    surrogate_intersect_target = surrogate_interventions.intersection(query.target_interventions)
    if not surrogate_intersect_target:
        return None
    if not all_transports_d_separated(
        graph,
        target_interventions=query.target_interventions,
        target_outcomes=query.target_outcomes,
    ):
        return None

    new_query = deepcopy(query)
    new_query.target_interventions = query.target_interventions - surrogate_interventions
    new_query.domain = domain
    new_query.graphs[new_query.domain] = graph.remove_nodes_from(surrogate_intersect_target)
    new_query.active_interventions = surrogate_intersect_target
    return new_query


def activate_domain_and_interventions(
    expression: Expression, interventions: Set[Variable], domain: Population
) -> Expression:
    """Intervene on the target variables of expression using the active interventions.

    :param expression: A probability expression.
    :param interventions: Set of active interventions
    :param domain: A given population
    :returns: A new expression, intervened
    :raises NotImplementedError: If an expression type that is not handled gets passed
    """
    if isinstance(expression, Probability):
        assert isinstance(expression, PopulationProbability)
        return PopulationProbability(
            population=domain,
            distribution=Distribution.safe(set(expression.children) - interventions),
        ).intervene(interventions)
    if isinstance(expression, Sum):
        # TODO need full integration test to trso() function that covers this branch
        # Don't intervene the ranges because counterfactual variables shouldn't be in ranges
        # intervened_ranges = tuple(
        #     variable.activate_domain_and_interventions(active_interventions) for variable in expression.ranges
        # )
        return Sum.safe(
            activate_domain_and_interventions(expression.expression, interventions, domain),
            expression.ranges,
        )
    if isinstance(expression, Fraction):
        numerator = activate_domain_and_interventions(expression.numerator, interventions, domain)
        denominator = activate_domain_and_interventions(
            expression.denominator, interventions, domain
        )
        return cast(Fraction, numerator / denominator).simplify()
    if isinstance(expression, Product):
        # TODO need full integration test to trso() function that covers this branch
        return Product.safe(
            activate_domain_and_interventions(expr, interventions, domain)
            for expr in expression.expressions
        )
    raise NotImplementedError(f"Unhandled expression type: {type(expression)}")


def all_transports_d_separated(graph, target_interventions, target_outcomes) -> bool:
    """Check if all target_interventions are d-separated from target_outcomes.

    :param graph: The graph with transport nodes in this domain.
    :param target_interventions: Set of target interventions
    :param target_outcomes: Set of target interventions
    :returns: True if all interventions are d-separated from all outcomes, False otherwise.
    """
    transportability_nodes = get_transport_nodes(graph)
    graph_without_interventions = graph.remove_in_edges(target_interventions)
    return all(
        are_d_separated(
            graph_without_interventions,
            transportability_node,
            outcome,
            conditions=target_interventions,
        )
        for transportability_node in transportability_nodes
        if transportability_node in graph_without_interventions
        for outcome in target_outcomes
    )


def trso_line9(query: TRSOQuery, district: set[Variable]) -> Expression:
    """Get the probability in the case with exactly one districts_without_interventions and it is present in districts.

    :param query: A TRSO query
    :param district: The C-component present in both districts_without_interventions and districts
    :returns: An Expression
    :raises RuntimeError: If the query's expression is zero. This should never happen
    """
    logger.debug(
        "Calling trso algorithm line 9 with expression %s \n district %s",
        query.expression,
        district,
    )
    if isinstance(query.expression, Zero):  # pragma: no cover
        # TODO if we can't create an integration test (i.e., a call to trso)
        #  that triggers this line, then it can be safely removed
        raise RuntimeError

    ordering = list(query.graphs[query.domain].topological_sort())
    ordering_set = set(ordering)
    my_product: Expression = One()
    for node in district:
        i = ordering.index(node)
        pre, post = ordering[:i], ordering[: i + 1]
        pre_set = ordering_set - set(post)
        post_set = ordering_set - set(pre)
        numerator = Sum.safe(query.expression, pre_set)
        denominator = Sum.safe(query.expression, post_set)
        my_product *= numerator / denominator
    my_product = cast(Fraction, my_product).simplify()

    logger.debug(
        "Returning trso algorithm line 9 with expression %s",
        Sum.safe(my_product, district - query.target_outcomes),
    )
    return Sum.safe(my_product, district - query.target_outcomes)


def trso_line10(
    query: TRSOQuery,
    district: set[Variable],
    new_surrogate_interventions: Dict[Population, Set[Variable]],
) -> TRSOQuery:
    """Update the TRSO query to restrict interventions and graph to district.

    :param query: A TRSO query
    :param district: The C-component of districts which contains district_without_interventions
    :param new_surrogate_interventions: Dict mapping domains to interventions performed in that domain.
    :returns: A modified TRSOQuery
    """
    ordering = list(query.graphs[query.domain].topological_sort())
    expressions = []
    for node in district:
        i = ordering.index(node)
        pre_node = set(ordering[:i])
        # note tikka splits this into two expressions that when taken together equal pre_node
        distribution = Distribution.safe(node | pre_node)
        expressions.append(
            PopulationProbability(population=query.domain, distribution=distribution)
        )

    new_query = deepcopy(query)
    new_query.target_interventions = query.target_interventions.intersection(district)
    new_query.expression = canonicalize(Product.safe(expressions))
    new_query.graphs[query.domain] = query.graphs[query.domain].subgraph(district)
    new_query.surrogate_interventions = new_surrogate_interventions
    return new_query


def trso(query: TRSOQuery) -> Optional[Expression]:  # noqa:C901
    """Run the TRSO algorithm to evaluate a transport problem.

    :param query: A TRSO query, which contains 8 instance variables needed for TRSO
    :returns: An Expression evaluating the given query, or None
    :raises RuntimeError: when an impossible condition is met
    """
    # Check that domain is in query.domains
    # check that query.surrogate_interventions keys are equals to domains
    # check that query.graphs keys are equal to domains
    logger.debug(
        "Calling trso algorithm with "
        "\t- target_interventions: %s\n"
        "\t- target_outcomes: %s\n"
        "\t- expression: %s\n"
        "\t- active_interventions: %s\n"
        "\t- domain: %s\n"
        "\t- domains: %s\n"
        "\t- graph[domain] nodes: %s\n"
        "\t- surrogate_interventions: %s",
        query.target_interventions,
        query.target_outcomes,
        query.expression,
        query.active_interventions,
        query.domain,
        query.domains,
        query.graphs[query.domain].nodes(),
        query.surrogate_interventions,
    )

    graph = query.graphs[query.domain]
    # line 1
    if not query.target_interventions:
        logger.debug("Calling trso algorithm line 1")
        return canonicalize(trso_line1(query.target_outcomes, query.expression, graph))

    # line 2
    outcome_ancestors = graph.ancestors_inclusive(query.target_outcomes)
    if get_regular_nodes(graph) - outcome_ancestors:
        new_query = trso_line2(query, outcome_ancestors)
        logger.debug("Calling trso algorithm line 2")
        return _c14n_safe(trso(new_query))

    # line 3
    additional_interventions = graph.get_no_effect_on_outcomes(
        query.target_interventions, query.target_outcomes
    )
    if additional_interventions:
        new_query = trso_line3(query, additional_interventions)
        logger.debug("Calling trso algorithm line 3")
        return _c14n_safe(trso(new_query))

    # line 4
    districts_without_interventions: set[frozenset[Variable]] = graph.remove_nodes_from(
        query.target_interventions
    ).districts()
    if len(districts_without_interventions) > 1:
        subqueries = trso_line4(
            query,
            districts_without_interventions,
        )
        terms = []
        logger.debug("Calling trso algorithm line 4 with %d subqueries", len(subqueries))
        for i, subquery in enumerate(subqueries.values()):
            logger.debug("Calling subquery %d of trso algorithm line 4", i + 1)
            term = trso(subquery)
            if term is None:
                return None
            terms.append(term)

        product = Product.safe(terms)
        summand = canonicalize(product)  # fix sort order inside product
        return canonicalize(
            Sum.safe(
                summand,
                get_regular_nodes(graph) - query.target_interventions.union(query.target_outcomes),
            )
        )

    # line 6
    if not query.active_interventions and query.surrogate_interventions:
        expressions: Dict[Population, Expression] = {}
        for domain, subquery in trso_line6(query).items():
            logger.debug("Calling trso algorithm line 6 for domain %s", domain)
            expression = trso(subquery)
            if expression is None:
                continue
            expression = activate_domain_and_interventions(
                expression, subquery.active_interventions, domain
            )
            if expression is not None:  # line7
                logger.debug(
                    "Calling trso algorithm line 7",
                )
                expressions[domain] = expression
        if len(expressions) == 1:
            return canonicalize(list(expressions.values())[0])
        elif len(expressions) > 1:
            # TODO need full integration test to trso() function that covers this branch
            #  or change to ``raise RuntimeError`` if it's not possible to reach in practice
            logger.warning("more than one expression were non-none")
            # What if more than 1 expression doesn't fail?
            # Is it non-deterministic or can we prove it will be length 1?
            return canonicalize(list(expressions.values())[0])
        else:
            # if there are no expressions, then we move on to line 8
            pass

    # line8 checks that len(districts)) != 1
    districts = graph.districts()
    # line 11 states return fail if len(districts)==1
    # keep explict tests for 0 and 1 to ensure adequate testing
    if len(districts) == 0:
        # TODO we need an integration test (i.e., call to trso()) that covers this.
        #  if it's not possible to cover in a real setting, then we can change this
        #  to raising a runtime error. Nathaniel notes that this probably only occurs
        #  if the graph is empty, but it's not clear if it makes sense to have an empty
        #  graph
        return None
    elif len(districts) == 1:
        return None
    # line 8, i.e. len(districts)>1

    # line 9
    if len(districts_without_interventions) == 0:  # pragma: no cover
        # This would happen if there is an intervention that is also an outcome,
        # which we ensure is not possible when calling the algorithm from its harness
        raise RuntimeError

    # at this point, we already checked for cases where len > 2 and len == 0,
    # so we can safely pop the only element
    district_without_interventions = districts_without_interventions.pop()
    if district_without_interventions in districts:
        return canonicalize(trso_line9(query, set(district_without_interventions)))

    # line10
    logger.debug("Calling trso algorithm line 10")
    target_districts = [
        district for district in districts if district_without_interventions.issubset(district)
    ]
    if len(target_districts) != 1:  # pragma: no cover
        # At this point, the mathematics require this, and therfore this
        # test should never evaluate to true
        raise RuntimeError
    target_district = target_districts.pop()
    # district is C' districts should be D[C'], but we chose to return set of nodes instead of subgraph
    if len(query.active_interventions) == 0:
        # TRSO Line 6 could return an empty list and skip over the returns, allowing this line to be reached.
        new_surrogate_interventions = dict()
    elif _pillow_has_transport(graph, target_district):
        return None
    else:
        new_surrogate_interventions = query.surrogate_interventions

    new_query = trso_line10(
        query,
        set(target_district),
        new_surrogate_interventions,
    )
    return _c14n_safe(trso(new_query))


def _pillow_has_transport(graph: NxMixedGraph, district: Collection[Variable]) -> bool:
    return any(is_transport_node(node) for node in graph.get_markov_pillow(district))


def check_and_raise_missing(nodes: set[Variable], graph: NxMixedGraph, name: str) -> None:
    """Verify that nodes are present in the graph.

    :param nodes: A set of nodes that should be in the graph.
    :param graph: An NxMixedGraph(), the graph of the target domain.
    :param name: Name of the set of nodes
    :raises ValueError: If any element of nodes is not in the graph.
    """
    missing_nodes = nodes - graph.nodes()
    missing_nodes_text = {node.to_text() for node in missing_nodes}
    if missing_nodes_text:
        raise ValueError(
            f"The following {name} are not in the graph: {', '.join(missing_nodes_text)}"
        )


def identify_target_outcomes(
    graph: NxMixedGraph,
    *,
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    surrogate_outcomes: Dict[Population, Set[Variable]],
    surrogate_interventions: Dict[Population, Set[Variable]],
) -> Expression | None:
    r"""Get the estimand for the target outcome givne the surrogate outcomes.

    .. seealso:: Originally described in https://arxiv.org/abs/1806.07172.

    :param target_outcomes: A set of target variables for causal effects.
    :param target_interventions: A set of interventions for the target domain.
    :param graph: The graph of the target domain.
    :param surrogate_outcomes: A dictionary of outcomes in other populations
    :param surrogate_interventions: A dictionary of interventions in other populations
    :returns: An Expression evaluating the given query, or None
    :raises ValueError: If the target outcomes and target interventions intersect

    The example from figure 8 of the original paper can be executed with
    the following code:

    .. code-block:: python

        from y0.algorithm.transport import identify_target_outcome
        from y0.dsl import X1, X2, Y1, Y2, Pi1, Pi2
        from y0.examples import tikka_trso_figure_8_graph

        estimand = identify_target_outcome(
            graph=tikka_trso_figure_8_graph,
            target_outcomes={Y1, Y2},
            target_interventions={X1, X2},
            surrogate_outcomes={Pi1: {Y1}, Pi2: {Y2}},
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

    This returns the following estimand:
    $\sum_{W, Z} P(W, Z) \frac{P_{X_1}^{π_1}(W, Y_1, Z)}{P_{X_1}(W, Z)}
    \frac{P_{X_2}^{π_2}(W, X_1, Y_2, Z)}{P_{X_2}(W, X_1, Z)}$
    """
    # TODO add vanilla identification check?
    # vanilla_estimand = identify_outcomes(
    #     graph=graph, outcomes=target_outcomes, treatments=target_interventions
    # )
    # if vanilla_estimand is not None:
    #     logger.warning(f"This query is identifiable without surrogates: {vanilla_estimand}")

    check_and_raise_missing(target_outcomes, graph, "target_outcomes")
    check_and_raise_missing(target_interventions, graph, "target_interventions")
    check_and_raise_missing(set().union(*surrogate_outcomes.values()), graph, "surrogate_outcomes")
    check_and_raise_missing(
        set().union(*surrogate_interventions.values()), graph, "surrogate_interventions"
    )
    outcome_is_intervention = target_outcomes.intersection(target_interventions)
    if outcome_is_intervention:
        raise ValueError(
            f"The variables {outcome_is_intervention} cannot be target_outcomes and target_interventions"
        )

    transport_query = surrogate_to_transport(
        graph=graph,
        target_outcomes=target_outcomes,
        target_interventions=target_interventions,
        surrogate_outcomes=surrogate_outcomes,
        surrogate_interventions=surrogate_interventions,
    )
    initial_expression = PopulationProbability(
        population=TARGET_DOMAIN,
        distribution=Distribution.safe(graph.nodes()),
    )
    trso_query = TRSOQuery(
        target_interventions=transport_query.target_interventions,
        target_outcomes=transport_query.target_outcomes,
        expression=initial_expression,
        active_interventions=set(),
        domain=TARGET_DOMAIN,
        domains=transport_query.domains,
        graphs=transport_query.graphs,
        surrogate_interventions=transport_query.surrogate_interventions,
    )
    return trso(trso_query)
