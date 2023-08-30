"""Implement of surrogate outcomes and transportability from https://arxiv.org/abs/1806.07172."""

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Union, cast

from y0.algorithm.conditional_independencies import are_d_separated
from y0.dsl import (
    PP,
    CounterfactualVariable,
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
)
from y0.graph import NxMixedGraph
from y0.mutate.canonicalize_expr import canonicalize

__all__ = [
    "transport",
    "TransportQuery",
]

logger = logging.getLogger(__name__)

TARGET_DOMAIN = Population("pi*")


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
    if isinstance(surrogate_interventions, Variable):
        surrogate_interventions = {surrogate_interventions}
    if isinstance(surrogate_outcomes, Variable):
        surrogate_outcomes = {surrogate_outcomes}

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
    """Find all of the transport nodes in a graph.

    :param graph: an NxMixedGraph which may have transport nodes
    :returns: Set containing all transport nodes in the graph
    """
    return {node for node in graph.nodes() if is_transport_node(node)}


def get_regular_nodes(graph: NxMixedGraph) -> Set[Variable]:
    """Find all of the nodes in a graph which are not transport nodes.

    :param graph: an NxMixedGraph
    :returns: Set containing all nodes which are not transport nodes
    """
    return {node for node in graph.nodes() if not is_transport_node(node)}


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
    # TODO we discussed the possibility of using a dictionary with needed nodes
    #  instead of creating a graph for each diagram.
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
    target_outcomes: Set[Variable],
    target_interventions: Set[Variable],
    graph: NxMixedGraph,
    surrogate_outcomes: Dict[Population, Set[Variable]],
    surrogate_interventions: Dict[Population, Set[Variable]],
) -> TransportQuery:
    """Create transportability diagrams and query from a surrogate outcome problem.

    :param target_outcomes: A set of target variables for causal effects.
    :param target_interventions: A set of interventions for the target domain.
    :param graph: The graph of the target domain.
    :param surrogate_outcomes: A dictionary of outcomes in other populations
    :param surrogate_interventions: A dictionary of interentions in other populations
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
    new_query.graphs[new_query.domain] = query.graphs[query.domain].subgraph(outcomes_ancestors)
    if isinstance(query.expression, PopulationProbability):
        # This is true by the chain rule and marginalizing
        new_query.expression = PP[query.domain](
            set(query.expression.children).intersection(outcomes_ancestors)
        )
    else:
        logger.debug(
            "Calling trso algorithm line 2 else loop",
            query.expression,
        )
        new_query.expression = Sum.safe(
            query.expression,
            get_regular_nodes(query.graphs[query.domain]) - outcomes_ancestors,
        )

    return new_query


def trso_line3(query: TRSOQuery, additional_interventions: Set[Variable]) -> TRSOQuery:
    """Add nodes that will effect the outcome to the interventions of the TRSOQuery.

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


def add_active_interventions(
    expression: Expression,
    active_interventions: Set[Variable],
    target_outcomes: Set[Variable],  # FIXME this doesn't appear to be used
) -> Expression:
    """Intervene on the target variables of expression using the active interventions.

    :param expression: A probability expression.
    :param active_interventions: Set of active interventions
    :param target_outcomes: Set of outcomes on which we will intervene
    :returns: boolean True if all interventions are d-separated from all outcomes, False otherwise.
    :raises NotImplementedError: If an expression type that is not handled gets passed
    """
    if isinstance(expression, Probability):
        return expression.intervene(active_interventions)
    if isinstance(expression, Sum):
        intervened_expression = add_active_interventions(
            expression.expression, active_interventions, target_outcomes
        )
        intervened_ranges = tuple(
            variable.intervene(active_interventions) for variable in expression.ranges
        )
        return Sum.safe(intervened_expression, intervened_ranges)
    if isinstance(expression, Fraction):
        new_numerator = add_active_interventions(
            expression.numerator, active_interventions, target_outcomes
        )
        new_denominator = add_active_interventions(
            expression.denominator, active_interventions, target_outcomes
        )
        return cast(Fraction, new_numerator / new_denominator).simplify()
    if isinstance(expression, Product):
        intervened_expression = add_active_interventions(
            # FIXME, this should be handled in a loop over `expression.expressions`,
            #  not expression.expression (note the plural)
            expression.expression,
            active_interventions,
            target_outcomes,
        )
        # FIXME product doesn't have ranges
        intervened_ranges = tuple(
            variable.intervene(active_interventions) for variable in expression.ranges
        )
        # FIXME doesn't make sense to return a sum when handling product, was this
        #  copy+pasted?
        return Sum.safe(intervened_expression, intervened_ranges)
    raise NotImplementedError(f"Unhandled expression type: {type(expression)}")


def all_transports_d_separated(graph, target_interventions, target_outcomes) -> bool:
    """Check if all target_interventions are d-separated from target_outcomes.

    :param graph: The graph with transport nodes in this domain.
    :param target_interventions: Set of target interventions
    :param target_outcomes: Set of target interventions
    :returns: boolean True if all interventions are d-separated from all outcomes, False otherwise.
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
        if transportability_node
        in graph_without_interventions  # FIXME check if this is okay to exclude
        for outcome in target_outcomes
    )


def trso_line9(query: TRSOQuery, district: set[Variable]) -> Expression:
    """Return the probability in the case with exactly one districts_without_interventions and it is present in districts.

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
    # first simplify before this check
    if isinstance(query.expression, Zero):
        # TODO is this possible to be zero, should we have a check?
        # from charlie: no, it's not possible to be zero, since you're
        # wrapping some expression in a sum. This comparison doesn't actually
        # make sense, either, since this is a DSL object and not an integer.
        # however, if you do some kind of processing/evaluation, then you
        # might be able to find out if it's zero
        raise RuntimeError

    ordering = list(query.graphs[query.domain].topological_sort())
    ordering_set = set(ordering)  # TODO this is just all nodes in the graph
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
    :returns: An modified TRSOQuery
    """
    ordering = list(query.graphs[query.domain].topological_sort())
    ordering_set = set(ordering)  # TODO this is just all nodes in the graph
    my_product: Expression = One()
    for node in district:
        i = ordering.index(node)
        pre, post = ordering[:i], ordering[: i + 1]
        pre_set = ordering_set - set(post)
        post_set = ordering_set - set(pre)
        numerator = Sum.safe(query.expression, pre_set)
        denominator = Sum.safe(query.expression, post_set)
        my_product *= numerator / denominator
    my_product = cast(Fraction, my_product).simplify()  # FIXME duplicate code?

    expressions = []
    for node in district:
        i = ordering.index(node)
        pre_node = set(ordering[:i])
        # FIXME Doesn't this expression just return pre_node?
        prob = Probability.safe(
            node.given(pre_node.intersection(district).union(pre_node - district))
        )
        # or is it supposed to be this?
        prob = Probability.safe(
            node.given(pre_node.intersection(district).union(set(ordering[i - 1]) - district))
        )
        expressions.append(
            PopulationProbability(population=query.domain, distribution=prob.distribution)
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
    :raises NotImplementedError: when a part of the algorithm is not yet handled
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
        # "\t- graphs: %s\n"
        "\t- surrogate_interventions: %s",
        query.target_interventions,
        query.target_outcomes,
        query.expression,
        query.active_interventions,
        query.domain,
        query.domains,
        # query.graphs,
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
        return canonicalize(trso(new_query))

    # line 3
    additional_interventions = graph.get_no_effect_on_outcomes(
        query.target_interventions, query.target_outcomes
    )
    if additional_interventions:
        new_query = trso_line3(query, additional_interventions)
        logger.debug("Calling trso algorithm line 3")
        return canonicalize(trso(new_query))

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
                raise NotImplementedError
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
    if not query.active_interventions:
        expressions: Dict[Population, Expression] = {}
        for domain, subquery in trso_line6(query).items():
            logger.debug("Calling trso algorithm line 6 for domain %s", domain)
            expression = trso(subquery)
            if expression is None:
                raise NotImplementedError
            expression = add_active_interventions(
                expression, subquery.active_interventions, subquery.target_outcomes
            )
            if expression is not None:  # line7
                logger.debug(
                    "Calling trso algorithm line 7",
                )
                expressions[domain] = expression
        if len(expressions) == 1:
            return canonicalize(list(expressions.values())[0])
        elif len(expressions) > 1:
            logger.warning("more than one expression were non-none")
            # What if more than 1 expression doesn't fail?
            # Is it non-deterministic or can we prove it will be length 1?
            return canonicalize(list(expressions.values())[0])
        else:
            pass

    # line8
    districts = graph.districts()
    # line 11, return fail. keep explict tests for 0 and 1 to ensure adequate testing
    if len(districts) == 0:
        logger.debug(
            "Fail on algorithm line 11 (length of districts equals 0)",
        )
        return None
    if len(districts) == 1:
        logger.debug(
            "Fail on trso algorithm line 11 (length of districts equals 1)",
        )
        return None
    # line 8, i.e. len(districts)>1

    # line9
    # TODO check which of the raises below should be passthroughs, annotate explicitly
    if len(districts_without_interventions) == 1:
        district_without_interventions = districts_without_interventions.pop()
        if district_without_interventions in districts:
            return canonicalize(trso_line9(query, set(district_without_interventions)))
        # raise NotImplementedError(
        #     "single district without interventions found, but it's not in the districts"
        # )
        # This case is covered by line 10.
        # FIXME ^ doesn't seem quite right since this is exact checking while line 10 is subsets
    elif len(districts_without_interventions) == 0:
        raise NotImplementedError("no districts without interventions found")
    else:  # multiple districts
        raise NotImplementedError("multiple districts without interventions found")

    # line10
    target_districts = []
    for district in districts:
        if district_without_interventions.issubset(district):
            target_districts.append(district)
    if len(target_districts) != 1:
        logger.warning("Incorrect number of districts found on line 10")
        # TODO This shouldn't be possible, should we remove this check?
    target_district = target_districts.pop()
    # district is C' districts should be D[C'], but we chose to return set of nodes instead of subgraph
    if len(query.active_interventions) == 0:
        # FIXME is this even possible? doesn't line 6 check this and return something else?
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
    raise NotImplementedError


def _pillow_has_transport(graph, district) -> bool:
    return any(is_transport_node(node) for node in graph.get_markov_pillow(district))


def transport(
    graph: NxMixedGraph,
    transports: Mapping[Variable, List[Population]],
    treatments: List[Variable],
    outcomes: List[Variable],
    conditions: Optional[List[Variable]] = None,
):
    """Transport algorithm from https://arxiv.org/abs/1806.07172."""
    if conditions is not None:
        raise NotImplementedError
    raise NotImplementedError
