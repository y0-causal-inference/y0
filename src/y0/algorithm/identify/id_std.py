# -*- coding: utf-8 -*-

"""An implementation of the identification algorithm."""

from typing import List, Sequence

from .utils import Identification, Unidentifiable
from ...dsl import Expression, P, Probability, Product, Sum, Variable
from ...graph import NxMixedGraph

__all__ = [
    "identify",
]


def identify(identification: Identification) -> Expression:
    """Run the ID algorithm from [shpitser2006]_.

    :param identification: The identification tuple
    :returns: the expression corresponding to the identification
    :raises Unidentifiable: If no appropriate identification can be found

    See also :func:`identify_outcomes` for a more idiomatic way of running
    the ID algorithm given a graph, treatments, and outcomes.
    """
    graph = identification.graph
    treatments = identification.treatments
    outcomes = identification.outcomes
    vertices = set(graph.nodes())

    # line 1
    if not treatments:
        return line_1(identification)

    # line 2
    outcomes_and_ancestors = graph.ancestors_inclusive(outcomes)
    not_outcomes_or_ancestors = vertices.difference(outcomes_and_ancestors)
    if not_outcomes_or_ancestors:
        return identify(line_2(identification))

    # line 3
    no_effect_on_outcome = graph.get_no_effect_on_outcomes(treatments, outcomes)
    if no_effect_on_outcome:
        return identify(line_3(identification))

    # line 4
    graph_without_treatments = graph.remove_nodes_from(treatments)
    if not graph_without_treatments.is_connected():
        expression = Product.safe(map(identify, line_4(identification)))
        return Sum.safe(
            expression=expression,
            ranges=vertices.difference(outcomes | treatments),
        )

    # line 5
    if graph.is_connected():  # e.g., there's only 1 c-component, and it encompasses all vertices
        raise Unidentifiable(graph.nodes(), graph_without_treatments.districts())

    # line 6
    district_without_treatment = _get_single_district(graph_without_treatments)

    if district_without_treatment in graph.districts():
        parents = list(graph.topological_sort())
        expression = Product.safe(p_parents(v, parents) for v in district_without_treatment)
        ranges = district_without_treatment - outcomes
        return Sum.safe(
            expression=expression,
            ranges=ranges,
        )

    # line 7
    return identify(line_7(identification))


def _get_single_district(graph: NxMixedGraph) -> frozenset[Variable]:
    districts = graph.districts()
    if len(districts) != 1:
        raise RuntimeError
    return districts.pop()


def line_1(identification: Identification) -> Expression:
    r"""Run line 1 of identification algorithm.

    If no action has been taken, the effect on :math:`\mathbf Y` is just the marginal of
    the observational distribution

    :param identification: The data structure with the treatment, outcomes, estimand, and graph
    :returns:  The marginal of the outcome variables
    """
    outcomes = identification.outcomes
    vertices = set(identification.graph.nodes())
    return Sum.safe(
        expression=identification.estimand,
        ranges=vertices.difference(outcomes),
    )


def line_2(identification: Identification) -> Identification:
    r"""Run line 2 of the identification algorithm.

    If we are interested in the effect on :math:`\mathbf Y`, it is sufficient to restrict our attention
    on the parts of the model ancestral to :math:`\mathbf Y`.

    .. math::

        \text{if }\mathbf V - An(\mathbf Y)_G \neq \emptyset \\
        \text{ return } \mathbf{ ID}\left(\mathbf y, \mathbf x\cap An(\mathbf Y)_G, \sum_{\mathbf V -
         An(Y)_G}P, G_{An(\mathbf Y)}\right)

    :param identification: The data structure with the treatment, outcomes, estimand, and graph
    :returns: The new estimand
    :raises ValueError: If the line 2 precondition is not met
    """
    graph = identification.graph
    treatments = identification.treatments
    outcomes = identification.outcomes

    vertices = set(graph.nodes())
    outcomes_and_ancestors = graph.ancestors_inclusive(outcomes)
    not_outcomes_or_ancestors = vertices.difference(outcomes_and_ancestors)
    outcome_ancestral_graph = graph.subgraph(outcomes_and_ancestors)

    if not not_outcomes_or_ancestors:
        raise ValueError("line 2 precondition not met")
    return Identification.from_parts(
        outcomes=outcomes,
        treatments=treatments & outcomes_and_ancestors,
        estimand=Sum.safe(expression=identification.estimand, ranges=not_outcomes_or_ancestors),
        graph=outcome_ancestral_graph,
    )


def line_3(identification: Identification) -> Identification:
    r"""Run line 3 of the identification algorithm.

    Forces an action on any node where such an action would have no
    effect on :math:\mathbf Y`—assuming we already acted on
    :math:`\mathbf X`. Since actions remove incoming arrows, we can
    view line 3 as simplifying the causal graph we consider by
    removing certain arcs from the graph, without affecting the
    overall answer.

    :param identification: The data structure with the treatment, outcomes, estimand, and graph
    :returns: The new estimand
    :raises ValueError: If the preconditions for line 3 aren't met.
    """
    outcomes = identification.outcomes
    treatments = identification.treatments
    graph = identification.graph

    no_effect_on_outcome = graph.get_no_effect_on_outcomes(treatments, outcomes)
    if not no_effect_on_outcome:
        raise ValueError(
            'Line 3 precondition not met. There were no variables in "no_effect_on_outcome"'
        )

    return identification.with_treatments(no_effect_on_outcome)


def line_4(identification: Identification) -> List[Identification]:
    r"""Run line 4 of the identification algorithm.

    The key line of the algorithm, it decomposes the problem into a set
    of smaller problems using the key property of *c-component
    factorization* of causal models. If the entire graph is a single
    C-component already, further problem decomposition is impossible,
    and we must provide base cases. :math:`\mathbf{ID}` has three base
    cases.

    :param identification: The data structure with the treatment, outcomes, estimand, and graph
    :returns: A list of new estimands
    :raises ValueError: If the precondition that there are more than 1 districts without treatments is not met
    """
    treatments = identification.treatments
    estimand = identification.estimand
    graph = identification.graph
    vertices = set(graph.nodes())

    # line 4
    graph_without_treatments = graph.remove_nodes_from(treatments)
    districts_without_treatment = graph_without_treatments.districts()
    if len(districts_without_treatment) <= 1:
        raise ValueError("Line 4 precondition not met")
    return [
        Identification.from_parts(
            outcomes=set(district_without_treatment),
            treatments=vertices - district_without_treatment,
            estimand=estimand,
            graph=graph,
        )
        for district_without_treatment in districts_without_treatment
    ]


def line_5(identification: Identification) -> None:
    r"""Run line 5 of the identification algorithm.

    Fails because it finds two C-components, the graph :math:`G`
    itself, and a subgraph :math:`S` that does not contain any
    :math:`\mathbf X` nodes. But that is exactly one of the properties
    of C-forests that make up a hedge. In fact, it turns out that it
    is always possible to recover a hedge from these two c-components.

    :param identification: The data structure with the treatment, outcomes, estimand, and graph
    :raises Unidentifiable: If line 5 realizes that identification is not possible
    """
    treatments = identification.treatments
    graph = identification.graph
    vertices = set(graph.nodes())
    graph_without_treatments = graph.remove_nodes_from(treatments)
    districts_without_treatment = graph_without_treatments.districts()

    # line 5
    districts = graph.districts()
    if districts == {frozenset(vertices)}:
        raise Unidentifiable(districts, districts_without_treatment)


def line_6(identification: Identification) -> Expression:
    r"""Run line 6 of the identification algorithm.

    Asserts that if there are no bidirected arcs from :math:`X` to the other nodes in the current subproblem
    under consideration, then we can replace acting on :math:`X` by conditioning, and thus solve the subproblem.

    ..math::

        \text{ if }S\in C(G) \\
        \text{ return }\sum_{S - \mathbf y}\prod_{\{i|V_i\in S\}}P\left(v_i|v_\pi^{(i-1)}\right)

    :param identification: The data structure with the treatment, outcomes, estimand, and graph
    :returns: A list of new estimands
    :raises ValueError: If line 6 precondition is not met
    """
    outcomes = identification.outcomes
    treatments = identification.treatments
    graph = identification.graph

    districts = graph.districts()
    graph_without_treatments = graph.remove_nodes_from(treatments)
    district_without_treatments = _get_single_district(graph_without_treatments)

    # line 6
    if district_without_treatments not in districts:
        raise ValueError("Line 6 precondition not met")

    parents = list(graph.topological_sort())
    expression = Product.safe(p_parents(v, parents) for v in district_without_treatments)
    ranges = district_without_treatments - outcomes
    return Sum.safe(
        expression=expression,
        ranges=ranges,
    )


def line_7(identification: Identification) -> Identification:
    r"""Run line 7 of the identification algorithm.

    The most complex case where :math:`\mathbf X` is partitioned into
    two sets, :math:`\mathbf W` which contain bidirected arcs into
    other nodes in the subproblem, and :math:`\mathbf Z` which do
    not. In this situation, identifying :math:`P(\mathbf y|do(\mathbf
    x))` from :math:`P(v)` is equivalent to identifying
    :math:`P(\mathbf y|do(\mathbf w))` from :math:`P(\mathbf
    V|do(\mathbf z))`, since :math:`P(\mathbf y|do(\mathbf x)) =
    P(\mathbf y|do(\mathbf w), do(\mathbf z))`. But the term
    :math:`P(\mathbf V|do(\mathbf z))` is identifiable using the
    previous base case, so we can consider the subproblem of
    identifying :math:`P(\mathbf y|do(\mathbf w))`

    .. math::

       \text{ if }(\exists S')S\subset S'\in C(G) \\
       \text{ return }\mathbf{ID}\left(\mathbf y, \mathbf x\cap S',
       \prod_{\{i|V_i\in S'\}}P(V_i|V_\pi^{(i-1)}\cap S', V_\pi^{(i-1)} -
       S'), G_{S'}\right)

    :param identification: The data structure with the treatment, outcomes, estimand, and graph
    :returns: A new estimand
    :raises ValueError: If line 7 does not find a suitable district
    """
    outcomes = identification.outcomes
    treatments = identification.treatments
    graph = identification.graph

    graph_without_treatments = graph.remove_nodes_from(treatments)
    # line 7 precondition requires single district
    district_without_treatments = _get_single_district(graph_without_treatments)

    # line 7
    for district in graph.districts():
        if district_without_treatments < district:
            parents = list(graph.topological_sort())
            return Identification.from_parts(
                outcomes=outcomes,
                treatments=treatments & district,
                estimand=Product.safe(p_parents(v, parents) for v in district),
                graph=graph.subgraph(district),
            )

    raise ValueError("Could not identify suitable district")


def p_parents(child: Variable, ordering: Sequence[Variable]) -> Probability:
    """Get a probability expression based on a topological ordering.

    :param child: The child variable
    :param ordering: A topologically ordered sequence of all variables. All occurring before the
        child will be used as parents.
    :return: A probability expression
    """
    return P(child | ordering[: ordering.index(child)])
