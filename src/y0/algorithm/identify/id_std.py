# -*- coding: utf-8 -*-

"""An implementation of the identification algorithm."""

from typing import List, TypeVar

from y0.dsl import Expression, P, Probability, Product, Sum
from .utils import Fail, Identification

X = TypeVar("X")


def identify(identification: Identification) -> Expression:
    """Run the identification algorithm.

    :param identification: The data structure with the treatment, outcomes, estimand, and graph
    :returns: the expression corresponding to the identification
    :raises Fail: If no appropriate identification can be found
    """
    outcomes = identification.outcome_variables
    treatments = identification.treatment_variables
    graph = identification.graph.str_nodes_to_variable_nodes()
    vertices = set(graph.nodes())

    # line 1
    if not treatments:
        return Sum.safe(expression=P(vertices), ranges=vertices.difference(outcomes))

    # line 2
    outcomes_and_ancestors = graph.ancestors_inclusive(outcomes)
    not_outcomes_or_ancestors = vertices.difference(outcomes_and_ancestors)
    if not_outcomes_or_ancestors:
        return identify(line_2(identification))

    # line 3
    intervened_graph = graph.intervene(treatments)
    no_effect_on_outcome = (vertices - treatments) - intervened_graph.ancestors_inclusive(outcomes)
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
        raise Fail(graph.nodes(), graph_without_treatments.get_c_components())

    # line 6
    districts = graph.get_c_components()
    parents = graph.get_topological_sort()

    # There can be only 1 district without treatments because of line 4
    districts_without_treatment = graph_without_treatments.get_c_components()
    district_without_treatment = districts_without_treatment[0]

    if district_without_treatment in districts:
        expression = Product.safe(p_parents(v, parents) for v in district_without_treatment)
        ranges = district_without_treatment - outcomes
        if not ranges:
            return expression
        return Sum.safe(
            expression=expression,
            ranges=ranges,
        )

    # line 7
    return identify(line_7(identification))


def line_1(identification: Identification) -> Expression:
    r"""Run line 1 of identification algorithm.

    If no action has been taken, the effect on :math:`\mathbf Y` is just the marginal of
    the observational distribution

    :param identification: The data structure with the treatment, outcomes, estimand, and graph
    :returns:  The marginal of the outcome variables
    """
    outcomes = identification.outcome_variables
    graph = identification.graph.str_nodes_to_variable_nodes()
    vertices = set(graph.nodes())
    return Sum.safe(
        expression=P(vertices),
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
    outcomes = identification.outcome_variables
    treatments = identification.treatment_variables
    estimand = identification.estimand
    graph = identification.graph.str_nodes_to_variable_nodes()
    vertices = set(graph.nodes())
    outcomes_and_ancestors = graph.ancestors_inclusive(outcomes)
    not_outcomes_or_ancestors = vertices.difference(outcomes_and_ancestors)
    outcome_ancestral_graph = graph.subgraph(outcomes_and_ancestors)

    if not not_outcomes_or_ancestors:
        raise ValueError("line 2 precondition not met")
    return Identification.from_parts(
        outcomes=outcomes,
        treatments=treatments & outcomes_and_ancestors,
        estimand=Sum.safe(expression=estimand, ranges=not_outcomes_or_ancestors),
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
    outcomes = identification.outcome_variables
    treatments = identification.treatment_variables
    estimand = identification.estimand
    graph = identification.graph.str_nodes_to_variable_nodes()
    vertices = set(graph.nodes())

    intervened_graph = graph.intervene(treatments)
    no_effect_on_outcome = (vertices - treatments) - intervened_graph.ancestors_inclusive(outcomes)
    if no_effect_on_outcome:
        return Identification.from_parts(
            outcomes=outcomes,
            treatments=treatments | no_effect_on_outcome,
            estimand=estimand,
            graph=graph,
        )

    # TODO what happens if it gets here
    raise ValueError


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
    treatments = identification.treatment_variables
    estimand = identification.estimand
    graph = identification.graph.str_nodes_to_variable_nodes()
    vertices = set(graph.nodes())

    # line 4
    graph_without_treatments = graph.remove_nodes_from(treatments)
    districts_without_treatment = graph_without_treatments.get_c_components()
    if len(districts_without_treatment) <= 1:
        raise ValueError("Line 4 precondition not met")
    return list(
        Identification.from_parts(
            outcomes=district_without_treatment,
            treatments=vertices - district_without_treatment,
            estimand=estimand,
            graph=graph,
        )
        for district_without_treatment in districts_without_treatment
    )


def line_5(identification: Identification) -> None:
    r"""Run line 5 of the identification algorithm.

    Fails because it finds two C-components, the graph :math:`G`
    itself, and a subgraph :math:`S` that does not contain any
    :math:`\mathbf X` nodes. But that is exactly one of the properties
    of C-forests that make up a hedge. In fact, it turns out that it
    is always possible to recover a hedge from these two c-components.

    :param identification: The data structure with the treatment, outcomes, estimand, and graph
    :raises Fail: If line 5 realizes that identification is not possible
    """
    # outcomes = identification.outcomes
    treatments = identification.treatment_variables
    # estimand = identification.estimand
    graph = identification.graph.str_nodes_to_variable_nodes()
    vertices = set(graph.nodes())
    # outcomes_and_ancestors = ancestors_and_self(graph, outcomes)
    # not_outcomes_or_ancestors = vertices.difference(outcomes_and_ancestors)
    # outcome_ancestral_graph = graph.subgraph(outcomes_and_ancestors)
    graph_withoyt_treatments = graph.remove_nodes_from(treatments)
    districts_without_treatment = graph_withoyt_treatments.get_c_components()

    # line 5
    districts = graph.get_c_components()
    if districts == [frozenset(vertices)]:
        raise Fail(districts, districts_without_treatment)


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
    outcomes = identification.outcome_variables
    treatments = identification.treatment_variables
    graph = identification.graph.str_nodes_to_variable_nodes()

    districts = graph.get_c_components()
    graph_without_treatments = graph.remove_nodes_from(treatments)
    districts_without_treatments = graph_without_treatments.get_c_components()

    # line 6
    parents = graph.get_topological_sort()
    if districts_without_treatments[0] not in districts:
        raise ValueError("Line 6 precondition not met")
    expression = Product.safe(p_parents(v, parents) for v in districts_without_treatments[0])
    ranges = districts_without_treatments[0] - outcomes
    if not ranges:
        return expression
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
    outcomes = identification.outcome_variables
    treatments = identification.treatment_variables
    graph = identification.graph.str_nodes_to_variable_nodes()

    districts = graph.get_c_components()
    graph_without_treatments = graph.remove_nodes_from(treatments)
    districts_without_treatments = graph_without_treatments.get_c_components()
    parents = graph.get_topological_sort()

    # line 7
    for district in districts:
        if districts_without_treatments[0] < district:
            return Identification.from_parts(
                outcomes=outcomes,
                treatments=treatments & district,
                estimand=Product.safe(p_parents(v, parents) for v in district),
                graph=graph.subgraph(district),
            )

    raise ValueError("Could not identify suitable district")


def p_parents(variable, parents) -> Probability:
    """Get a probability expression based on a topological ordering."""
    return P(variable | parents[: parents.index(variable)])


# def str_list(node_list):
#     """ return a string listing the nodes in node_list - this is used in the ID and IDC algorithms """
#     str_out = ''

#     for item in node_list:
#         str_out += item + ','

#     return str_out[:-1]


# def id_alg(graph: NxMixedGraph, y, x, p_in=None):
#     """calculate P(y | do(x)) or return failure if this is not possible"""
#     graph_temp = graph.directed
#     graph_c = graph.undirected

#     if p_in is None:
#         p_expr = "P(" + str_list(graph.directed.nodes()) + ")"
#     else:
#         p_expr = p_in

#     if np.any([item in y for item in x]):
#         print("Error -- overlap between x and y")
#         print(x)
#         print(y)
#         print(p_in)
#         print(graph_temp.nodes)
#         raise ValueError

#     y_anc = y.copy()

#     # identify ancestors of y
#     for item in y:
#         set_temp = nx.algorithms.dag.ancestors(graph_temp, item)
#         y_anc += [item2 for item2 in set_temp if item2 not in y_anc]

#     # identify all nodes in the graph
#     v_not_anc_y = [item for item in graph_temp.nodes if item not in y_anc]

#     # remove edges to x
#     graph_xbar = nx.DiGraph(graph_temp)
#     for item in x:
#         graph_xbar.remove_edges_from(list(graph_temp.in_edges(item)))

#     y_anc_x_bar = y.copy()

#     for item in y:
#         set_temp = nx.algorithms.dag.ancestors(graph_xbar, item)
#         y_anc_x_bar += [item2 for item2 in set_temp if item2 not in y_anc_x_bar]

#     w_set = [
#         item for item in graph_temp.nodes if item not in x and item not in y_anc_x_bar
#     ]

#     # line 1
#     if not x:
#         # return sum over all non-y variables

#         node_list = [item for item in graph_temp.nodes if item not in y]
#         str_out = "[sum_{" + str_list(node_list) + "} " + p_expr + "]"
#         # print('Step 1')

#         return str_out

#     # line 2
#     elif v_not_anc_y:

#         x_temp = [item for item in y_anc if item in x]
#         str_out = "[sum_{" + str_list(v_not_anc_y) + "} " + p_expr + "]"
#         graph_anc = graph_temp.subgraph(y_anc)

#         # print('Begin Step 2')
#         # print(v_not_anc_y)
#         expr_out = id_alg(y, x_temp, str_out, graph_anc)
#         # print('End Step 2')

#         return expr_out

#     # line 3
#     elif w_set:
#         # print('Begin Step 3')
#         # print(w_set)
#         expr_out = id_alg(y, x + w_set, p_expr, graph_temp)
#         # print('End Step 3')

#         return expr_out

#     else:
#         # calculate graph C-components
#         graph_temp_c = nx.Graph(graph_c.subgraph(graph_temp.nodes))
#         graph_temp_c.remove_nodes_from(x)
#         s_sets = [list(item) for item in nx.connected_components(graph_temp_c)]

#         # line 4
#         if len(s_sets) > 1:
#             # print('Begin Step 4')
#             # print(s_sets)
#             node_list = [
#                 item for item in graph_temp.nodes if item not in y and item not in x
#             ]
#             str_out = "[sum_{" + str_list(node_list) + "} "

#             for item in s_sets:
#                 v_s_set = [item2 for item2 in graph_temp.nodes if item2 not in item]
#                 s_in = [item2 for item2 in item]

#                 if np.any([item2 in v_s_set for item2 in s_in]):
#                     print("Error -- x/y overlap")
#                     print(v_s_set)
#                     print(s_in)

#                 str_out += id_alg(graph, s_in, v_s_set, p_expr)

#             # print('End Step 4')
#             str_out += "]"

#             return str_out

#         else:
#             graph_temp_c_prime = graph_c.subgraph(graph_temp.nodes)

#             s_sets_prime = [
#                 list(item) for item in nx.connected_components(graph_temp_c_prime)
#             ]

#             # line 5
#             if sorted(s_sets_prime[0]) == sorted(graph_temp.nodes):

#                 node_list = [ind for ind in s_sets_prime[0]]
#                 node_list2 = [ind for ind in graph_temp.nodes if ind in s_sets[0]]

#                 str_out = (
#                     "FAIL("
#                     + str_list(node_list)
#                     + ","
#                     + str_list(node_list2)
#                     + ")"
#                 )

#                 # print('Step 5')
#                 return str_out

#             # line 6
#             elif np.any([sorted(s_sets[0]) == sorted(item) for item in s_sets_prime]):

#                 node_list = [item for item in s_sets[0] if item not in y]
#                 str_out = "[sum_{" + str_list(node_list) + "}"

#                 for item in s_sets[0]:
#                     # identify parents of node i
#                     parents = list(graph_temp.predecessors(item))

#                     if parents:
#                         str_out += "P(" + item + "|" + str_list(parents) + ")"
#                     else:
#                         str_out += "P(" + item + ")"
#                 # print(s_sets[0])
#                 # print('Step 6')
#                 return str_out + "]"

#             # line 7
#             elif np.any(
#                 [
#                     np.all([item in item2 for item in s_sets[0]])
#                     for item2 in s_sets_prime
#                 ]
#             ):
#                 ind = np.where(
#                     [
#                         np.all([item in item2 for item in s_sets[0]])
#                         for item2 in s_sets_prime
#                     ]
#                 )[0][0]

#                 graph_prime = graph_temp.subgraph(s_sets_prime[ind])
#                 x_prime = [item for item in s_sets_prime[ind] if item in x]
#                 str_out = ""

#                 for item in s_sets_prime[ind]:

#                     pred = list(nx.algorithms.dag.ancestors(graph_temp, item))
#                     par_set = [item2 for item2 in pred if item2 in s_sets_prime[ind]]
#                     par_set += [
#                         item2 for item2 in pred if item2 not in s_sets_prime[ind]
#                     ]

#                     if par_set:
#                         str_out += "P(" + item + "|" + str_list(par_set) + ")"
#                     else:
#                         str_out += "P(" + item + ")"

#                 # print('Begin Step 7')
#                 # print((s_sets[0],s_sets_prime[ind]))

#                 if np.any([item2 in x_prime for item2 in y]):
#                     print("Error -- x/y overlap")
#                     print(x_prime)
#                     print(y)

#                 expr_out = id_alg(y, x_prime, str_out, graph_prime)
#                 # print('End Step 7')

#                 return expr_out

#             else:

#                 print("error")
#                 return ""
