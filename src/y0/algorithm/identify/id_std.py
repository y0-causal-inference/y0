# -*- coding: utf-8 -*-

from typing import Union, Set, List
from ananke.graphs import ADMG
import networkx as nx
from y0.algorithm.identify.utils import (
    ancestors_and_self,
    nxmixedgraph_to_causal_graph,
    outcomes_and_treatments_to_query,
    query_to_outcomes_and_treatments,
)
from y0.dsl import Expression, P, Sum, Variable
from y0.graph import NxMixedGraph
from y0.identify import _get_outcomes, _get_treatments
from .utils import Identification


def identify(graph: Union[ADMG, NxMixedGraph], query: Expression) -> Expression:
    """Currently a wrapper for y0.algorithm.identifiy.utils.causal_graph.id_alg()"""
    if isinstance(graph, ADMG):
        graph = NxMixedGraph.from_admg(graph)
    treatments = _get_treatments(query.get_variables())
    outcomes = _get_outcomes(query.get_variables())
    cg = nxmixedgraph_to_causal_graph(graph)
    expr = cg.id_alg(outcomes, treatments)
    return expr
    # expr = id_alg(graph, outcomes, treatments)
    # try:
    #     r = grammar.parseString(expr)
    # except ParseException:
    #     raise ValueError(f'graph produced unparsable expression: {expr}')
    # else:
    #     return r[0]


def ID(
    *, outcomes: Set[str], treatments: Set[str], estimand: Expression, G: NxMixedGraph
) -> Expression:
    # line 1
    if len(treatments) == 0:
        return line_1(outcomes, treatments, estimand, G)

    # line 2
    if len(G.directed.nodes() - ancestors_and_self(G, outcomes)) > 0:
        return line_2(outcomes, treatments, estimand, G)

    # line 3


def line_1(
    *, outcomes: Set[str], treatments: Set[str], estimand: Expression, G: NxMixedGraph
) -> Expression:
    r"""Line 1 of ID algorithm.

    If no action has been taken, the effect on :math:`\mathbf Y` is just the marginal of
    the observational distribution
        :param outcomes:
        :param interventions:
        :param estimand: Probabilistic expression
        :param G: NxMixedGraph
        :returns:  The marginal of the outcome variables
        :raises ValueError:  There should not be any interventional variables
    """

    V = set(G.nodes())
    return Sum(
        P(*[Variable(v) for v in V]),
        tuple(Variable(v) for v in V.difference(outcomes)),
    )


def line_2(
    *, outcomes: Set[str], treatments: Set[str], estimand: Expression, G: NxMixedGraph
) -> Identification:
    r"""Line 2 of the ID algorithm.

    If we are interested in the effect on :math:`\mathbf Y`, it is sufficient to restrict our attention
    on the parts of the model ancestral to :math:`\mathbf Y`.

.. math::
    \text{if }\mathbf V - An(\mathbf Y)_G \neq \emptyset \\
    \text{ return } \mathbf{ ID}\left(\mathbf y, \mathbf x\cap An(\mathbf Y)_G, \sum_{\mathbf V - An(Y)_G}P, G_{An(\mathbf Y)}\right)

    :param outcomes: set of outcomes
    :param treatments: set of interventions
    :param estimand: Probabilistic expression
    :param G: ADMG
    :returns: The probability expression
    :raises Fail: if the query is not identifiable.
    """
    V = set(G.nodes())
    ancestors_and_Y_in_G = ancestors_and_self(G, outcomes)
    not_ancestors_of_Y = V.difference(ancestors_and_Y_in_G)
    G_ancestral_to_Y = G.subgraph(ancestors_and_Y_in_G)
    if len(not_ancestors_of_Y) == 0:
        raise ValueError("No ancestors of Y")
    return Identification(
        query = outcomes_and_treatments_to_query(outcomes = outcomes,
                                                 treatments = treatments & ancestors_and_Y_in_G),
        estimand=Sum(estimand, [Variable(v) for v in not_ancestors_of_Y]),
        graph=G_ancestral_to_Y,
    )


def line_3(
    *, outcomes: Set[str], treatments: Set[str], estimand: Expression, G: NxMixedGraph
) -> Identification:
    r"""Line 3 of ID algorithm.

    Forces an action on any node where such an action would have no
    effect on :math:\mathbf Y`—assuming we already acted on
    :math:`\mathbf X`. Since actions remove incoming arrows, we can
    view line 3 as simplifying the causal graph we consider by
    removing certain arcs from the graph, without affecting the
    overall answer.

    :param outcomes:
    :param treatments:
    :param estimand: Current probabilistic expression
    :param G: ADMG
    :returns:  The new estimand
    :raises Fail: if the query is not identifiable.

    """
    vertices = G.nodes()
    G_bar_x = G.intervene(treatments)
    no_effect_nodes = (vertices - treatments) - ancestors_and_self(G_bar_x, outcomes)
    if len(no_effect_nodes) > 0:
        query = outcomes_and_treatments_to_query(outcomes = outcomes,
                                                 treatments = treatments | no_effect_nodes)
        return Identification(query=query, estimand=estimand, graph=G)


def line_4(
    *, outcomes: Set[str], treatments: Set[str], estimand: Expression, G: NxMixedGraph
) -> List[Identification]:
    r"""Line 4 of the ID algorithm

    The key line of the algorithm, it decomposes the problem into a set
    of smaller problems using the key property of *c-component
    factorization* of causal models. If the entire graph is a single
    C-component already, further problem decomposition is impossible,
    and we must provide base cases. :math:`\mathbf{ID}` has three base
    cases.

    """
    V = G.nodes()
    C_components_of_G = get_c_components(G)
    C_components_of_G_without_X = get_c_components(G.remove_nodes_from(treatments))
    S = C_components_of_G_without_X[0]
    parents = list(nx.topological_sort(G.directed))

    if len(C_components_of_G_without_X) > 1:

        return [
            Identification(
                query = outcomes_and_treatments_to_query(outcomes=district, treatments=V-district),
                estimand=estimand,
                graph=G,
            )
            for district in C_components_of_G_without_X
        ]
        # return Sum(Product(*[ID(outcomes=district,
        #                         treatments=V - district,
        #                         estimand=estimand,
        #                         G=G)
        #                      for district in C_components_of_G_without_X]),
        #            list(V - (treatments | outcomes )))
    elif C_components_of_G_without_X == set([S]):
        if C_components_of_G == [frozenset(V)]:
            line_5(outcomes=outcomes, treatments=treatments, estimand=estimand, G=G)
        elif S in C_components_of_G:
            return Sum(
                Product(*[P(v | parents[: parents.index(v)]) for v in S]),
                list(V - (outcomes | treatments)),
            )
        else:
            for district in C_components_of_G:
                if S in district:
                    return Identification(
                        query=outcomes_and_treatments_to_query(
                            outcomes, treatments & district
                        ),
                        estimand=Product(
                            *[
                                P(
                                    v
                                    | list(
                                        (set(parents[: parents.index(v)]) & district)
                                        | (set(parents[: parents.index(v)]) - district)
                                    )
                                )
                                for v in district
                            ]
                        ),
                        graph=G.subgraph(district),
                    )


def line_5(
    *, outcomes: Set[str], treatments: Set[str], estimand: Expression, G: NxMixedGraph
):
    r"""line 5 of the identification algorithm.

    Fails because it finds two C-components, the graph :math:`G`
    itself, and a subgraph :math:`S` that does not contain any
    :math:`\mathbf X` nodes. But that is exactly one of the properties
    of C-forests that make up a hedge. In fact, it turns out that it
    is always possible to recover a hedge from these two c-components.

    """

    V = G.nodes()
    C_components_of_G = get_c_components(G)
    C_components_of_G_without_X = get_c_components(G.remove_nodes_from(treatments))

    if C_components_of_G == [frozenset(V)]:
        raise Fail(C_components_of_G, C_components_of_G_without_X)


def line_6(
    *, outcomes: Set[str], treatments: Set[str], estimand: Expression, G: NxMixedGraph
):
    V = G.nodes()
    C_components_of_G = get_c_components(G)
    C_components_of_G_without_X = get_c_components(G.remove_nodes_from(treatments))
    S = C_components_of_G_without_X[0]
    parents = list(nx.topological_sort(G.directed))
    if S in C_components_of_G:
        return Sum(
            Product(*[P(v | parents[: parents.index(v)]) for v in S]),
            list(V - (outcomes | treatments)),
        )


def line_7(
    *, outcomes: Set[str], treatments: Set[str], estimand: Expression, G: NxMixedGraph
):
    r"""Line 7 of the ID algorithm.

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

    """
    V = G.nodes()
    C_components_of_G = get_c_components(G)
    C_components_of_G_without_X = get_c_components(G.remove_nodes_from(treatments))
    S = C_components_of_G_without_X[0]
    parents = list(nx.topological_sort(G.directed))
    for district in C_components_of_G:
        if S < district:
            return dict(
                outcomes=outcomes,
                treatments=treatments & district,
                estimand=Product(
                    *[P(v | parents[: parents.index(v)]) for v in district]
                ),
                G=G.subgraph(district),
            )
        #  list(
        #  (set(parents[:parents.index(v)]) & district) |
        #  (set(parents[:parents.index(v)]) - district)))


def get_c_components(G: NxMixedGraph) -> Set[str]:
    return [frozenset(district) for district in nx.connected_components(G.undirected)]


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
