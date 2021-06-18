# -*- coding: utf-8 -*-

"""Utilities for identifiaction algorithms"""

from dataclasses import dataclass
from typing import Any, Set, Tuple

import networkx as nx
import numpy as np

from y0.dsl import Expression, P, Product, Sum, Variable
from y0.graph import NxMixedGraph
from y0.identify import _get_outcomes, _get_treatments
from y0.mutate import canonicalize

__all__ = [
    "str_graph",
    "nxmixedgraph_to_causal_graph",
    "ancestors_and_self",
    "Identification",
    "expr_equal",
    "Fail",
    "get_outcomes_and_treatments",
    "outcomes_and_treatments_to_query",
]


# TODO copy code for causal_graph class


class Fail(Exception):
    """Raised on failure of the identification algorithm."""


@dataclass
class Identification:
    """A package of a query and resulting estimand from identification on a graph."""

    query: Expression
    estimand: Expression
    graph: NxMixedGraph

    def __eq__(self, other: Any) -> bool:
        """Check fi the query, estimand, and graph are equal."""
        return (
            isinstance(other, Identification)
            and expr_equal(self.query, other.query)
            and expr_equal(self.estimand, other.estimand)
            and (self.graph == other.graph)
        )


def expr_equal(expected: Expression, actual: Expression) -> bool:
    """Return if two expressions are equal after canonicalization."""
    expected_outcomes, expected_treatments = get_outcomes_and_treatments(query=expected)
    actual_outcomes, actual_treatments = get_outcomes_and_treatments(query=actual)

    if (expected_outcomes != actual_outcomes) or (
        expected_treatments != actual_treatments
    ):
        return False
    ordering = tuple(expected.get_variables())  # need to impose ordering, any will do.
    expected_canonical = canonicalize(expected, ordering)
    actual_canonical = canonicalize(actual, ordering)
    return expected_canonical == actual_canonical


def get_outcomes_and_treatments(*, query: Expression) -> Tuple[Set[Variable], Set[Variable]]:
    """Get outcomes and treatments sets from the query expression."""
    return (
        set(_get_outcomes(query.get_variables())),
        set(_get_treatments(query.get_variables()))
    )


def outcomes_and_treatments_to_query(
    *, outcomes: Set[Variable], treatments: Set[Variable]
) -> Expression:
    if len(treatments) == 0:
        return P(*[Variable(y) if type(y) is str  else y  for y in outcomes])
    else:
        return P(
            *[y @ tuple(x
                                  if type(x) is Variable
                                  else Variable(x)
                                  for x in treatments)
              if type(y) is Variable
              else Variable(y) @ tuple(x
                             if type(x) is Variable
                             else Variable(x)
                             for x in treatments)
              for y in outcomes
            ]
        )


def ancestors_and_self(graph: NxMixedGraph, sources: Set[str]):
    """Ancestors of a set include the set itself."""
    rv = sources.copy()
    for source in sources:
        rv.update(nx.algorithms.dag.ancestors(graph.directed, source))
    return rv


def nxmixedgraph_to_bel2scm_causal_graph(graph: NxMixedGraph):
    """Converts NxMixedGraph to bel2scm.causal_graph"""
    from bel2scm import causal_graph

    di_edges = list(graph.directed.edges())
    bi_edges = list(graph.undirected.edges())
    vertices = list(graph.directed)  # could be either since they're maintained together
    str_list = [f"{U} => {V}" for U, V in di_edges]
    type_dict = dict([(U, "continuous") for U in vertices])
    cg = causal_graph.str_graph(str_list, "SCM", type_dict)
    cg.add_confound([[U, V] for U, V in bi_edges])
    return cg


def nxmixedgraph_to_causal_graph(graph: NxMixedGraph):
    """Converts NxMixedGraph to y0.algorithm.identify.utils.causal_graph"""
    di_edges = list(graph.directed.edges())
    bi_edges = list(graph.undirected.edges())
    vertices = list(graph.directed)  # could be either since they're maintained together
    str_list = [f"{U} => {V}" for U, V in di_edges]
    type_dict = dict([(U, "continuous") for U in vertices])
    cg = str_graph(str_list, "SCM", type_dict)
    cg.add_confound([[U, V] for U, V in bi_edges])
    return cg


class cg_graph:
    """define a superclass for causal graphs"""

    def proc_data(self, graph_type, type_dict=None):
        """take the list of edges and entities (i.e., nodes) and process that information to produce
        parent -> children and child -> parent mappings
        initialize all of the nodes of the causal graph"""
        if type_dict is None:
            type_dict = {}

        self.graph_type = graph_type
        n_nodes = len(self.entity_list)
        self.n_nodes = n_nodes

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.entity_list)
        self.graph.add_edges_from([[item[0], item[1]] for item in self.edge_list])

        # adj_mat = np.zeros((self.n_nodes,self.n_nodes),dtype=int)

        # for item in self.edge_list:
        # out_ind = self.entity_list.index(item[0])
        # in_ind = self.entity_list.index(item[1])
        # adj_mat[out_ind,in_ind] = 1

        # self.adj_mat = adj_mat

        # graph_temp = nx.DiGraph(adj_mat)
        # dict_temp = {}

        # for i in range(0,n_nodes):
        # dict_temp[i] = self.entity_list[i]

        # self.graph = nx.relabel_nodes(graph_temp, dict_temp)

        # check to make sure that it's a DAG
        if nx.algorithms.dag.is_directed_acyclic_graph(self.graph):
            print("The causal graph is a acyclic")

        else:
            print("The causal graph has cycles -- this is a problem")

            # identify edges that, if removed, would lead to the causal graph being acyclic
            c_bas = list(nx.simple_cycles(self.graph))
            print("There are " + str(len(c_bas)) + " simple cycles")

            cycle_edge_list = []

            for item in c_bas:
                for i in range(0, len(item)):
                    sub_temp = self.entity_list[item[i - 1]]
                    obj_temp = self.entity_list[item[i]]
                    rel_temp = [
                        item2[2]
                        for item2 in edge_list
                        if (item2[0] == sub_temp and item2[1] == obj_temp)
                    ]
                    cycle_edge_list += [
                        [sub_temp, obj_temp, item2] for item2 in rel_temp
                    ]
            print("Cycle edges:")
            for item in cycle_edge_list:
                print(item)

        self.cond_list = []

        self.sample_dict = {}

        # self.parent_ind_list = []
        # self.child_ind_list = []
        self.parent_dict = {}
        self.child_dict = {}

        # self.parent_ind_list = [np.where(self.adj_mat[:,i] > 0)[0] for i in range(0,self.n_nodes)]
        # self.child_ind_list = [np.where(self.adj_mat[i,:] > 0)[0] for i in range(0,self.n_nodes)]
        node_dict = {}

        for item in self.entity_list:
            self.parent_dict[item] = list(self.graph.predecessors(item))
            self.child_dict[item] = list(self.graph.successors(item))

            n_pars = len(self.parent_dict[item])

            if type_dict:
                node_type = type_dict[item]

            else:

                bel_dict = {}
                bel_dict["activity"] = ["activity", "act", "molecularActivity", "ma"]
                bel_dict["abundance"] = [
                    "a",
                    "abundance",
                    "complex",
                    "complexAbundance",
                    "geneAbundance",
                    "g",
                    "microRNAAbundance",
                    "m",
                    "populationAbundance",
                    "pop",
                    "proteinAbundance",
                    "p",
                    "rnaAbundance",
                    "r",
                    "compositeAbundance",
                    "composite",
                ]
                bel_dict["reaction"] = ["reaction", "rxn"]
                bel_dict["process"] = ["biologicalProcess", "bp"]
                bel_dict["pathology"] = ["pathology", "path"]

                vartype_dict = {}
                vartype_dict["activity"] = "Bernoulli"
                vartype_dict["abundance"] = "Gamma"
                vartype_dict["reaction"] = "Normal"
                vartype_dict["process"] = "Bernoulli"
                vartype_dict["pathology"] = "Bernoulli"

                ind_temp = item.find("(")
                str_temp = item[:ind_temp]
                node_type = ""

                for item in bel_dict:
                    if str_temp in bel_dict[item]:
                        node_type = vartype_dict[item]

                if node_type == "":
                    node_type = "Normal"
                    print(
                        "BEL node type "
                        + str_temp
                        + " not known -- defaulting to Normal"
                    )

            if self.graph_type == "Bayes":
                node_dict[item] = bayes_node(n_pars, item, node_type)
            elif self.graph_type == "MLE":
                node_dict[item] = mle_node(n_pars, item, node_type)
            elif self.graph_type == "SCM":
                node_dict[item] = scm_node(n_pars, item, node_type)
            else:
                print(
                    "node type "
                    + self.graph_type
                    + "not recognized -- defaulting to MLE"
                )
                node_dict[item] = mle_node(n_pars, item, node_type)

        self.node_dict = node_dict

        return

    def remove_edge(self, edge_rem):
        """remove all of the edges in edge_rem from the causal graph"""

        for item in edge_rem:
            self.graph.remove_edge(item)

            ind_remove = [
                i
                for i in range(0, len(self.edge_list))
                if (
                    self.edge_list[i][0] == edge_rem[0]
                    and self.edge_list[i][1] == edge_rem[1]
                )
            ]
            for ind in ind_remove:
                self.edge_list.remove(self.edge_list[i])

        for item in self.entity_list:
            self.parent_dict[item] = list(graph_temp.predecessors(item))
            self.child_dict[item] = list(graph_temp.successors(item))
        return

    def add_confound(self, confound_pairs):
        """add a list of pairs of nodes that share unobserved confounders"""

        graph_c = nx.Graph()
        graph_c.add_nodes_from(self.graph.nodes)
        graph_c.add_edges_from([tuple(item) for item in confound_pairs])

        self.graph_c = graph_c

        return

    def str_list(self, node_list):
        """return a string listing the nodes in node_list - this is used in the ID and IDC algorithms"""
        str_out = ""

        for item in node_list:
            str_out += item + ","

        return str_out[:-1]

    def d_sep(self, x, y, z, graph_in=None, conf_in=None):
        # determine if all paths from x to y are d-separated by z in graph_temp

        # convert digraph to undirected graph for d-separation

        if graph_in is None:
            graph_temp = self.graph.to_undirected()
        else:
            graph_temp = graph_in.to_undirected()

        if conf_in is None:
            graph_temp.add_edges_from(self.graph_c.edges)
        else:
            graph_temp.add_edges_from(conf_in.edges)

        # ensure that x, y, and z are disjoint
        if np.any([[item1 == item2 for item1 in x] for item2 in y]):
            print("x and y not disjoint")
            return

        if np.any([[item1 == item2 for item1 in x] for item2 in z]):
            print("x and z not disjoint")
            return

        if np.any([[item1 == item2 for item1 in z] for item2 in y]):
            print("y and z not disjoint")
            return

        # identify all paths from x to y
        path_list = []

        for item in x:
            for path in nx.all_simple_paths(graph_temp, source=item, target=y):
                path_list.append(path)

        # print(str(len(path_list)) + ' total paths')

        # iterate through paths
        for item in path_list:
            # if an element of z is in the path, path is d-separated
            # else, path is not d-separated, return False

            if not np.any([ind in item for ind in z]):
                return False

        # if all paths d-separated, return True

        return True

    def id_alg(self, y, x, p_in=None, graph_in=None):
        """calculate P(y | do(x)) or return failure if this is not possible"""

        if graph_in is None:
            graph_temp = nx.DiGraph(self.graph)
        else:
            graph_temp = graph_in

        if p_in is None:
            p_expr = P(*[Variable(v) for v in graph_temp.nodes])
        else:
            p_expr = p_in

        if np.any([item in y for item in x]):
            print("Error -- overlap between x and y")
            print(x)
            print(y)
            print(p_in)
            print(graph_in.nodes)

        y_anc = y.copy()

        # identify ancestors of y
        for item in y:
            set_temp = nx.algorithms.dag.ancestors(graph_temp, item)
            y_anc += [item2 for item2 in set_temp if item2 not in y_anc]

        # identify all nodes in the graph
        v_not_anc_y = [item for item in graph_temp.nodes if item not in y_anc]

        # remove edges to x
        graph_xbar = nx.DiGraph(graph_temp)
        for item in x:
            graph_xbar.remove_edges_from(list(graph_temp.in_edges(item)))

        y_anc_x_bar = y.copy()

        for item in y:
            set_temp = nx.algorithms.dag.ancestors(graph_xbar, item)
            y_anc_x_bar += [item2 for item2 in set_temp if item2 not in y_anc_x_bar]

        w_set = [
            item
            for item in graph_temp.nodes
            if item not in x and item not in y_anc_x_bar
        ]

        # line 1
        if not x:
            # return sum over all non-y variables

            node_list = [item for item in graph_temp.nodes if item not in y]

            # print('Step 1')

            return Sum(
                P(*[Variable(node) for node in graph_temp.nodes]),
                [Variable(node) for node in node_list],
            )

        # line 2
        elif v_not_anc_y:

            x_temp = [item for item in y_anc if item in x]
            str_out = Sum(
                P(Variable(v) for v in graph_temp.nodes),
                [Variable(v) for v in v_not_anc_y],
            )
            graph_anc = graph_temp.subgraph(y_anc)

            # print('Begin Step 2')
            # print(v_not_anc_y)
            expr_out = self.id_alg(y, x_temp, str_out, graph_anc)
            # print('End Step 2')

            return expr_out

        # line 3
        elif w_set:  # TODO @jeremy needs test case
            # print('Begin Step 3')
            # print(w_set)
            expr_out = self.id_alg(y, x + w_set, p_expr, graph_temp)
            # print('End Step 3')

            return expr_out

        else:
            # calculate graph C-components
            graph_temp_c = nx.Graph(self.graph_c.subgraph(graph_temp.nodes))
            graph_temp_c.remove_nodes_from(x)
            s_sets = [list(item) for item in nx.connected_components(graph_temp_c)]

            # line 4
            if len(s_sets) > 1:
                # print('Begin Step 4')
                # print(s_sets)
                node_list = [
                    item for item in graph_temp.nodes if item not in y and item not in x
                ]
                str_out = []

                for item in s_sets:
                    v_s_set = [item2 for item2 in graph_temp.nodes if item2 not in item]
                    s_in = [item2 for item2 in item]

                    if np.any([item2 in v_s_set for item2 in s_in]):
                        print("Error -- x/y overlap")
                        print(v_s_set)
                        print(s_in)

                    str_out += [self.id_alg(s_in, v_s_set, p_expr, graph_temp)]

                # print('End Step 4')

                return Sum(Product(str_out), [Variable(v) for v in node_list])

            else:
                graph_temp_c_prime = self.graph_c.subgraph(graph_temp.nodes)

                s_sets_prime = [
                    list(item) for item in nx.connected_components(graph_temp_c_prime)
                ]

                # line 5
                if sorted(s_sets_prime[0]) == sorted(
                    graph_temp.nodes
                ):  # TODO @jeremy needs test case

                    node_list = [ind for ind in s_sets_prime[0]]
                    node_list2 = [ind for ind in graph_temp.nodes if ind in s_sets[0]]

                    raise Fail(
                        f"Identification Failure: C-components of U {node_list} and C-components of (U-x) {node_list2} form a hedge"
                    )

                # line 6
                elif np.any(
                    [sorted(s_sets[0]) == sorted(item) for item in s_sets_prime]
                ):

                    node_list = [item for item in s_sets[0] if item not in y]
                    str_out = []

                    for item in s_sets[0]:
                        # identify parents of node i
                        parents = list(graph_temp.predecessors(item))

                        if parents:
                            str_out += [
                                P(Variable(item) | [Variable(p) for p in parents])
                            ]
                        else:
                            str_out += [P(Variable(item))]
                    # print(s_sets[0])
                    # print('Step 6')
                    return Sum(Product(str_out), [Variable(v) for v in node_list])

                # line 7
                elif np.any(
                    [
                        np.all([item in item2 for item in s_sets[0]])
                        for item2 in s_sets_prime
                    ]
                ):
                    ind = np.where(
                        [
                            np.all([item in item2 for item in s_sets[0]])
                            for item2 in s_sets_prime
                        ]
                    )[0][0]

                    graph_prime = graph_temp.subgraph(s_sets_prime[ind])
                    x_prime = [item for item in s_sets_prime[ind] if item in x]
                    str_out = []

                    for item in s_sets_prime[ind]:

                        pred = list(nx.algorithms.dag.ancestors(graph_temp, item))
                        par_set = [
                            item2 for item2 in pred if item2 in s_sets_prime[ind]
                        ]
                        par_set += [
                            item2 for item2 in pred if item2 not in s_sets_prime[ind]
                        ]

                        if par_set:
                            str_out += [
                                P(Variable(item) | [Variable(p) for p in par_set])
                            ]
                        else:
                            str_out += [P(Variable(item))]

                    # print('Begin Step 7')
                    # print((s_sets[0],s_sets_prime[ind]))

                    if np.any([item2 in x_prime for item2 in y]):
                        print("Error -- x/y overlap")
                        print(x_prime)
                        print(y)

                    expr_out = self.id_alg(y, x_prime, Product(str_out), graph_prime)
                    # print('End Step 7')

                    return expr_out

                else:

                    print("error")
                    return ""

    def craig_id_alg(self, y, x, p_in=None, graph_in=None):
        """calculate P(y | do(x)) or return failure if this is not possible"""

        if graph_in is None:
            graph_temp = nx.DiGraph(self.graph)
        else:
            graph_temp = graph_in

        if p_in is None:
            p_expr = "P(" + self.str_list(graph_temp.nodes) + ")"
        else:
            p_expr = p_in

        if np.any([item in y for item in x]):
            print("Error -- overlap between x and y")
            print(x)
            print(y)
            print(p_in)
            print(graph_in.nodes)

        y_anc = y.copy()

        # identify ancestors of y
        for item in y:
            set_temp = nx.algorithms.dag.ancestors(graph_temp, item)
            y_anc += [item2 for item2 in set_temp if item2 not in y_anc]

        # identify all nodes in the graph
        v_not_anc_y = [item for item in graph_temp.nodes if item not in y_anc]

        # remove edges to x
        graph_xbar = nx.DiGraph(graph_temp)
        for item in x:
            graph_xbar.remove_edges_from(list(graph_temp.in_edges(item)))

        y_anc_x_bar = y.copy()

        for item in y:
            set_temp = nx.algorithms.dag.ancestors(graph_xbar, item)
            y_anc_x_bar += [item2 for item2 in set_temp if item2 not in y_anc_x_bar]

        w_set = [
            item
            for item in graph_temp.nodes
            if item not in x and item not in y_anc_x_bar
        ]

        # line 1
        if not x:
            # return sum over all non-y variables

            node_list = [item for item in graph_temp.nodes if item not in y]
            str_out = "[sum_{" + self.str_list(node_list) + "} " + p_expr + "]"
            # print('Step 1')

            return str_out

        # line 2
        elif v_not_anc_y:

            x_temp = [item for item in y_anc if item in x]
            str_out = "[sum_{" + self.str_list(v_not_anc_y) + "} " + p_expr + "]"
            graph_anc = graph_temp.subgraph(y_anc)

            # print('Begin Step 2')
            # print(v_not_anc_y)
            expr_out = self.id_alg(y, x_temp, str_out, graph_anc)
            # print('End Step 2')

            return expr_out

        # line 3
        elif w_set:  # TODO @jeremy needs test case
            # print('Begin Step 3')
            # print(w_set)
            expr_out = self.id_alg(y, x + w_set, p_expr, graph_temp)
            # print('End Step 3')

            return expr_out

        else:
            # calculate graph C-components
            graph_temp_c = nx.Graph(self.graph_c.subgraph(graph_temp.nodes))
            graph_temp_c.remove_nodes_from(x)
            s_sets = [list(item) for item in nx.connected_components(graph_temp_c)]

            # line 4
            if len(s_sets) > 1:
                # print('Begin Step 4')
                # print(s_sets)
                node_list = [
                    item for item in graph_temp.nodes if item not in y and item not in x
                ]
                str_out = "[sum_{" + self.str_list(node_list) + "} "

                for item in s_sets:
                    v_s_set = [item2 for item2 in graph_temp.nodes if item2 not in item]
                    s_in = [item2 for item2 in item]

                    if np.any([item2 in v_s_set for item2 in s_in]):
                        print("Error -- x/y overlap")
                        print(v_s_set)
                        print(s_in)

                    str_out += self.id_alg(s_in, v_s_set, p_expr, graph_temp)

                # print('End Step 4')
                str_out += "]"

                return str_out

            else:
                graph_temp_c_prime = self.graph_c.subgraph(graph_temp.nodes)

                s_sets_prime = [
                    list(item) for item in nx.connected_components(graph_temp_c_prime)
                ]

                # line 5
                if sorted(s_sets_prime[0]) == sorted(
                    graph_temp.nodes
                ):  # TODO @jeremy needs test case

                    node_list = [ind for ind in s_sets_prime[0]]
                    node_list2 = [ind for ind in graph_temp.nodes if ind in s_sets[0]]

                    str_out = (
                        "FAIL("
                        + self.str_list(node_list)
                        + ","
                        + self.str_list(node_list2)
                        + ")"
                    )

                    # print('Step 5')
                    return str_out

                # line 6
                elif np.any(
                    [sorted(s_sets[0]) == sorted(item) for item in s_sets_prime]
                ):

                    node_list = [item for item in s_sets[0] if item not in y]
                    str_out = "[sum_{" + self.str_list(node_list) + "}"

                    for item in s_sets[0]:
                        # identify parents of node i
                        parents = list(graph_temp.predecessors(item))

                        if parents:
                            str_out += "P(" + item + "|" + self.str_list(parents) + ")"
                        else:
                            str_out += "P(" + item + ")"
                    # print(s_sets[0])
                    # print('Step 6')
                    return str_out + "]"

                # line 7
                elif np.any(
                    [
                        np.all([item in item2 for item in s_sets[0]])
                        for item2 in s_sets_prime
                    ]
                ):
                    ind = np.where(
                        [
                            np.all([item in item2 for item in s_sets[0]])
                            for item2 in s_sets_prime
                        ]
                    )[0][0]

                    graph_prime = graph_temp.subgraph(s_sets_prime[ind])
                    x_prime = [item for item in s_sets_prime[ind] if item in x]
                    str_out = ""

                    for item in s_sets_prime[ind]:

                        pred = list(nx.algorithms.dag.ancestors(graph_temp, item))
                        par_set = [
                            item2 for item2 in pred if item2 in s_sets_prime[ind]
                        ]
                        par_set += [
                            item2 for item2 in pred if item2 not in s_sets_prime[ind]
                        ]

                        if par_set:
                            str_out += "P(" + item + "|" + self.str_list(par_set) + ")"
                        else:
                            str_out += "P(" + item + ")"

                    # print('Begin Step 7')
                    # print((s_sets[0],s_sets_prime[ind]))

                    if np.any([item2 in x_prime for item2 in y]):
                        print("Error -- x/y overlap")
                        print(x_prime)
                        print(y)

                    expr_out = self.id_alg(y, x_prime, str_out, graph_prime)
                    # print('End Step 7')

                    return expr_out

                else:

                    print("error")
                    return ""

    def idc_alg(self, y, x, z, p_in=None, graph_in=None):
        """calculate P(y | do(x), z) or return failure if this is not possible"""

        if graph_in is None:
            graph_temp = nx.DiGraph(self.graph)
        else:
            graph_temp = graph_in

        if p_in is None:
            p_expr = "P(" + self.str_list(graph_temp.nodes) + ")"
        else:
            p_expr = p_in

        if np.any([item in y for item in x]):
            print("Error -- overlap between x and y")
            print(x)
            print(y)
            print(p_in)
            print(graph_in.nodes)

        if np.any([item in y for item in z]):
            print("Error -- overlap between z and y")
            print(z)
            print(y)
            print(p_in)
            print(graph_in.nodes)

        if np.any([item in z for item in x]):
            print("Error -- overlap between x and z")
            print(x)
            print(z)
            print(p_in)
            print(graph_in.nodes)

        digraph_xbar = nx.DiGraph(graph_temp)
        for item in x:
            digraph_xbar.remove_edges_from(graph_temp.in_edges(item))

        # identify edges from z
        z_inds = [ind for ind in graph_temp.nodes if ind in z]
        z_edges = [list(graph_temp.out_edges(item2)) for item2 in z_inds]

        # check for d-separation
        for item in z:
            digraph_xbar_zbar = nx.DiGraph(digraph_xbar)
            digraph_xbar_zbar.remove_edges_from(graph_temp.out_edges(item))
            digraph_xbar_zbar = digraph_xbar_zbar.to_undirected()

            digraph_xbar_zbar.add_edges_from(
                self.graph_c.subgraph(graph_temp.nodes).edges
            )

            # calculate d-separation
            d_sep = self.d_sep(
                y,
                [item],
                [item2 for item2 in x + z if item2 != item],
                digraph_xbar_zbar,
                self.graph_c.subgraph(graph_temp.nodes),
            )

            if d_sep:
                return self.idc_alg(
                    y,
                    x + [item],
                    [item2 for item2 in z if item2 != item],
                    p_expr,
                    graph_temp,
                )

        p_prime = self.id_alg(y + z, x, p_expr, graph_temp)

        str_out = "[" + p_prime + "]/[ sum_{" + self.str_list(y) + "}" + p_prime + "]"

        return str_out

    def make_pw_graph(self, do_in, graph_in=None):
        """create the parallel-world graph of subgraph of graph_in or self.graph"""

        # graph_out only has 'real' nodes -- conf_out has confounding nodes
        if graph_in is None:
            graph_temp = nx.DiGraph(self.graph)
            conf_temp = nx.Graph(self.graph_c)
        else:
            graph_temp = nx.DiGraph(graph_in)
            conf_temp = self.graph_c.subgraph(graph_temp.nodes)

        # record all nodes with unobserved confounders in the original graph
        vars_with_conf = []
        for item in conf_temp.edges:
            if item[0] not in vars_with_conf:
                vars_with_conf.append(item[0])
            if item[1] not in vars_with_conf:
                vars_with_conf.append(item[1])

        # confounding nodes corresponding to duplicate pw-graph nodes
        conf_nodes = [
            "U^{" + item + "}"
            for item in graph_temp.nodes
            if item not in vars_with_conf
        ]

        # confounding nodes corresponding to confounders in the original graph
        conf_nodes += [
            "U^{" + item[0] + "," + item[1] + "}" for item in conf_temp.edges
        ]

        graph_out = nx.DiGraph(graph_temp)
        graph_out.add_nodes_from(conf_nodes)
        # add confounders - now a digraph because we've added nodes for each confounder
        conf_out = nx.DiGraph()
        conf_out.add_nodes_from(graph_out.nodes)

        # add confounding edges
        conf_edges_add = [
            ("U^{" + item + "}", item)
            for item in graph_temp.nodes
            if item not in vars_with_conf
        ]
        conf_edges_add += [
            ("U^{" + item[0] + "," + item[1] + "}", item[0]) for item in conf_temp.edges
        ]
        conf_edges_add += [
            ("U^{" + item[0] + "," + item[1] + "}", item[1]) for item in conf_temp.edges
        ]
        conf_out.add_edges_from(conf_edges_add)

        # add duplicate edges and nodes
        for item in do_in:
            str_temp = self.str_list(item[1])

            # create nodes and edges corresponding to duplicate graph
            # don't add edges going into do-variable nodes
            node_list = [
                item2 + "_{" + str_temp + "}"
                for item2 in graph_temp.nodes
                if item2 not in [item3.replace("*", "") for item3 in item[1]]
            ]
            node_list += item[1]

            edge_list = [
                (item2[0] + "_{" + str_temp + "}", item2[1] + "_{" + str_temp + "}")
                for item2 in graph_temp.edges
                if item2[1] not in [item3.replace("*", "") for item3 in item[1]]
                and item2[0] not in [item3.replace("*", "") for item3 in item[1]]
            ]

            for item2 in item[1]:
                edge_list += [
                    (item2, item3[1] + "_{" + str_temp + "}")
                    for item3 in graph_temp.edges
                    if item3[0] == item2.replace("*", "")
                ]

            # add duplicate nodes and edges to the underlying digraph
            graph_out.add_nodes_from(node_list)
            graph_out.add_edges_from(edge_list)

            # create confounder edges for duplicate variables
            conf_edge_list = [
                ("U^{" + item2 + "}", item2 + "_{" + str_temp + "}")
                for item2 in graph_temp.nodes
                if item2 not in vars_with_conf
                and item2 not in [item3.replace("*", "") for item3 in item[1]]
            ]

            # create confounder edges for confounders from the original graph
            conf_edge_list += [
                (
                    "U^{" + item2[0] + "," + item2[1] + "}",
                    item2[0] + "_{" + str_temp + "}",
                )
                for item2 in conf_temp.edges
                if item2[0] not in [item3.replace("*", "") for item3 in item[1]]
            ]
            conf_edge_list += [
                (
                    "U^{" + item2[0] + "," + item2[1] + "}",
                    item2[1] + "_{" + str_temp + "}",
                )
                for item2 in conf_temp.edges
                if item2[1] not in [item3.replace("*", "") for item3 in item[1]]
            ]

            # add duplicate nodes and confounder edges to confounding digraph
            conf_out.add_nodes_from(node_list)
            conf_out.add_edges_from(conf_edge_list)

        return graph_out, conf_out

    def make_cf_graph(self, do_in, obs_in=None, graph_in=None):
        """create the counterfactual graph of subgraph of graph_in or self.graph"""

        if graph_in is None:
            graph_temp = nx.DiGraph(self.graph)
        else:
            graph_temp = nx.DiGraph(graph_in)

        if obs_in is None:
            conf_temp = nx.Graph(self.graph_c)
        else:
            conf_temp = self.graph_c.subgraph(graph_temp.nodes)

        gamma_list = self.conv_to_gamma(do_in, obs_in)

        # create parallel worlds graph
        graph_out, conf_out = self.make_pw_graph(do_in, graph_in)

        # iterate through nodes and merge variables
        node_list = [
            item for item in graph_temp.nodes if graph_temp.in_degree(item) == 0
        ]
        traversed_nodes = []

        while sorted(traversed_nodes) != sorted(graph_temp.nodes) and node_list:

            # start with the first item of node_list
            node_temp = node_list[0]

            # identify parents of node_temp
            par_temp = [item[0] for item in graph_out.edges if item[1] == node_temp]

            # cycle through all of the duplicate graphs and merge nodes

            for item in do_in:
                str_temp = self.str_list(item[1])
                # identify the node to check
                node_temp2 = node_temp + "_{" + str_temp + "}"

                # see if all the parents are identical in graph_out
                graph_pars = sorted(par_temp) == sorted(
                    [item2[0] for item2 in graph_out.edges if item2[1] == node_temp2]
                )

                # see if all the parents are identical in conf_out
                conf_pars = sorted(
                    [item2[0] for item2 in conf_out.edges if item2[1] == node_temp]
                ) == sorted(
                    [item2[0] for item2 in conf_out.edges if item2[1] == node_temp2]
                )

                # identify all of the parents that are not the same
                par_diff = [
                    item2[0]
                    for item2 in graph_out.edges
                    if item2[1] == node_temp2 and item2[0] not in par_temp
                ]

                # if the parents all match up, merge the nodes
                # elif the node being checked has all of the nodes in par_diff as do-variables and
                # they're also observed, do the merge
                # identify cases where the parents don't match exactly but the values line up
                # B_{} -> A_{} => B -> A if B_{} and A_{} are both observed

                if graph_pars and conf_pars:
                    # print('Merge ' + node_temp + ' and ' + node_temp2 + ': Parents Match')
                    graph_out = nx.contracted_nodes(
                        graph_out, node_temp, node_temp2, self_loops=False
                    )
                    conf_out = nx.contracted_nodes(
                        conf_out, node_temp, node_temp2, self_loops=False
                    )

                    if node_temp2 in gamma_list:

                        # check for inconsistency
                        if node_temp in gamma_list and node_temp2 in gamma_list:
                            gamma_list = ["INCONSISTENT"]
                            return graph_out, conf_out, gamma_list
                        else:
                            gamma_list = [
                                item2 if item2 != node_temp2 else node_temp
                                for item2 in gamma_list
                            ]

                elif (
                    np.all([item2 in item[1] and item2 in obs_in for item2 in par_diff])
                    and conf_pars
                ):
                    # print('Merge ' + node_temp + ' and ' + node_temp2 + ': Parents Match with Do-Variables')
                    # remove edges from the duplicate parents
                    graph_out.remove_edges_from(
                        [
                            item2
                            for item2 in graph_out.edges
                            if item2[0] in par_diff and item2[1] == node_temp2
                        ]
                    )

                    # merge nodes
                    graph_out = nx.contracted_nodes(
                        graph_out, node_temp, node_temp2, self_loops=False
                    )
                    conf_out = nx.contracted_nodes(
                        conf_out, node_temp, node_temp2, self_loops=False
                    )

                    # check for inconsistency
                    if node_temp in gamma_list and node_temp2 in gamma_list:
                        gamma_list = ["INCONSISTENT"]
                        return graph_out, conf_out, gamma_list
                    else:
                        gamma_list = [
                            item2 if item2 != node_temp2 else node_temp
                            for item2 in gamma_list
                        ]

            # only add nodes whose parents have all been traversed
            node_list = node_list[1:] + [
                item[1]
                for item in graph_temp.edges
                if np.all(
                    [
                        item2[0] in node_list
                        for item2 in graph_temp.edges
                        if item2[1] == item[1]
                    ]
                )
            ]

            traversed_nodes += [node_temp]

        # remove self-loops
        # graph_out.remove_edges_from(nx.selfloop_edges(graph_out))
        # conf_out.remove_edges_from(nx.selfloop_edges(conf_out))

        if np.any([item not in graph_out.nodes for item in gamma_list]):
            print("Missing Nodes")
            print(gamma_list)
            print(graph_out.nodes)
            print(graph_temp.nodes)

        # identify ancestors of nodes in gamma_list
        anc_list = []
        anc_list += gamma_list

        # identify ancestors in graph_out
        for item in gamma_list:
            anc_list += [
                item2
                for item2 in nx.algorithms.dag.ancestors(graph_out, item)
                if item2 not in anc_list
            ]

        # identify ancestors in conf_out
        conf_anc_list = []
        for item in anc_list:
            anc_list += [
                item2
                for item2 in nx.algorithms.dag.ancestors(conf_out, item)
                if item2 not in anc_list
            ]

        graph_out = nx.DiGraph(graph_out.subgraph(anc_list))
        conf_out = nx.DiGraph(conf_out.subgraph(anc_list))

        # removing (apparently) unneccesary nodes/edges may cause problems in ID* because of recursion
        # check this!

        # remove confounding nodes that only connect to one node
        rem_nodes = [
            item
            for item in conf_out.nodes
            if conf_out.degree(item) == 1 and item[0] == "U"
        ]
        graph_out.remove_nodes_from(rem_nodes)
        conf_out.remove_nodes_from(rem_nodes)

        # remove disconnected nodes

        node_list = list(graph_out.nodes)
        for item in node_list:
            list_temp1 = list(nx.isolates(graph_out))
            list_temp2 = list(nx.isolates(conf_out))

            if item in list_temp1 and item in list_temp2:
                if item in gamma_list:
                    if "*" in item and item.replace("*", "") in gamma_list:
                        print("duplicate observables")
                        print(item)
                        print(gamma_list)
                    elif item.replace("*", "") in list_temp1 + list_temp2:
                        gamma_list = [
                            item2 if item2 != item else item.replace("*", "")
                            for item2 in gamma_list
                        ]
                    else:
                        gamma_list = [item2 for item2 in gamma_list if item2 != item]
                        graph_out.remove_node(item)
                        conf_out.remove_node(item)

        # restrict interventions to ancestors of the variables they act on
        node_list = list(graph_out.nodes)
        do_temp, obs_temp = self.conv_from_gamma(node_list)
        node_list = self.conv_to_gamma(do_temp, obs_temp)

        do_temp2 = []
        for i in range(0, len(do_temp)):
            do_temp2.append(
                [
                    do_temp[i][0],
                    [
                        item
                        for item in do_temp[i][1]
                        if item
                        in list(nx.algorithms.dag.ancestors(graph_out, node_list[i]))
                    ],
                ]
            )

        node_list2 = self.conv_to_gamma(do_temp2, obs_temp)

        remap_key = {}
        gamma_list2 = []
        for i in range(0, len(node_list)):
            remap_key[node_list[i]] = node_list2[i]
            if node_list[i] in gamma_list:
                gamma_list2.append(node_list2[i])

        graph_out2 = nx.relabel_nodes(graph_out, remap_key)
        conf_out2 = nx.relabel_nodes(conf_out, remap_key)

        return graph_out2, conf_out2, gamma_list2

    def conv_to_gamma(self, do_in, obs_in):
        """convert from do_in, obs_in to gamma_list"""

        gamma_list = []
        for item in do_in:
            gamma_list.append(item[0] + "_{" + self.str_list(item[1]) + "}")

        for item in obs_in:
            if item not in gamma_list:
                gamma_list.append(item)

        return gamma_list

    def conv_from_gamma(self, gamma_list):
        """convert from gamma_list to do_in, obs_in"""

        do_in = []
        obs_in = []

        # this can handle nested do statements: e.g., y_{x_{z}}, but x_z now becomes an intervention
        for item in gamma_list:
            if "_" in item:
                temp = item.replace("_{", ",", 1).split(",")
                if temp[-1][-1] == "}":
                    temp[-1] = temp[-1].replace("}", "")
                do_in.append([temp[0], temp[1:]])
            else:
                obs_in.append(item)

        return do_in, obs_in

    def id_star_alg(self, do_in, obs_in=None, graph_in=None):
        """implement ID* algorithm
        Denote interventions with asterisks (e.g., 'X*') and observations without asterisks (e.g., 'X')"""

        if graph_in is None:
            graph_temp = nx.DiGraph(self.graph)
        else:
            graph_temp = nx.DiGraph(graph_in)

        if obs_in is None:
            conf_temp = nx.Graph(self.graph_c)
        else:
            conf_temp = self.graph_c.subgraph(graph_temp.nodes)

        gamma_list = self.conv_to_gamma(do_in, obs_in)
        # print(gamma_list)

        do_vars = []
        for item in do_in:
            do_vars += [item2 for item2 in item[1] if item2 not in do_vars]

        if not gamma_list:
            # print('Step 1')
            # print()
            return "1"

        elif np.any([item[0] + "*" in item[1] for item in do_in]):
            # print('Step 2')
            # print()
            return "0"

        elif np.any([item[0] in item[1] for item in do_in]):

            temp_inds = [
                ind
                for ind in range(0, len(do_in))
                if do_in[ind][0] not in do_in[ind][1]
            ]

            # print('Step 3')
            # print(do_in)
            # print(do_in[ind])
            # print()

            return self.id_star_alg(do_in[ind], obs_in, graph_in)

        else:
            graph_out, conf_out, gamma_list = self.make_cf_graph(
                do_in, obs_in, graph_in
            )

            do_temp, obs_temp = self.conv_from_gamma(gamma_list)

            do_vars_temp = []
            for item in do_temp:
                do_vars_temp += [
                    item2 for item2 in item[1] if item2 not in do_vars_temp
                ]

            # print(gamma_list)
            # print('Step 4')

            # calculate graph C-components
            s_sets = [
                list(item) for item in nx.connected_components(conf_out.to_undirected())
            ]

            # nodes in graph_out/conf_out fixed by interventions aren't included in any C-component
            # enforce this manually

            s_sets = [item for item in s_sets if item[0] not in do_vars_temp]

            # print('C sets')
            # print(do_vars_temp)
            # print(s_sets)

            if "INCONSISTENT" in gamma_list:
                # print('Step 5')
                return "0"

            elif len(s_sets) > 1:
                # print('Start Step 6')

                sum_list = []
                # get all variables contained in gamma_list
                d_temp, o_temp = self.conv_from_gamma(gamma_list)
                g_temp = []

                for item in d_temp:
                    g_temp += [item[0]]
                    g_temp += [item2 for item2 in item[1] if item2 not in g_temp]
                g_temp += [item for item in o_temp if item not in g_temp]

                # for item in d_temp:
                # g_temp += [item[0]]
                # g_temp += [item2.replace('*','') for item2 in item[1] if item2.replace('*','') not in g_temp]
                # g_temp += [item.replace('*','') for item in o_temp if item.replace('*','') not in g_temp]

                # get all node variables contained in graph_out
                node_list_temp = [item for item in graph_out.nodes if item[:3] != "U^{"]

                d_temp2, o_temp2 = self.conv_from_gamma(node_list_temp)
                v_temp = []

                for item in d_temp2:
                    v_temp += [item[0]]
                    v_temp += [item2 for item2 in item[1] if item2 not in v_temp]
                v_temp += [item for item in o_temp2 if item not in v_temp]

                # for item in d_temp2:
                # v_temp += [item[0]]
                # v_temp += [item2.replace('*','') for item2 in item[1] if item2.replace('*','') not in v_temp]
                # v_temp += [item.replace('*','') for item in o_temp2 if item.replace('*','') not in v_temp]

                sum_list = [item for item in v_temp if item not in g_temp]

                str_out = "sum_{" + self.str_list(sum_list) + "}"
                for item in s_sets:

                    check_list = []

                    # define the new do-statements
                    do_in_temp = []

                    for item2 in item:
                        # don't include confounding variables
                        if item2[:3] != "U^{":
                            # split variables that already have do-statements in them
                            if "_{" in item2:
                                item_temp = (
                                    item2.replace("_{", ",").replace("}", "").split(",")
                                )
                                do_list_temp = item_temp[1:]
                            else:
                                item_temp = [item2]
                                do_list_temp = []

                            for item3 in nx.algorithms.dag.ancestors(graph_out, item2):
                                if "U^{" not in item3 and item3 not in item:
                                    if (
                                        item3.find("_") > 0
                                        and item3[: item3.find("_")] not in do_list_temp
                                    ):
                                        do_list_temp.append(
                                            item3[: item3.find("_")] + "*"
                                        )
                                    elif item3 not in do_list_temp:
                                        do_list_temp.append(item3 + "*")

                            # add variables to do_list_temp
                            # for item3 in graph_out.nodes:
                            # if (item3 not in item and 'U^{' not in item3 and item3.replace('*','')
                            # not in [item4.replace('*','') for item4 in do_vars_temp]):

                            # if item3.find('_') > 0:
                            # do_list_temp.append(item3[:item3.find('_')] + '*')
                            # else:
                            # do_list_temp.append(item3 + '*')

                            # only consider interventions on ancestors of the target node
                            # do_list_temp = [item3 for item3 in do_list_temp
                            # if item3 in nx.algorithms.dag.ancestors(graph_out,item2)]
                            do_in_temp += [[item_temp[0], do_list_temp]]
                            check_list += [
                                item3
                                for item3 in do_list_temp
                                if item3 not in check_list
                            ]

                    # print(do_in_temp)

                    str_temp = self.id_star_alg(do_in_temp, [], graph_temp)
                    # make sure that all the variables being summed over don't have asterisks
                    # for item2 in sum_list:
                    # str_temp = str_temp.replace(item2 + '*',item2)

                    # make sure that the value of variables correspond to those in the current graph
                    for item2 in check_list:
                        if item2 not in sum_list and item2 not in g_temp:
                            str_temp = str_temp.replace(item2, item2.replace("*", ""))

                    # for item2 in do_list_temp:
                    # if item2 not in sum_list and item2.replace('*','') in g_temp:
                    # str_temp = str_temp.replace(item2, item2.replace('*',''))

                    str_out += str_temp
                    # print()

                # print('End Step 6')
                # print()

                return str_out

            else:
                # print('Step 7')

                if s_sets == []:
                    print("s_sets is empty")

                gamma_temp = [item for item in s_sets[0] if "U^{" not in item]

                # separate nodes into nodes with and without do-statements as part of them
                do_temp, obs_temp = self.conv_from_gamma(gamma_temp)

                # simplify do-statements to get rid of extraneous interventions
                do_temp2 = []
                for item in do_temp:
                    do_temp2.append(
                        [
                            item[0],
                            [
                                item2
                                for item2 in item[1]
                                if item2
                                in nx.algorithms.dag.ancestors(
                                    graph_out,
                                    item[0] + "_{" + self.str_list(item[1]) + "}",
                                )
                            ],
                        ]
                    )

                gamma_subs = []
                for item in do_temp2:
                    gamma_subs += [
                        item2 for item2 in item[1] if item2 not in gamma_subs
                    ]
                do_vars = [item[0] for item in do_temp2]

                # print(do_temp2)
                # print(obs_temp)
                # print(gamma_subs)

                # step 8 - basically make sure that you don't have y_x and observed x' != x in graph_out
                # for my code, look for x_{x...} and x in graph_out

                # check for differing interventions on the same variable - this step doesn't seem
                # to be defined very precisely in the write-up
                do_diff = np.any([item + "*" in gamma_subs for item in gamma_subs])
                do_obs_diff = np.any([item + "*" in gamma_subs for item in obs_temp])

                if do_diff:
                    # print('Step 8')
                    # print()
                    return " FAIL "
                else:

                    # print('Step 9')
                    # print('P(' + self.str_list(gamma_temp) + ')')
                    # print()
                    return "P(" + self.str_list(gamma_temp) + ")"
        return

    def idc_star_alg(self, do_in, do_delta, obs_in=None, obs_delta=None, graph_in=None):
        """Implement IDC* algorithm
        Denote interventions with asterisks (e.g., 'X*') and observations without asterisks (e.g., 'X')"""

        if graph_in is None:
            graph_temp = nx.DiGraph(self.graph)
            conf_temp = nx.Graph(self.graph_c)
        else:
            graph_temp = nx.DiGraph(graph_in)
            conf_temp = self.graph_c.subgraph(graph_temp.nodes)

        if obs_in is None:
            obs_in = []
        if obs_delta is None:
            obs_delta = []

        if self.id_star_alg(do_delta, obs_delta, graph_in) == "0":
            # print('IDC* Step 1')
            return "UNDEFINED"
        else:
            graph_out, conf_out, gamma_list = self.make_cf_graph(
                do_in + do_delta, obs_in + obs_delta, graph_in
            )
            # print(gamma_list)
            # print('IDC* Step 2')

            if "INCONSISTENT" in gamma_list:
                # print('IDC* Step 3')
                return "0"
            else:
                n_gam = len(do_in) + len(obs_in)
                n_del = len(do_delta) + len(obs_delta)

                d_sep_list = []

                for item in gamma_list[n_gam:]:
                    # check for d-separation
                    graph_sep = nx.DiGraph(graph_out)
                    graph_sep.remove_edges_from(
                        [item2 for item2 in graph_sep.edges if item2[0] == item]
                    )
                    d_sep = self.d_sep(
                        item, gamma_list[:n_gam], [], graph_sep, conf_out
                    )

                    if d_sep:
                        d_sep_list += [item]

                if d_sep_list:
                    # print(d_sep_list)

                    gamma_list_gamma = gamma_list[:n_gam]
                    gamma_list_delta = [
                        item for item in gamma_list[n_gam:] if item not in d_sep_list
                    ]

                    do_gam_temp, obs_gam_temp = self.conv_from_gamma(gamma_list_gamma)
                    do_del_temp, obs_del_temp = self.conv_from_gamma(gamma_list_delta)

                    gam_temp = [[item[0], item[1] + d_sep_list] for item in do_gam_temp]
                    gam_temp += [[item, d_sep_list] for item in obs_gam_temp]

                    # simplify do-statements to get rid of extraneous interventions
                    gam_temp = []
                    for item in do_gam_temp:
                        gam_temp.append(
                            [
                                item[0],
                                [
                                    item2.replace("*", "", 1) + "*"
                                    for item2 in item[1] + d_sep_list
                                    if item2
                                    in nx.algorithms.dag.ancestors(
                                        graph_out,
                                        item[0] + "_{" + self.str_list(item[1]) + "}",
                                    )
                                ],
                            ]
                        )

                    for item in obs_gam_temp:
                        gam_temp.append(
                            [
                                item,
                                [
                                    item2.replace("*", "", 1) + "*"
                                    for item2 in d_sep_list
                                    if item2
                                    in nx.algorithms.dag.ancestors(graph_out, item)
                                ],
                            ]
                        )

                    # print(gamma_list_gamma)
                    # print(gamma_list_delta)

                    # print('IDC* Step 4')
                    # print(gam_temp)
                    # print(do_del_temp)
                    # print(obs_del_temp)

                    str_temp = self.idc_star_alg(
                        gam_temp, do_del_temp, [], obs_del_temp, graph_temp
                    )

                    # remove extraneous asterisks
                    do_temp, obs_temp = self.conv_from_gamma(gamma_list)
                    do_vars = []
                    for item in do_temp:
                        do_vars += [item2 for item2 in item[1] if item2 not in do_vars]

                    # print(obs_temp)
                    # print(do_temp)
                    for item in obs_temp:
                        if item + "*" in str_temp and item + "*" not in do_vars:
                            str_temp = str_temp.replace(item + "*", item)

                    return str_temp

                else:
                    do_temp, obs_temp = self.conv_from_gamma(gamma_list)

                    # print('IDC* Step 5')

                    P_prime = self.id_star_alg(do_temp, obs_temp, graph_temp)

                    do_temp2, obs_temp2 = self.conv_from_gamma(gamma_list[:n_gam])
                    sum_list = [item[0] for item in do_temp2]
                    sum_list += obs_temp2

                    # remove extraneous asterisks
                    do_vars = []
                    for item in do_temp:
                        do_vars += [item2 for item2 in item[1] if item2 not in do_vars]

                    # print(obs_temp)
                    # print(do_temp)
                    for item in obs_temp:
                        if item + "*" in P_prime and item + "*" not in do_vars:
                            P_prime = P_prime.replace(item + "*", item)

                    return (
                        "["
                        + P_prime
                        + "]/[sum_{"
                        + self.str_list(sum_list)
                        + "}["
                        + P_prime
                        + "]]"
                    )

        return


class str_graph(cg_graph):
    """define class of causal graphs initialized using a list of BEL-statements represented as strings"""

    def __init__(self, str_list, graph_type, type_dict=None):
        if type_dict is None:
            type_dict = {}
        edge_list = []
        entity_list = []

        # construct graph from list of BEL statement strings

        for item in str_list:

            sub_ind = item.find("=")

            sub_temp = item[: sub_ind - 1]
            obj_temp = item[sub_ind + 3 :]

            rel_temp = item[sub_ind : sub_ind + 2]

            if sub_temp not in entity_list:
                entity_list.append(sub_temp)
            if obj_temp not in entity_list:
                entity_list.append(obj_temp)

            nodes_temp = [sub_temp, obj_temp]
            list_temp = [[item[0], item[1]] for item in edge_list]
            if nodes_temp in list_temp:
                ind_temp = list_temp.index(nodes_temp)
                edge_list[ind_temp][2] += "," + rel_temp
            else:
                edge_list.append([sub_temp, obj_temp, rel_temp])

        self.entity_list = entity_list
        self.edge_list = edge_list

        self.proc_data(graph_type, type_dict)

    # create class of causal graph nodes


class cg_node:
    """Define a superclass of nodes for a causal graph"""

    def __init__(self, n_inputs, name, node_type):

        self.n_inputs = n_inputs
        self.name = name
        self.node_type = node_type

        if n_inputs == 0:
            self.label = "exogenous"
        else:
            self.label = "endogenous"

        return


class scm_node(cg_node):
    """Define a node that uses a sigmoid regression with additive noise within the sigmoid.
    Continuous variables are scaled to lie between min and max values.
    Binary variables use a thresholding function on the sigmoid."""

    def __init__(self, n_inputs, name, node_type):
        super().__init__(n_inputs, name, node_type)

        return

    def logistic_fcn(self, data_in, alpha):
        """Evaluate the logistic function with a linear regression in the function's argument."""

        return torch.sigmoid(torch.matmul(data_in, alpha[1:]) + alpha[0])

    def logistic_inv(self, data_in):
        """Evaluate the inverse of the logistic function."""

        return torch.log(data_in / (1 - data_in))

    def dup_check(self, data_np, n_max):
        # identify how many distinct points are present in the input data

        n_pts = 0
        distinct_pts = [data_np[0]]
        for item in data_np[1:]:
            if np.min([np.sqrt((item - item2) ** 2) for item2 in distinct_pts]) > 1e-3:
                distinct_pts.append(item)
                n_pts += 1
                if n_pts == n_max:
                    return n_max

        return n_pts

    def data_bin(self, data_in, data_out):
        """Use k-means clustering to identify centroids and assign data points to clusters."""

        # get total number of data points
        n_data = data_in.shape[0]
        n_cents = int(n_data ** (1 / self.n_inputs) / 10)

        # convert to numpy
        data_np = np.asarray([[item2.item() for item2 in item] for item in data_in])
        out_np = np.asarray([item.item() for item in data_out])

        # check for duplicate points
        n_cents = self.dup_check(data_np, n_cents)

        kmeans = KMeans(n_clusters=n_cents, random_state=0).fit(data_np)

        m_vals = torch.tensor([np.sum(kmeans.labels_ == i) for i in range(0, n_cents)])

        p_vals = (
            torch.tensor(
                [np.sum(out_np[kmeans.labels_ == i]) for i in range(0, n_cents)]
            )
            / m_vals
        )

        # make sure that p is never 0 or 1
        p_vals = p_vals * (1 - 2e-3) + 1e-3

        # calculate p_j*m_j
        pm_vals = m_vals * self.logistic_inv(p_vals) / n_data

        # calculate centroids * m_j
        centroids_m = torch.tensor(
            [
                kmeans.cluster_centers_[i, :] * m_vals[i].item()
                for i in range(0, n_cents)
            ]
        )

        return pm_vals, centroids_m

    def prob_init(self, input_data, var_data, lr):
        """Initialize node parameters associated with its distribution."""

        if self.node_type == "continuous":

            n_data = input_data.shape[0]

            # normalize with respect to y_max and y_min

            self.y_max = torch.max(var_data) + 1e-3 * torch.abs(torch.max(var_data))
            self.y_min = torch.min(var_data) - 1e-3 * torch.abs(torch.min(var_data))

            out_data = (var_data - self.y_min) / (self.y_max - self.y_min)
            x_temp = input_data

            big_y = self.logistic_inv(out_data)

        elif self.node_type == "binary" or self.node_type == "ternary":

            self.y_max = 1.0
            self.y_min = 0.0

            # bin the data
            big_y, x_temp = self.data_bin(input_data, var_data)
            n_data = x_temp.shape[0]

        else:
            print("node type not recognized")

        # add vector of ones to input data
        if self.n_inputs > 0:

            big_x = torch.cat([torch.ones((n_data, 1)), x_temp], dim=1)
            self.alpha = (
                torch.matmul(
                    torch.matmul(
                        torch.inverse(
                            torch.matmul(torch.t(big_x) / n_data, big_x / n_data)
                        ),
                        torch.t(big_x),
                    ),
                    big_y,
                )
                / n_data ** 2
            )

            self.std = torch.sqrt(
                torch.mean((big_y - torch.matmul(big_x, self.alpha)) ** 2)
            )

        else:
            self.alpha = torch.mean(big_y)
            self.std = torch.sqrt(torch.mean((big_y - self.alpha) ** 2))

        return

    def sample(self, data_in=[]):

        eps = pyro.sample(self.name + "_eps", pyro.distributions.Normal(0, self.std))

        if self.n_inputs > 0:
            y_mean = torch.sum(data_in * self.alpha[1:]) + self.alpha[0]

        else:
            y_mean = self.alpha

        y_arg = pyro.sample(
            self.name + "_y", pyro.distributions.Normal(y_mean + eps, 0)
        )

        if self.node_type == "continuous":
            return (self.y_max - self.y_min) * torch.sigmoid(y_arg) + self.y_min, eps
        elif self.node_type == "binary":
            return torch.round(torch.sigmoid(y_arg)), eps
        else:
            print("node type not recognized")
            return
