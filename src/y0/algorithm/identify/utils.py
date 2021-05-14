# -*- coding: utf-8 -*-

"""Utilities for identifiaction algorithms"""

from bel2scm import causal_graph

from y0.graph import NxMixedGraph

__all__ = [
    'causal_graph',
    'nxmixedgraph_to_causal_graph',
]


# TODO copy code for causal_graph class

def nxmixedgraph_to_causal_graph(graph: NxMixedGraph):
    """Converts NxMixedGraph to bel2scm.causal_graph"""
    di_edges = list(graph.directed.edges())
    bi_edges = list(graph.undirected.edges())
    vertices = list(graph.directed)  # could be either since they're maintained together
    str_list = [f'{U} => {V}' for U, V in di_edges]
    type_dict = dict([(U, "continuous") for U in vertices])
    cg = causal_graph.str_graph(str_list, 'SCM', type_dict)
    cg.add_confound([[U, V] for U, V in bi_edges])
    return cg
