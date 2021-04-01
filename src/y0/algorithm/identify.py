from y0.dsl import Expression, P, Sum, X, Y, Z
from y0.parser.craig.grammar import grammar
from y0.graph import NxMixedGraph
from bel2scm import causal_graph
from y0.identify import _get_treatments, _get_outcomes

def identify(graph:NxMixedGraph, query:Expression) -> Expression:
    """Currently a wrapper for bel2scm.causal_graph.id_alg()"""
    cg = nxmixedgraph_to_causal_graph(graph)
    treatments = _get_treatments(query.get_variables())
    outcomes = _get_outcomes(query.get_variables())
    expr = cg.id_alg(outcomes, treatments)
    return grammar.parseString(expr)[0]


def nxmixedgraph_to_causal_graph(graph:NxMixedGraph) -> causal_graph:
    """Converts NxMixedGraph to bel2scm.causal_graph"""
    di_edges = list(graph.directed.edges())
    bi_edges = list(graph.undirected.edges())
    vertices = list(graph.directed)  # could be either since they're maintained together
    str_list = [f'{U} => {V}' for U, V in di_edges]
    type_dict = dict([(U, "continuous") for U in vertices])
    cg = causal_graph.str_graph(str_list, 'SCM', type_dict)
    cg.add_confound([[U, V] for U, V in bi_edges])
    return cg

