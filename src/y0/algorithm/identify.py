from y0.dsl import Expression, P, Sum, X, Y, Z
from y0.parser.craig.grammar import grammar
from y0.graph import NxMixedGraph
from bel2scm import causal_graph as cg

def identify(graph,query):
    graph_craig = nxmixedgraph_to_craig(graph)
    cause, effect = cause_effect_from_query(query)
    expr = graph_craig.id_alg(effect,cause)
    return grammar.parseString(expr)[0]
#grammar.parseString(s)

def nxmixedgraph_to_craig(graph:NxMixedGraph):
    di_edges = list(graph.directed.edges())
    bi_edges = list(graph.undirected.edges())
    vertices = list(graph.directed)  # could be either since they're maintained together
    str_list = [f'{U} => {V}' for U, V in di_edges]
    type_dict = dict([(U, "continuous") for U in vertices])
    graph_craig = cg.str_graph(str_list,'SCM',type_dict)
    graph_craig.add_confound([[U,V] for U,V in bi_edges])
    return graph_craig

def cause_effect_from_query(query:Expression):
    return ['X'],['Y']