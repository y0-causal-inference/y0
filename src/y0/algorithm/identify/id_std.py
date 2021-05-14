# -*- coding: utf-8 -*-

from y0.algorithm.ident.utils import nxmixedgraph_to_causal_graph
from y0.dsl import Expression
from y0.graph import NxMixedGraph
from y0.identify import _get_outcomes, _get_treatments
from y0.parser.craig.grammar import grammar


def identify(graph: NxMixedGraph, query: Expression) -> Expression:
    """Currently a wrapper for bel2scm.causal_graph.id_alg()"""
    cg = nxmixedgraph_to_causal_graph(graph)
    treatments = _get_treatments(query.get_variables())
    outcomes = _get_outcomes(query.get_variables())
    expr = cg.id_alg(outcomes, treatments)
    return grammar.parseString(expr)[0]
