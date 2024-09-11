"""Implementation of algorithms from Hierarchical Causal Models by E.N. Weinstein and D.M. Blei."""

from collections.abc import Iterable

import pygraphviz as pgv

from y0.dsl import Variable
from y0.graph import NxMixedGraph

__all__ = [
    "get_observed",
    "HCM_from_lists",
    "get_unobserved",
    "get_subunits",
    "get_units",
    "parents",
    "create_Qvar",
    "direct_unit_descendents",
    "collapse_HCM"]

def HCM_from_lists(*,
                   obs_subunits: list[str] | None = None,
                   unobs_subunits: list[str] | None = None,
                   obs_units: list[str] | None = None,
                   unobs_units: list[str] | None = None,
                   edges: list[str] | None = None) -> pgv.AGraph:
    """Create a hierarchical causal model from the given node and edge lists.

    :param obs_subunits: a list of names for the observed subunit variables
    :param unobs_subunits: a list of names for the unobserved subunit variables
    :param obs_units: a list of names for the observed unit variables
    :param unobs_units: a list of names for the unobserved unit variables
    :param edges: a list of edges
    :returns: a pygraphviz AGraph with subunit variables in the 'cluster_subunits' subgraph
    """
    if obs_subunits is None:
        obs_subunits = []
    if unobs_subunits is None:
        unobs_subunits = []
    if obs_units is None:
        obs_units = []
    if unobs_units is None:
        unobs_units = []
    HCM = pgv.AGraph(directed=True)
    for obs in (obs_subunits+obs_units):
        HCM.add_node(obs, style="filled", color="lightgrey")
    for unobs in (unobs_subunits+unobs_units):
        HCM.add_node(unobs)
    for edge in edges or []:
        HCM.add_edge(edge)
    HCM.add_subgraph(obs_subunits+unobs_subunits, name="cluster_subunits", style="dashed", label="m")
    return HCM

def get_observed(HCM: pgv.AGraph) -> set[pgv.Node]:
    """Return the set of observed variables (both unit and subunit) in the HCM."""
    observed_nodes = set()
    for node_name in HCM.nodes():
        node = HCM.get_node(node_name)
        if node.attr.get('style') == 'filled':
            observed_nodes.add(node_name)
    return observed_nodes

def get_unobserved(HCM: pgv.AGraph) -> set[pgv.Node]:
    """Return the set of unobserved variables (both unit and subunit) in the HCM."""
    all_nodes = set(HCM.nodes())
    return all_nodes - get_observed(HCM)

def get_subunits(HCM: pgv.AGraph) -> set[pgv.Node]:
    """Return the set of subunit variables in the HCM."""
    return set(HCM.get_subgraph('cluster_subunits').nodes())

def get_units(HCM: pgv.AGraph) -> set[pgv.Node]:
    """Return the set of unit variables in the HCM."""
    subunits = get_subunits(HCM)
    return set(HCM.nodes()) - subunits

def parents(HCM: pgv.AGraph, node: pgv.Node) -> set[pgv.Node]:
    """Return the set of parent/predecessor variables of the given variable in the HCM."""
    parents = set(HCM.predecessors(node))
    return parents

def _node_string(nodes: Iterable[pgv.Node]) -> str:
    """Return a formated string for use in creating Q variables for collapsed HCMs."""
    s = ""
    for node in nodes:
        s += node.get_name().lower() + ","
    return s[: -1]

def create_Qvar(HCM: pgv.AGraph, subunit_node: pgv.Node) -> Variable:
    """Return a y0 Variable for the unit-level Q variable of the given subunit variable in the HCM."""
    subunit_parents = parents(HCM, subunit_node) & get_subunits(HCM)
    parent_str = _node_string(subunit_parents)
    if parent_str == '':
        Q_str = 'Q_'+subunit_node.lower()
    else:
        Q_str = 'Q_{'+subunit_node.lower()+'|'+parent_str+'}'
    return Variable(Q_str)

def direct_unit_descendents(HCM: pgv.AGraph, subunit_node: pgv.Node) -> set[pgv.Node]:
    """Return the set of direct unit descendents of the given subunit variable in the HCM."""
    units = get_units(HCM)
    subunits = get_subunits(HCM)
    descendents = HCM.successors(subunit_node)
    duds = set()
    go = True
    while go:
        if descendents == []:
            go = False
        else:
            next_descendents = []
            for d in descendents:
                if d in units:
                    duds.add(d)
                elif d in subunits:
                    next_descendents.append(d)
            descendents = set()
            for nd in next_descendents:
                try:
                    descendents.add(*HCM.successors(nd))
                except TypeError:
                    pass
            descendents = list(descendents)
    return duds

def collapse_HCM(HCM: pgv.AGraph) -> NxMixedGraph:
    """Return a collapsed hierarchical causal model.

    :param HCM: pygraphviz AGraph of the hierarchical causal model to be collapsed
    :returns: NxMixedGraph
    """
    # unobs_Qs = set()
    directed_edges = []
    undirected_edges = []
    units = get_units(HCM)
    subunits = get_subunits(HCM)
    observed = get_observed(HCM)
    for s in subunits:
        Q = create_Qvar(HCM, s)
        parents_set = set(parents(HCM, s))
        if (s in observed) & ((parents_set & subunits) <= observed):
            for unit_parent in (parents_set & units):
                if unit_parent in observed:
                    edge = (Variable(unit_parent), Q)
                    directed_edges.append(edge)
                else:
                    descends = HCM.successors(unit_parent)
                    if len(descends) != 2:
                        raise ValueError("Latent variables must have exactly 2 descendents")
                    for d in descends:
                        if d == s:
                            pass
                        else:
                            other_descend = d
                    if other_descend in subunits:
                        edge = (Q, create_Qvar(HCM, other_descend))
                    else:
                        edge = (Q, Variable(other_descend))
                    undirected_edges.append(edge)

            for dud in direct_unit_descendents(HCM, s):
                edge = (Q, Variable(dud))
                directed_edges.append(edge)
        else:
            raise ValueError("Unobserved Q variables not currently supported")

    return NxMixedGraph.from_edges(directed=directed_edges, undirected=undirected_edges)
