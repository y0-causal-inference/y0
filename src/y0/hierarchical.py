import pygraphviz as pgv
import networkx as nx
from IPython.display import SVG
from y0.graph import NxMixedGraph
from y0.dsl import Variable

def HCM_from_lists(*, obs_subunits=[], unobs_subunits=[], obs_units=[], unobs_units=[], edges=[]):
    HCM = pgv.AGraph(directed=True)
    for obs in (obs_subunits+obs_units):
        HCM.add_node(obs, style="filled", color="lightgrey")
    for unobs in (unobs_subunits+unobs_units):
        HCM.add_node(unobs)
    for edge in edges:
        HCM.add_edge(edge)
    HCM.add_subgraph(obs_subunits+unobs_subunits, name="cluster_subunits", style="dashed", label="m")
    return HCM

def get_observed(HCM):
    observed_nodes = set()
    for node_name in HCM.nodes():
        node = HCM.get_node(node_name)
        if node.attr.get('style') == 'filled':
            observed_nodes.add(node_name)
    return observed_nodes

def get_unobserved(HCM):
    all_nodes = set(HCM.nodes())
    return all_nodes - get_observed(HCM)

def get_subunits(HCM):
    return set(HCM.get_subgraph('cluster_subunits').nodes())

def get_units(HCM):
    subunits = get_subunits(HCM)
    return set(HCM.nodes()) - subunits

def parents(HCM, node):
    parents = set(HCM.predecessors(node))
    return parents

def node_string(nodes):
    s = ""
    for node in nodes:
        s += node.get_name().lower() + ","
    return s[: -1]

def create_Qvar(HCM, subunit_node):
    subunit_parents = parents(HCM, subunit_node) & get_subunits(HCM)
    parent_str = node_string(subunit_parents)
    if parent_str == '':
        Q_str = 'Q_'+subunit_node.lower()
    else:
        Q_str = 'Q_{'+subunit_node.lower()+'|'+parent_str+'}'
    return Variable(Q_str)

def direct_unit_descendents(HCM, subunit_node):
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

def collapse_HCM(HCM):
    # unobs_Qs = set()
    directed_edges = []
    undirected_edges = []
    units = get_units(HCM)
    # unit_vars = [Variable(unit) for unit in units]
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
