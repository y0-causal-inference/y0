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