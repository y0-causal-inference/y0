"""Implementation of algorithms from Hierarchical Causal Models by E.N. Weinstein and D.M. Blei."""

from collections.abc import Iterable
from itertools import combinations

import pygraphviz as pgv

from y0.dsl import Variable
from y0.graph import NxMixedGraph

__all__ = [
    "get_observed",
    "HCM_from_lists",
    "get_directed_edges",
    "get_undirected_edges",
    "get_unobserved",
    "get_subunits",
    "get_units",
    "parents",
    "create_Qvar",
    "convert_to_HCGM",
    "copy_HCM",
    "direct_unit_descendents",
    "collapse_HCM",
    "augment_collapsed_model",
    "marginalize_augmented_model",
]


def HCM_from_lists(
    *,
    obs_subunits: list[str] | None = None,
    unobs_subunits: list[str] | None = None,
    obs_units: list[str] | None = None,
    unobs_units: list[str] | None = None,
    edges: list[str] | None = None,
) -> pgv.AGraph:
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
    for obs in obs_subunits + obs_units:
        HCM.add_node(obs, style="filled", color="lightgrey")
    for unobs in unobs_subunits + unobs_units:
        HCM.add_node(unobs)
    for edge in edges or []:
        HCM.add_edge(edge)
    HCM.add_subgraph(
        obs_subunits + unobs_subunits, name="cluster_subunits", style="dashed", label="m"
    )
    return HCM


def get_observed(HCM: pgv.AGraph) -> set[pgv.Node]:
    """Return the set of observed variables (both unit and subunit) in the HCM."""
    observed_nodes = set()
    for node_name in HCM.nodes():
        node = HCM.get_node(node_name)
        if node.attr.get("style") == "filled":
            observed_nodes.add(node_name)
    return observed_nodes


def get_unobserved(HCM: pgv.AGraph) -> set[pgv.Node]:
    """Return the set of unobserved variables (both unit and subunit) in the HCM."""
    all_nodes = set(HCM.nodes())
    return all_nodes - get_observed(HCM)


def get_subunits(HCM: pgv.AGraph) -> set[pgv.Node]:
    """Return the set of subunit variables in the HCM."""
    return set(HCM.get_subgraph("cluster_subunits").nodes())


def get_units(HCM: pgv.AGraph) -> set[pgv.Node]:
    """Return the set of unit variables in the HCM."""
    subunits = get_subunits(HCM)
    return set(HCM.nodes()) - subunits


def parents(HCM: pgv.AGraph, node: pgv.Node) -> set[pgv.Node]:
    """Return the set of parent/predecessor variables of the given variable in the HCM."""
    parents = set(HCM.predecessors(node))
    return parents


def copy_HCM(HCM: pgv.AGraph) -> pgv.AGraph:
    """Return a copy of the HCM."""
    obs = get_observed(HCM)
    unobs = get_unobserved(HCM)
    units = get_units(HCM)
    subunits = get_subunits(HCM)
    copy = HCM_from_lists(
        obs_subunits=list(obs & subunits),
        unobs_subunits=list(unobs & subunits),
        obs_units=list(obs & units),
        unobs_units=list(unobs & units),
        edges=HCM.edges(),
    )
    return copy


def _node_string(nodes: Iterable[pgv.Node]) -> str:
    """Return a formated string for use in creating Q variables for collapsed HCMs."""
    s = ""
    for node in nodes:
        s += node.get_name().lower() + ","
    return s[:-1]


def create_Qvar(HCM: pgv.AGraph, subunit_node: pgv.Node) -> Variable:
    """Return a y0 Variable for the unit-level Q variable of the given subunit variable in the HCM."""
    subunit_parents = parents(HCM, subunit_node) & get_subunits(HCM)
    parent_str = _node_string(sorted(subunit_parents))
    if parent_str == "":
        Q_str = "Q_" + subunit_node.lower()
    else:
        Q_str = "Q_{" + subunit_node.lower() + "|" + parent_str + "}"
    return Variable(Q_str)


def convert_to_HCGM(HCM: pgv.AGraph) -> pgv.AGraph:
    """Convert an HCM to an HCGM with promoted Q variables."""
    HCGM = copy_HCM(HCM)
    observed = get_observed(HCM)
    subunits = get_subunits(HCM)
    for s in subunits:
        Q = create_Qvar(HCGM, s)
        parent_set = set(parents(HCM, s))
        if (s in observed) & ((parent_set & subunits) <= observed):
            HCGM.add_node(Q, style="filled", color="lightgrey")
        else:
            HCGM.add_node(Q)
        for unit_parent in parent_set & get_units(HCGM):
            HCGM.delete_edge(unit_parent, s)
            HCGM.add_edge(unit_parent, Q)
        HCGM.add_edge(Q, s)
    HCGM.subgraphs()[0].graph_attr["style"] = "solid"
    return HCGM


def get_directed_edges(HCM: pgv.AGraph) -> list[str]:
    """Return the list of directed edges in the HCM that do not contain latent variables."""
    edges = []
    unobserved = get_unobserved(HCM)
    for edge in HCM.edges():
        if edge[0] in unobserved:
            pass
        else:
            edges.append(edge)
    return edges


def get_undirected_edges(HCM: pgv.AGraph) -> list[str]:
    """Return the list of undirected edges in the HCM generated by its latent variables."""
    edges = []
    unobserved = get_unobserved(HCM)
    for u in unobserved:
        descends = HCM.successors(u)
        for pair in combinations(descends, r=2):
            edges.append(pair)
    return edges


def direct_unit_descendents(HCM: pgv.AGraph, subunit_node: pgv.Node) -> set[pgv.Node]:
    """Return the set of direct unit descendents of the given subunit variable in the HCM."""
    units = get_units(HCM)
    subunits = get_subunits(HCM)
    descendents = set(HCM.successors(subunit_node))
    duds = set()
    go = True
    while go:
        if descendents == set():
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
                descendents.update(HCM.successors(nd))
    return duds


def collapse_HCM(HCM: pgv.AGraph, return_HCGM: bool = False) -> NxMixedGraph:
    """Return a collapsed hierarchical causal model.

    :param HCM: pygraphviz AGraph of the hierarchical causal model to be collapsed
    :param return_HCGM: if True, returns the intermediate HCGM with subunits and promoted Q variables
    :returns: NxMixedGraph
    """
    HCGM = convert_to_HCGM(HCM)
    if return_HCGM:
        HCGM_original = convert_to_HCGM(HCM)
    subunits = get_subunits(HCM)
    for s in subunits:
        Q = create_Qvar(HCM, s)
        for dud in direct_unit_descendents(HCM, s):
            HCGM.add_edge(Q, dud)
        HCGM.delete_node(s)
    undirected = get_undirected_edges(HCGM)
    directed = get_directed_edges(HCGM)
    collapsed = NxMixedGraph.from_edges(directed=directed, undirected=undirected)
    if return_HCGM:
        return (collapsed, HCGM_original)
    else:
        return collapsed


def augment_collapsed_model(
    collapsed: NxMixedGraph, augmentation_variable: Variable, mechanism: Iterable[Variable]
):
    """Augment a collapsed model with a given augmentation variable and its mechanism.

    :param collapsed: NxMixedGraph of the input collapsed model
    :param augmentation_variable: new variable to add into the collapsed model
    :param mechanism: collection of variables in the collapsed model that determine the augmentation_variable
    :raises ValueError: input mechanism variables must be contained in the collapsed model
    :returns: NxMixedGraph of the augmented model
    """
    augmented = collapsed.copy()
    mechanism = set(mechanism)
    if not mechanism <= collapsed.nodes():
        raise ValueError("The input mechanism must be contained in the collapsed model.")
    aug = augmentation_variable
    augmented.add_node(aug) 
    for var in mechanism:
        augmented.add_directed_edge(var, aug)
    for var in set(augmented.nodes()) - {aug}:
        parents = set(augmented.directed.predecessors(var))
        if mechanism <= parents:
            augmented.add_directed_edge(aug, var)
            for parent in mechanism:
                augmented.directed.remove_edge(parent, var)
    return augmented


def marginalize_augmented_model(
    augmented: NxMixedGraph, augmentation_variable: Variable, marginal_parents: Iterable[Variable]
):
    """Marginalize out a given collection of variables from an augmented model.

    :param augmented: NxMixedGraph of the input augmented model
    :param augmentation_variable: the variable that was previously augmented into the model
    :param marginal_parents: collection of parents of the augmentation variable to be marginalized out.
    :raises ValueError: augmentation_variable must be in the augmented model
    :raises ValueError: marginal_parents cannot be all of the parents of augmentation_variable
    :raises ValueError: augmentation_variable must be the only child of the each marginal parent
    :returns: NxMixedGraph of the marginaled model
    """
    marginalized = augmented.copy()
    check_set = {augmentation_variable}
    mechanism = set(augmented.directed.predecessors(augmentation_variable))
    if augmentation_variable not in augmented.nodes():
        raise ValueError("Augmentation variable must be in the input augmented model.")
    if set(marginal_parents) == mechanism:
        raise ValueError("Cannot marginalize all parents of the augmentation varaible.")
    for parent in marginal_parents:
        if set(marginalized.directed.successors(parent)) == check_set:
            grandparents = marginalized.directed.predecessors(parent)
            for gp in grandparents:
                marginalized.add_directed_edge(gp, augmentation_variable)
            marginalized.directed.remove_node(parent)
        else:
            raise ValueError(
                "The augmentation variable must be the only child of the marginalized parents."
            )
    return marginalized
