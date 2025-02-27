"""Implementation of algorithms from Hierarchical Causal Models by E.N. Weinstein and D.M. Blei.

.. seealso:: https://arxiv.org/abs/2401.05330
"""

from __future__ import annotations

import itertools as itt
import typing
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any, TypeAlias

import networkx as nx

from y0.dsl import Variable
from y0.graph import NxMixedGraph

if TYPE_CHECKING:
    import pygraphviz

__all__ = [
    "HierarchicalCausalModel",
    "HierarchicalStructuralCausalModel",
    "QVariable",
    "augment_collapsed_model",
    "augment_from_mechanism",
    "augmentation_mechanism",
    "marginalize_augmented_model",
]

SubunitGraph: TypeAlias = nx.DiGraph

SUBUNITS_KEY = "cluster_subunits"

VHint: TypeAlias = typing.Union[str, Variable, "QVariable"]


def _upgrade(v: VHint) -> Variable:
    if not isinstance(v, str):
        return v
    if v.startswith("Q_"):
        return QVariable.parse_str(v)
    return Variable(v)


class HierarchicalCausalModel:
    """A class that wraps HCM functionality."""

    observed: set[Variable]
    subunits: set[Variable]

    def __init__(self) -> None:
        """Initialize the HCM."""
        self._graph = nx.DiGraph()
        self.observed = set()
        self.subunits = set()

    def add_observed_node(self, node: VHint) -> None:
        """Add an observed node."""
        node = _upgrade(node)
        self._graph.add_node(node)
        self.observed.add(node)

    def add_unobserved_node(self, node: VHint) -> None:
        """Add an unobserved node."""
        self._graph.add_node(_upgrade(node))

    def add_edge(
        self,
        u: VHint,
        v: VHint,
        **kwargs: Any,
    ) -> None:
        """Add an edge."""
        self._graph.add_edge(_upgrade(u), _upgrade(v), **kwargs)

    def add_subunits(self, subunit_nodes: Iterable[VHint]) -> None:
        """Annotate the given nodes as the subunit graph."""
        self.subunits.update(_upgrade(x) for x in subunit_nodes)

    def is_node_observed(self, node: VHint) -> bool:
        """Check if the node is observed."""
        return _upgrade(node) in self.observed

    def get_observed(self) -> set[Variable]:
        """Return the set of observed variables (both unit and subunit) in the HCM."""
        return self.observed

    def get_unobserved(self) -> set[Variable]:
        """Return the set of unobserved variables (both unit and subunit) in the HCM."""
        return set(self._graph.nodes()) - self.observed

    def get_subunits(self) -> set[Variable]:
        """Return the set of subunit variables in the HCM."""
        return self.subunits

    def get_units(self) -> set[Variable]:
        """Return the set of unit variables in the HCM."""
        return set(self._graph.nodes()) - self.subunits

    def get_subunit_graph(self) -> SubunitGraph:
        """Return the subunit subgraph of the input HCM."""
        return nx.subgraph(self._graph, self.subunits).copy()

    def get_parents(self, node: VHint) -> set[Variable]:
        """Return the set of parent/predecessor variables of the given variable in the HCM."""
        return set(self._graph.predecessors(_upgrade(node)))

    def delete_node(self, node: VHint) -> None:
        """Delete a node."""
        self._graph.remove_node(_upgrade(node))

    def delete_edge(self, u: VHint, v: VHint) -> None:
        """Delete an edge."""
        self._graph.remove_edge(_upgrade(u), _upgrade(v))

    def nodes(self) -> list[Variable]:
        """Get all nodes."""
        return list(self._graph.nodes())

    def predecessors(self, node: VHint) -> list[Variable]:
        """Get predecessors."""
        return list(self._graph.predecessors(_upgrade(node)))

    def successors(self, node: VHint) -> list[Variable]:
        """Get successors."""
        return list(self._graph.successors(_upgrade(node)))

    # def set_subgraph_style(self, style: str) -> None:
    #     """Set the style on the subgraph."""
    #     self.graph.subgraphs()[0].graph_attr["style"] = style
    #
    # def set_shape(self, v: VHint, shape: str) -> None:
    #     """Set the shape of a node."""
    #     node = self.graph.get_node(_safe_q(v))
    #     node.attr["shape"] = shape

    def edges(self) -> list[tuple[Variable, Variable]]:
        """Get all edges."""
        return list(self._graph.edges())

    def get_direct_unit_descendants(self, subunit_node: VHint) -> set[Variable]:
        """Return the set of direct unit descendants of the given subunit variable in the HCM."""
        descendants = set(self.successors(_upgrade(subunit_node)))
        direct_unit_descendants = set()
        while descendants:
            new_descendants = set()
            for descendant in descendants:
                if descendant in self.subunits:
                    new_descendants.update(self.successors(descendant))
                else:
                    direct_unit_descendants.add(descendant)
            descendants = new_descendants
        return direct_unit_descendants

    @classmethod
    def from_lists(
        cls,
        *,
        observed_subunits: Sequence[VHint] | None = None,
        unobserved_subunits: Sequence[VHint] | None = None,
        observed_units: Sequence[VHint] | None = None,
        unobserved_units: Sequence[VHint] | None = None,
        edges: Sequence[tuple[VHint, VHint]] | None = None,
    ) -> HierarchicalCausalModel:
        """Create a hierarchical causal model from the given node and edge lists.

        :param observed_subunits: a list of names for the observed subunit variables
        :param unobserved_subunits: a list of names for the unobserved subunit variables
        :param observed_units: a list of names for the observed unit variables
        :param unobserved_units: a list of names for the unobserved unit variables
        :param edges: a list of edges
        :returns: a hierarchical causal model with subunit variables in the :data:`SUBUNITS_KEY` subgraph
        """
        if observed_subunits is None:
            observed_subunits = []
        if unobserved_subunits is None:
            unobserved_subunits = []
        if observed_units is None:
            observed_units = []
        if unobserved_units is None:
            unobserved_units = []
        hcm = cls()
        for observed_node in itt.chain(observed_subunits, observed_units):
            hcm.add_observed_node(observed_node)
        for unobserved_node in itt.chain(unobserved_subunits, unobserved_units):
            hcm.add_unobserved_node(unobserved_node)
        for u, v in edges or []:
            hcm.add_edge(u, v)
        hcm.add_subunits(itt.chain(observed_subunits, unobserved_subunits))
        return hcm

    def copy_hcm(self) -> HierarchicalCausalModel:
        """Return a copy of the HCM."""
        obs = self.get_observed()
        unobs = self.get_unobserved()
        units = self.get_units()
        subunits = self.get_subunits()
        copy = self.from_lists(
            observed_subunits=list(obs & subunits),
            unobserved_subunits=list(unobs & subunits),
            observed_units=list(obs & units),
            unobserved_units=list(unobs & units),
            edges=self._graph.edges(),
        )
        return copy

    def to_hcgm(self: HierarchicalCausalModel) -> HierarchicalCausalModel:
        """Convert an HCM to a hierarchical causal graphical model (HCGM) with promoted Q variables."""
        hcgm = self.copy_hcm()
        observed = self.get_observed()
        subunits = self.get_subunits()
        subunit_graph = self.get_subunit_graph()
        for s in subunits:
            q_variable = _create_qvar(subunit_graph, s)
            parent_set = set(self.get_parents(s))
            if (s in observed) & ((parent_set & subunits) <= observed):
                hcgm.add_observed_node(q_variable)
            else:
                hcgm.add_unobserved_node(q_variable)
            for unit_parent in parent_set & hcgm.get_units():
                hcgm.delete_edge(unit_parent, s)
                hcgm.add_edge(unit_parent, q_variable)
            hcgm.add_edge(q_variable, s)
        # TODO what's this for? Is it used besides making diagrams?
        # hcgm.set_subgraph_style("solid")
        return hcgm

    def to_hscm(self: HierarchicalCausalModel) -> HierarchicalStructuralCausalModel:
        """Convert the input HCM to an explicit hierarchical structural causal model (HSCM)."""
        obs = self.get_observed()
        unobs = self.get_unobserved()
        units = self.get_units()
        subunits = self.get_subunits()
        edges = self.edges()
        hscm = HierarchicalStructuralCausalModel.from_lists(
            observed_subunits=list(obs & subunits),
            unobserved_subunits=list(unobs & subunits),
            observed_units=list(obs & units),
            unobserved_units=list(unobs & units),
            edges=edges,
        )
        return hscm

    def to_admg(self, *, return_hcgm: bool = False) -> NxMixedGraph:
        """Return a collapsed hierarchical causal model.

        :param return_hcgm:
            if True, returns the intermediate hierarchical causal
            graphical models (HCGM) with subunits and promoted Q variables
        :returns: a mixed graph
        """
        hcgm = self.to_hcgm()
        if return_hcgm:
            hgcm_original = self.to_hcgm()
        subunit_graph = self.get_subunit_graph()
        q_variables: set[QVariable] = set()
        for subunit in self.get_subunits():
            q_variable = _create_qvar(subunit_graph, subunit)
            q_variables.add(q_variable)
            for dud in self.get_direct_unit_descendants(subunit):
                hcgm.add_edge(q_variable, dud)
            hcgm.delete_node(subunit)
        undirected = [
            pair
            for node in hcgm.get_unobserved()
            for pair in combinations(hcgm.successors(node), r=2)
        ]
        directed = [(source, target) for source, target in hcgm.edges() if source in hcgm.observed]
        collapsed = NxMixedGraph.from_edges(directed=directed, undirected=undirected)
        for q_variable in q_variables:  # loop to check for and add disconnected Q variables
            if q_variable not in collapsed:
                collapsed.add_node(q_variable)
        if return_hcgm:
            return collapsed, hgcm_original  # type:ignore
        else:
            return collapsed

    def to_pygraphviz(self) -> pygraphviz.AGraph:
        """Get a pygraphviz object."""
        import pygraphviz as pgv

        def _pgv(n: Variable) -> str:
            if isinstance(n, QVariable):
                return n.pgv_str()
            else:
                return n.name

        rv = pgv.AGraph(directed=True)
        for node in self._graph.nodes():
            if node in self.observed:
                rv.add_node(_pgv(node), style="filled", color="lightgrey")
            else:
                rv.add_node(_pgv(node))

        rv.add_subgraph(
            [_pgv(node) for node in self.subunits],
            name=SUBUNITS_KEY,
            style="dashed",
            label="m",
        )

        for u, v in self._graph.edges():
            rv.add_edge(_pgv(u), _pgv(v))

        return rv


class HierarchicalStructuralCausalModel(HierarchicalCausalModel):
    """A subclass of HCM that wraps HSCM functionality."""

    exogenous_noise: set[Variable]

    def __init__(self) -> None:
        """Initialize the HSCM."""
        self.exogenous_noise = set()
        super().__init__()

    def add_unobserved_node(self, node: VHint) -> None:
        """Add an unobserved node and its exogenous noise."""
        node = _upgrade(node)
        unit_exogenous = _upgrade(f"y_i^{node}")
        subunit_exogenous = _upgrade(
            f"e_ij^{node}"
        )  # TODO how to do e_{ij} while also formatting {node}?
        self._graph.add_node(node)
        self._graph.add_edge(unit_exogenous, node)
        self._graph.add_edge(subunit_exogenous, node)
        self.add_subunits([subunit_exogenous])
        self.exogenous_noise.update({unit_exogenous, subunit_exogenous})

    def add_observed_node(self, node: VHint) -> None:
        """Add an observed node and its exogenous noise."""
        node = _upgrade(node)
        self.add_unobserved_node(node)
        self.observed.add(node)

    def add_edge(
        self,
        u: VHint,
        v: VHint,
        **kwargs: Any,
    ) -> None:
        """Add an edge."""
        if any(node in self.exogenous_noise for node in {u, v}):
            raise ValueError("Cannot add an edge to or from exogenous noise variables.")
        else:
            HierarchicalCausalModel.add_edge(self, u, v, **kwargs)
            # self._graph.add_edge(_upgrade(u), _upgrade(v), **kwargs)

    def get_exogenous_noise(self) -> set[Variable]:
        """Return the set of exogenous noise variables in the HSCM."""
        return self.exogenous_noise

    def to_hcm(self) -> HierarchicalCausalModel:
        """Convert the HSCM to a hierarchical causal model (HCM)."""
        endogenous = set(self.nodes()) - self.get_exogenous_noise()
        obs = self.get_observed() & endogenous
        unobs = self.get_unobserved() & endogenous
        units = self.get_units() & endogenous
        subunits = self.get_subunits() & endogenous
        hcm = HierarchicalCausalModel.from_lists(
            observed_subunits=list(obs & subunits),
            unobserved_subunits=list(unobs & subunits),
            observed_units=list(obs & units),
            unobserved_units=list(unobs & units),
            edges=self._graph.edges(nbunch=list(endogenous)),
        )
        return hcm

    def to_hcgm(self: HierarchicalStructuralCausalModel) -> HierarchicalCausalModel:
        """Convert an HSCM to a hierarchical causal graphical model (HCGM) with promoted Q variables."""
        hcm = self.to_hcm()
        return hcm.to_hcgm()

    def to_admg(self, *, return_hcgm: bool = False) -> NxMixedGraph:
        """Return a collapsed hierarchical causal model.

        :param return_hcgm:
            if True, returns the intermediate hierarchical causal
            graphical models (HCGM) with subunits and promoted Q variables
        :returns: a mixed graph
        """
        hcm = self.to_hcm()
        return hcm.to_admg(return_hcgm=return_hcgm)

    def to_pygraphviz(self) -> pygraphviz.AGraph:
        """Get a pygraphviz object."""
        import pygraphviz as pgv

        def _pgv(n: Variable) -> str:
            if isinstance(n, QVariable):
                return n.pgv_str()
            else:
                return n.name

        rv = pgv.AGraph(directed=True)
        for node in self._graph.nodes():
            if node in self.observed:
                rv.add_node(_pgv(node), style="filled", color="lightgrey")
            else:
                rv.add_node(_pgv(node))

        for node in self.nodes():
            if node in self.get_exogenous_noise():
                rv.get_node(_pgv(node)).attr["shape"] = "plaintext"
            else:
                rv.get_node(_pgv(node)).attr["shape"] = "square"

        rv.add_subgraph(
            [_pgv(node) for node in self.subunits],
            name=SUBUNITS_KEY,
            style="dashed",
            label="m",
        )

        for u, v in self._graph.edges():
            rv.add_edge(_pgv(u), _pgv(v))

        return rv


def get_ancestors(subunit_graph: SubunitGraph, start_node: VHint) -> set[Variable]:
    """Perform a depth-first search to get all ancestors of a node in a subunit graph.

    :param subunit_graph: A subunit graph
    :param start_node: the node to start the search from
    :returns: set of all ancestor nodes
    """
    start_node = _upgrade(start_node)
    stack = [start_node]
    ancestors = set()

    while stack:
        node = stack.pop()
        if node in ancestors:
            continue
        ancestors.add(node)
        for predecessor in subunit_graph.predecessors(node):
            if predecessor not in ancestors:
                stack.append(predecessor)

    # Remove the start_node from the visited set if you don't want to include it
    ancestors.remove(start_node)
    return ancestors


@dataclass(frozen=True, order=True, repr=False)
class QVariable(Variable):
    """A variable, extended with a list of parents."""

    parents: frozenset[Variable] = field(default_factory=frozenset)

    def get_lhs(self) -> Variable:
        """Get the left-hand side (i.e., child)."""
        return Variable(self.name)

    def get_all(self) -> frozenset[Variable]:
        """Get the union of the left-hand side and right-hand side."""
        return self.parents.union({self.get_lhs()})

    def pgv_str(self) -> str:
        """Get a string compatible with the V1 implementation."""
        child_name = self.name
        if not self.parents:
            return f"Q_{child_name}"
        parent_str = ",".join(sorted(p.name for p in self.parents))
        return f"Q_{{{child_name}|{parent_str}}}"

    def _iter_variables(self) -> Iterable[Variable]:
        yield self.get_lhs()
        yield from self.parents

    def to_text(self) -> str:
        """Get text."""
        return self.pgv_str()

    def to_y0(self) -> str:
        """Get a string that can be parsed by Y0."""
        p = [p.name for p in self.parents]
        return f"QVariable({self.name}, {p})"

    def to_latex(self) -> str:
        """Get latex for the q-variable."""
        return self.pgv_str()

    @classmethod
    def parse_str(cls, s: str) -> QVariable:
        """Return the subunit variables of the input Q variable, separated by the conditional."""
        if not s.startswith("Q_"):
            raise ValueError(f"Q-variable string should start with `Q_`: {s}")
        if "|" not in s:
            lhs = s[2:]
            if len(lhs) > 1:
                raise ValueError("Invalid format for input Q variable")
            if not lhs:
                raise ValueError(f"Invalid q-variable string: {s}\n\nMissing left-hand side")
            return cls(name=lhs)
        var_str = s[3:-1]
        parse1 = var_str.split("|")
        if len(parse1) != 2:
            raise ValueError("Invalid format for input Q variable")
        lhs = parse1[0]
        rhs = parse1[1].split(",")
        if len(lhs) > 1:
            raise ValueError("Invalid format for input Q variable")
        if not lhs:
            raise ValueError(f"Invalid q-variable string: {s}\n\nMissing left-hand side")
        return cls(name=lhs, parents=frozenset(Variable(p) for p in rhs))


def _create_qvar(subunit_graph: SubunitGraph, subunit_node: Variable) -> QVariable:
    """Return a y0 Variable for the unit-level Q variable of the given subunit variable in the HCM."""
    return QVariable(
        name=subunit_node.name, parents=frozenset(subunit_graph.predecessors(subunit_node))
    )


def _str_or_q(augmentation_variable: str | QVariable) -> QVariable:
    if isinstance(augmentation_variable, str):
        return QVariable.parse_str(augmentation_variable)
    return augmentation_variable


def augment_from_mechanism(
    collapsed: NxMixedGraph, aug: str | QVariable, mechanism: Iterable[QVariable]
) -> NxMixedGraph:
    """Augment a collapsed model with a given augmentation variable and its mechanism.

    :param collapsed: NxMixedGraph of the input collapsed model
    :param aug: new variable to add into the collapsed model
    :param mechanism: collection of variables in the collapsed model that determine the augmentation_variable
    :raises TypeError: if any of the parts of the mechanism aren't q-variables
    :raises ValueError: input mechanism variables must be contained in the collapsed model
    :returns: NxMixedGraph of the augmented model
    """
    aug = _str_or_q(aug)
    augmented = collapsed.copy()
    mechanism = set(mechanism)
    if any(not isinstance(m, QVariable) for m in mechanism):
        raise TypeError("all variables in mechanism need to be QVariables")
    if not mechanism <= collapsed.nodes():
        raise ValueError("The input mechanism must be contained in the collapsed model.")
    augmented.add_node(aug)
    # augmented.deterministic.add(aug)
    for var in mechanism:
        augmented.add_directed_edge(var, aug)
    for var in set(augmented.nodes()) - {aug}:
        parents = set(augmented.directed.predecessors(var))
        if mechanism <= parents:
            augmented.add_directed_edge(aug, var)
            for parent in mechanism:
                augmented.directed.remove_edge(parent, var)
    return augmented


def augmentation_mechanism(
    subunit_graph: SubunitGraph, augmentation_variable: str | QVariable
) -> list[QVariable]:
    """Generate augmentation mechanism."""
    augmentation_variable = _str_or_q(augmentation_variable)
    nodes = set(subunit_graph.nodes())
    lhs_var = augmentation_variable.get_lhs()
    if lhs_var not in nodes:
        raise KeyError(
            f"Augmentation variable's left hand side {lhs_var} is not in subunit graph: {nodes}"
        )
    if not augmentation_variable.parents.issubset(nodes):
        raise KeyError(
            f"Augmentation variable's right hand side {augmentation_variable.parents} are not all not in subunit graph: {nodes}"
        )
    rhs = augmentation_variable.parents
    mechanism = [_create_qvar(subunit_graph, lhs_var)]
    direct_subunit_descendants = get_ancestors(subunit_graph, lhs_var).difference(rhs)
    for dsd in direct_subunit_descendants:
        mechanism.append(_create_qvar(subunit_graph, dsd))
    return mechanism


def collapse_hcm(model: HierarchicalCausalModel, return_hcgm: bool = False) -> NxMixedGraph:
    """Collapse the given hierarchical model according to Algorithm 1 of the HCM paper."""
    return model.to_admg(return_hcgm=return_hcgm)  # TODO handle input HSCM class as well?


def augment_collapsed_model(
    model: NxMixedGraph,
    subunit_graph: SubunitGraph,
    augmentation_variable: QVariable | str,
    mechanism: Iterable[QVariable] | None = None,
) -> NxMixedGraph:
    """Augment given variable into the given collapsed model."""
    # TODO test
    augmentation_variable = _str_or_q(augmentation_variable)
    if mechanism is None:
        mechanism = augmentation_mechanism(subunit_graph, augmentation_variable)
    augmented = augment_from_mechanism(model, augmentation_variable, mechanism)
    return augmented


def marginalize_augmented_model(
    augmented: NxMixedGraph,
    augmentation_variable: str | QVariable,
    marginal_parents: Collection[QVariable],
) -> NxMixedGraph:
    """Marginalize out a given collection of variables from an augmented model.

    :param augmented: NxMixedGraph of the input augmented model
    :param augmentation_variable: the variable that was previously augmented into the model
    :param marginal_parents: collection of parents of the augmentation variable to be marginalized out.
    :raises ValueError: augmentation_variable must be in the augmented model
    :raises ValueError: marginal_parents cannot be all the parents of augmentation_variable
    :raises ValueError: augmentation_variable must be the only child of the each marginal parent
    :returns: NxMixedGraph of the marginalized model
    """
    augmentation_variable = _str_or_q(augmentation_variable)
    marginalized = augmented.copy()
    check_set = {augmentation_variable}
    mechanism = set(augmented.directed.predecessors(augmentation_variable))
    if augmentation_variable not in augmented.nodes():
        raise ValueError("Augmentation variable must be in the input augmented model.")
    if set(marginal_parents) == mechanism:
        raise ValueError("Cannot marginalize all parents of the augmentation variable.")
    for parent in marginal_parents:
        if set(marginalized.directed.successors(parent)) != check_set:
            raise ValueError(
                "The augmentation variable must be the only child of the marginalized parents."
            )
        directed_grandparents = marginalized.directed.predecessors(parent)
        for gp in directed_grandparents:
            marginalized.add_directed_edge(gp, augmentation_variable)
        marginalized.directed.remove_node(parent)
        undirected_grandparents = marginalized.undirected.neighbors(parent)
        for gp in undirected_grandparents:
            marginalized.add_undirected_edge(gp, augmentation_variable)
        marginalized.undirected.remove_node(parent)
    return marginalized
