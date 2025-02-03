"""An example directory of hierarchical causal models."""

from y0.dsl import A, B, C, D, U, Y, Z
from y0.graph import NxMixedGraph
from y0.hierarchical import HierarchicalCausalModel, QVariable

__all__ = [
    "confounder_hcm",
]

Q_A = QVariable(name="A")
Q_Y = QVariable(name="Y")
Q_Y_A = QVariable(name="Y", parents=frozenset([A]))
Q_A_Z = QVariable(name="A", parents=frozenset([Z]))
Q_Z = QVariable(name="Z")
Q_Y_A_Z = QVariable(name="Y", parents=frozenset([A, Z]))


def _make_confounder_hcm() -> HierarchicalCausalModel:
    """Pytest fixture for the Confounder HCM in Figure 2 (a)."""
    graph = HierarchicalCausalModel()
    graph.add_observed_node(A)
    graph.add_observed_node(Y)
    graph.add_edge(U, A)
    graph.add_edge(A, Y)
    graph.add_edge(U, Y)
    graph.add_subunits([A, Y])
    return graph


confounder_hcm = _make_confounder_hcm()


def get_confounder_hcgm() -> HierarchicalCausalModel:
    """Pytest fixture for the Confounder HCGM in Figure 2 (b)."""
    hcm = HierarchicalCausalModel()
    hcm.add_observed_node(A)
    hcm.add_observed_node(Y)
    hcm.add_observed_node(Q_A)
    hcm.add_observed_node(Q_Y_A)
    hcm.add_edge(U, Q_A)
    hcm.add_edge(Q_A, A)
    hcm.add_edge(A, Y)
    hcm.add_edge(U, Q_Y_A)
    hcm.add_edge(Q_Y_A, Y)
    hcm.add_subunits([A, Y])
    return hcm


def get_confounder_interference_hcm() -> HierarchicalCausalModel:
    """Pytest fixture for the Confounder Interference HCM in Figure 2 (e)."""
    hcm = HierarchicalCausalModel()
    hcm.add_observed_node(A)
    hcm.add_observed_node(Y)
    hcm.add_observed_node(Z)
    hcm.add_edge(U, A)
    hcm.add_edge(A, Y)
    hcm.add_edge(U, Y)
    hcm.add_edge(A, Z)
    hcm.add_edge(Z, Y)
    hcm.add_subunits([A, Y])
    return hcm


def get_confounder_interference_hcgm() -> HierarchicalCausalModel:
    """Pytest fixture for the Confounder Interference HCGM in FIgure 2 (f)."""
    hcm = HierarchicalCausalModel()
    hcm.add_observed_node(A)
    hcm.add_observed_node(Y)
    hcm.add_observed_node(Z)
    hcm.add_observed_node(Q_A)
    hcm.add_observed_node(Q_Y_A)
    hcm.add_edge(U, Q_A)
    hcm.add_edge(Q_A, A)
    hcm.add_edge(A, Y)
    hcm.add_edge(U, Q_Y_A)
    hcm.add_edge(Q_Y_A, Y)
    hcm.add_edge(A, Z)
    hcm.add_edge(Z, Q_Y_A)
    hcm.add_subunits([A, Y])
    return hcm


def get_instrument_hcm() -> HierarchicalCausalModel:
    """Pytest fixture for the Instrument HCM in Figure 2 (i)."""
    hcm = HierarchicalCausalModel()
    hcm.add_observed_node(A)
    hcm.add_observed_node(Y)
    hcm.add_observed_node(Z)
    hcm.add_edge(U, A)
    hcm.add_edge(A, Y)
    hcm.add_edge(U, Y)
    hcm.add_edge(Z, A)
    hcm.add_subunits([A, Z])
    return hcm


def get_instrument_subunit_graph() -> HierarchicalCausalModel:
    """Pytest fixture for the Instrument HCM subunit graph."""
    subg = HierarchicalCausalModel()
    subg.add_observed_node(A)
    subg.add_observed_node(Z)
    subg.add_edge(Z, A)
    return subg


def get_instrument_hcgm() -> HierarchicalCausalModel:
    """Pytest fixture for the Instrument HCGM in Figure 2(j)."""
    hcm = HierarchicalCausalModel()
    hcm.add_observed_node(A)
    hcm.add_observed_node(Y)
    hcm.add_observed_node(Z)
    hcm.add_observed_node(Q_Z)
    hcm.add_observed_node(Q_A_Z)
    hcm.add_edge(U, Q_A_Z)
    hcm.add_edge(Q_A_Z, A)
    hcm.add_edge(Q_Z, Z)
    hcm.add_edge(A, Y)
    hcm.add_edge(U, Y)
    hcm.add_edge(Z, A)
    hcm.add_subunits([A, Z])
    return hcm


def get_compl_subgraph_hcm() -> HierarchicalCausalModel:
    """Pytest fixture for HCM with complicated subgraph structure."""
    hcm = HierarchicalCausalModel.from_lists(
        observed_subunits=[A, B, C, Y],
        observed_units=["D"],
        unobserved_units=[U],
        edges=[
            (U, A),
            (U, B),
            (U, C),
            (A, B),
            (B, Y),
            (C, Y),
            (A, D),
            (D, C),
        ],
    )
    return hcm


confounder_augmented_admg = NxMixedGraph.from_edges(
    undirected=[(Q_A, Q_Y_A)], directed=[(Q_A, Q_Y), (Q_Y_A, Q_Y)]
)
"""Pytest fixture for augmented Confounder HCM in Figure 2 (d)."""


confounder_interference_augmented_admg = NxMixedGraph.from_edges(
    undirected=[(Q_A, Q_Y_A)], directed=[(Q_A, Q_Y), (Q_Y_A, Q_Y), (Q_A, Z), (Z, Q_Y_A)]
)
"""Pytest fixture for augmented Confounder Interference HCM in Figure 2 (h)."""


instrument_augmented_admg = NxMixedGraph.from_edges(
    undirected=[(Q_A_Z, Y)], directed=[(Q_A_Z, Q_A), (Q_Z, Q_A), (Q_A, Y)]
)
"""Pytest fixture for augmented Instrument HCM in Figure A2."""


instrument_marginalized_admg = NxMixedGraph.from_edges(
    undirected=[(Q_A_Z, Y)], directed=[(Q_A_Z, Q_A), (Q_A, Y)]
)
"""Pytest fixture for augmented Instrument HCM in Figure 2 (l)."""


confounder_collapsed_admg = NxMixedGraph.from_edges(undirected=[(Q_A, Q_Y_A)])
"""Pytest fixture for collapsed Confounder HCM in Figure 2 (c)."""


confounder_interference_collapsed_admg = NxMixedGraph.from_edges(
    undirected=[(Q_A, Q_Y_A)], directed=[(Q_A, Z), (Z, Q_Y_A)]
)
"""Pytest fixture for collapsed Confounder Interference HCM in Figure 2 (g)."""


instrument_collapsed_admg = NxMixedGraph.from_edges(
    undirected=[(Q_A_Z, Y)], directed=[(Q_A_Z, Y), (Q_Z, Y)]
)
"""Pytest fixture for collapsed Instrument HCM in Figure 2 (k)."""
