"""Data-driven graph repair."""

from y0.graph import NxMixedGraph
import pandas as pd
from y0.algorithm.conditional_independencies import get_conditional_independencies
from y0.algorithm.falsification import get_graph_falsifications

__all__ = [
    "repair_graph",
]


def repair_graph(graph: NxMixedGraph, data: pd.DataFrame) -> NxMixedGraph:
    """Repair the graph based on calculatable conditional dependencies in the data."""
    falsifications = get_graph_falsifications(graph=graph, df=data)
    d_separations = get_conditional_independencies(graph)
    raise NotImplementedError
