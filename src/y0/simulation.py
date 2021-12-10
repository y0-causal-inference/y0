# -*- coding: utf-8 -*-

"""A simulation sort of based on Sara's idea."""

from __future__ import annotations

from collections import Mapping
from functools import partial
from typing import Callable, MutableMapping, Optional, Tuple

import pandas as pd
from numpy.random import uniform
from tqdm.auto import trange

from .dsl import Variable
from .graph import NxMixedGraph

__all__ = [
    "Simulation",
    "simulate",
]


def simulate(graph: NxMixedGraph, trials: int = 600) -> pd.DataFrame:
    """Simulate a graph using gaussians for all variables."""
    simulation = Simulation(graph)
    trials = {
        n: simulation.trial()
        for n in trange(trials, desc="Simulation", unit="trial", unit_scale=True)
    }
    rv = pd.DataFrame(trials).T
    return rv


class Simulation:
    """A data structure for a simulation.

    Provides an implementation of the simulation presented in Figure 1 and Example 2.2 in
    `Graphical criteria for efficient total effect estimation via adjustment in causal
    linear models <https://arxiv.org/abs/1907.02435>`_
    """

    #: The generator functions for all nodes
    generators: Mapping[Variable, Callable[[], float]]

    #: Weights corresponding to each edge
    weights: Mapping[Tuple[Variable, Variable], float]

    def __init__(
        self,
        graph: NxMixedGraph,
        generators: Optional[Mapping[Variable, Callable[[], float]]] = None,
        weights: Optional[Mapping[Tuple[Variable, Variable], float]] = None,
    ) -> None:
        """Prepare a simulation.

        :param graph: The ADMG
        :param generators: Generator functions for each node. If none given, defaults to uniformly
            distributed between -1.0 and 1.0.
        :param weights: Weights for each directed edge. If none given, defaults to uniformly distributed
            weights between -1.0 and 1.0.
        :raises ValueError: if the generators or weights are given and don't match the graph
        """
        self.graph = graph

        if weights is None:
            self.weights = {edge: uniform(low=-1.0, high=1.0) for edge in graph.directed.edges()}
        elif set(weights) != set(self.graph.directed.edges()):
            raise ValueError("given weights do not exactly match directed edges in the graph")
        else:
            self.weights = weights

        if generators is None:
            self.generators = {
                node: partial(uniform, low=-1.0, high=1.0) for node in self.graph.nodes()
            }
        elif set(generators) != set(self.graph.nodes()):
            raise ValueError("given node generators do not exactly match nodes in the graph")
        else:
            self.generators = generators

    def generate(self, node: Variable) -> float:
        """Generate a value for the variable."""
        return self.generators[node]()

    def fix(self, values: Mapping[Variable, float]) -> Simulation:
        """Create a new simulation with the given nodes fixed to their values."""
        generators = {
            node: lambda: values[node] if node in values else generator
            for node, generator in self.generators.items()
        }
        return Simulation(graph=self.graph, generators=generators, weights=self.weights)

    def trial(self) -> Mapping[Variable, float]:
        """Perform a single trial.

        :return: A mapping from variable names to simulation values
        """
        values: MutableMapping[Variable, float] = {}
        for node in self.graph.topological_sort():
            values[node] = self.generate(node) + sum(
                self.weights[parent, node] * values[parent]
                for parent in self.graph.directed.out_edges(node)
            )
        return values
