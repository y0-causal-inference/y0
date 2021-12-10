# -*- coding: utf-8 -*-

"""A simulation sort of based on Sara's idea."""

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


def _identity(x):
    return x


def _constant_weights(graph: NxMixedGraph) -> Mapping[Tuple[Variable, Variable], float]:
    return {edge: 1.0 for edge in graph.directed.edges()}


def _uniform_random_weights(
    graph: NxMixedGraph, low: float = -1.0, high: float = 1.0
) -> Mapping[Tuple[Variable, Variable], float]:
    return {edge: uniform(low=low, high=high) for edge in graph.directed.edges()}


class Simulation:
    """A data structure for a simulation."""

    generators: MutableMapping[Variable, Callable[[], float]]

    def __init__(
        self,
        graph: NxMixedGraph,
        generators: Optional[Mapping[Variable, Callable[[], float]]] = None,
        weights: Optional[Mapping[Tuple[Variable, Variable], float]] = None,
        default_weight_function: bool = True,
    ) -> None:
        """Prepare a simulation by calculating source

        :param graph: The ADMG
        :param generators: Generator functions for each node. If none given, defaults to uniformly
            distributed between -1 and 1.
        :param default_weight_function: Should edge weights be randomly distributed using a uniform(0, 1)?
            If false, all edge weights are assigned 1.0.
        """
        self.graph = graph

        if weights is None:
            self.weights = _uniform_random_weights(graph) if default_weight_function else _constant_weights(graph)

        self.generators = generators or {}
        for node in self.graph.nodes():
            if node not in self.generators:
                self.generators[node] = partial(uniform, low=-1.0, high=1.0)

    def generate(self, node: Variable) -> float:
        """Generate a value for the variable."""
        return self.generators[node]()

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
