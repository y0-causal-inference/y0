# -*- coding: utf-8 -*-

"""A simulation sort of based on Sara's idea."""

from __future__ import annotations

from collections import Mapping
from functools import partial
from typing import Callable, MutableMapping, NamedTuple, Optional, Tuple, cast

import pandas as pd
from numpy.random import normal, uniform
from sklearn.linear_model import LinearRegression
from tqdm.auto import trange

from .dsl import V1, V2, V3, V4, V5, V6, Variable
from .graph import NxMixedGraph

__all__ = [
    "Simulation",
    "simulate",
]

example_graph = NxMixedGraph.from_edges(
    directed=[
        (V1, V2),
        (V1, V4),
        (V2, V5),
        (V4, V5),
        (V4, V6),
        (V3, V5),
        (V5, V6),
    ],
)
example_generators = {
    V1: partial(uniform, low=-1.0, high=1.0),
    V2: partial(uniform, low=-2.0, high=2.0),
    V3: partial(normal, loc=0.0, scale=1.0),
    V4: partial(normal, loc=0.0, scale=2.0),
    V5: partial(uniform, low=-3.0, high=3.0),
    V6: partial(normal, loc=0.0, scale=3.0),
}


class FitTuple(NamedTuple):
    """A tuple representing the linear regression fit."""

    regression: LinearRegression
    slope: float
    intercept: float
    r2: float


def simulate(
    graph: NxMixedGraph, trials: int = 200, **kwargs
) -> Tuple[pd.DataFrame, Mapping[Tuple[Variable, Variable], FitTuple]]:
    """Simulate a graph using gaussians for all variables."""
    simulation = Simulation(graph, **kwargs)
    results = {
        n: simulation.trial()
        for n in trange(trials, desc="Simulation", unit="trial", unit_scale=True)
    }
    rv = pd.DataFrame(results).T

    fits = {}
    for parent, child in graph.directed.edges():
        x, y = rv[parent].to_numpy().reshape(-1, 1), rv[child]
        reg = LinearRegression()
        reg.fit(x, y)
        score = reg.score(x, y)
        fits[parent, child] = FitTuple(reg, reg.coef_[0], reg.intercept_, score)

    return rv, fits


Generator = Callable[[], float]


class Simulation:
    """A data structure for a simulation.

    Provides an implementation of the simulation presented in Figure 1 and Example 2.2 in
    `Graphical criteria for efficient total effect estimation via adjustment in causal
    linear models <https://arxiv.org/abs/1907.02435>`_
    """

    #: The generator functions for all nodes
    generators: Mapping[Variable, Generator]

    #: Weights corresponding to each edge
    weights: Mapping[Tuple[Variable, Variable], float]

    def __init__(
        self,
        graph: NxMixedGraph,
        generators: Optional[Mapping[Variable, Generator]] = None,
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
            self.weights = {edge: uniform(low=0.0, high=1.0) for edge in graph.directed.edges()}
        elif set(weights) != set(self.graph.directed.edges()):
            raise ValueError("given weights do not exactly match directed edges in the graph")
        else:
            self.weights = weights

        if generators is None:
            self.generators = cast(
                Mapping[Variable, Generator],
                {node: partial(uniform, low=-1.0, high=1.0) for node in self.graph.nodes()},
            )
        elif set(generators) != set(self.graph.nodes()):
            raise ValueError("given node generators do not exactly match nodes in the graph")
        else:
            self.generators = generators

    def generate(self, node: Variable) -> float:
        """Generate a value for the variable."""
        return self.generators[node]()

    def fix(self, values: Mapping[Variable, float]) -> Simulation:
        """Create a new simulation with the given nodes fixed to their values."""
        generators = cast(
            Mapping[Variable, Generator],
            {
                node: lambda: values[node] if node in values else generator
                for node, generator in self.generators.items()
            },
        )
        return Simulation(graph=self.graph, generators=generators, weights=self.weights)

    def trial(self) -> Mapping[Variable, float]:
        """Perform a single trial.

        :return: A mapping from variable names to simulation values
        """
        values: MutableMapping[Variable, float] = {}
        for node in self.graph.topological_sort():
            values[node] = self.generate(node) + sum(
                self.weights[parent, node] * values[parent]
                for parent, _ in self.graph.directed.in_edges(node)
            )
        return values
