# -*- coding: utf-8 -*-

"""A simulation sort of based on Sara's idea."""

from collections import Mapping, defaultdict

import pandas as pd
from numpy.random import normal, uniform
from scipy.special import expit
from tqdm.auto import trange

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


class Simulation:
    """A data structure for a simulation."""

    def __init__(self, graph: NxMixedGraph, use_weights: bool = False, nonlinearity: bool = False) -> None:
        """Prepare a simulation by calculating source

        :param graph: The ADMG
        :param use_weights: Should edge weights be randomly distributed using a uniform(0, 1)?
            If false, all edge weights are assigned 1.0.
        """
        self.graph = graph
        self.weights = {
            edge: uniform(low=0, high=1) if use_weights else 1.0
            for edge in self.graph.directed.edges()
        }
        self.sources = {
            node
            for node in self.graph.nodes() if not self.graph.directed.in_edges(node)
        }
        if nonlinearity:
            self.activation = expit
        else:
            self.activation = _identity

    def trial(self, scale: float = 1.0) -> Mapping[str, float]:
        """Perform a single trial.

        :param scale: The scale parameter of the gaussians used
        :return: A mapping from variable names to simulation values
        """
        # there have to be at least one source since it's a DAG
        start = self.sources
        # assign random values to all sources
        values = {
            node: normal(loc=0.0, scale=scale)
            for node in start
        }

        # TODO this could be implemented iteratively if using a topological sort
        while start:
            downstream = {
                (u, v): normal(loc=self.weights[u, v] * values[u], scale=scale)
                for u, v in self.graph.directed.out_edges(start)
            }
            agg = defaultdict(set)
            for u, v in downstream:
                agg[v].add(downstream[u, v])
            for v, v_values in agg.items():
                assert v not in values
                values[v] = self.activation(sum(v_values))
            start = set(agg.keys())

        return values
