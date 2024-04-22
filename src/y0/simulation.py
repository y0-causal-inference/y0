# -*- coding: utf-8 -*-

"""Generate data using a linear structural causal model."""

from __future__ import annotations

import itertools as itt
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Mapping,
    MutableMapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
)

import pandas as pd
from numpy.random import normal, uniform
from sklearn.linear_model import LinearRegression
from tqdm.auto import trange

from .algorithm.conditional_independencies import get_conditional_independencies
from .algorithm.estimation.linear_scm import get_single_door
from .dsl import V1, V2, V3, V4, V5, V6, Variable
from .graph import NxMixedGraph
from .struct import DSeparationJudgement

__all__ = [
    "LinearSCM",
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
    d_separation: Optional[DSeparationJudgement]


FitsDict = Dict[FrozenSet[Variable], FitTuple]


def simulate(
    graph: NxMixedGraph,
    trials: int = 200,
    return_fits: bool = True,
    progress: bool = False,
    tqdm_kwargs: Optional[Mapping[str, Any]] = None,
    **kwargs,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, FitsDict]]:
    """Simulate a graph using a linear structural causal model."""
    judgements = get_conditional_independencies(graph)
    cis: Mapping[FrozenSet[Variable], DSeparationJudgement] = {
        frozenset((judgement.left, judgement.right)): judgement for judgement in judgements
    }

    _tqdm_kwargs = dict(desc="Simulation", unit="trial", unit_scale=True)
    if tqdm_kwargs:
        _tqdm_kwargs.update(tqdm_kwargs)

    linear_scm = LinearSCM(graph, **kwargs)
    results = {
        trial: {variable.name: values for variable, values in linear_scm.trial().items()}
        for trial in trange(trials, disable=not progress, **_tqdm_kwargs)
    }
    rv = pd.DataFrame(results).T

    # Get a prioritized ordering for edges actually in graph
    order = list(graph.directed.edges())
    _x = {frozenset(edge) for edge in order}
    order.extend(
        (left, right)
        for left, right in itt.combinations(sorted(graph.nodes(), key=lambda n: n.name), 2)
        if frozenset((left, right)) not in _x
    )

    if not return_fits:
        return rv

    fits: FitsDict = {}
    for parent, child in order:
        x, y = rv[parent.name].to_numpy().reshape(-1, 1), rv[child.name]
        regression = LinearRegression()
        regression.fit(x, y)
        r2 = regression.score(x, y)
        key = frozenset((parent, child))
        fits[key] = FitTuple(
            regression=regression,
            slope=regression.coef_[0],
            intercept=regression.intercept_,
            r2=r2,
            d_separation=cis.get(key),
        )

    return rv, fits


Generator = Callable[[], float]


def get_fits_df(fits_dict: FitsDict) -> pd.DataFrame:
    """Convert a fits dictionary into a pandas dataframe."""
    rows = [
        (
            a.name,
            b.name,
            fits_tuple.slope,
            fits_tuple.intercept,
            fits_tuple.r2,
            # t.edge,
            False if fits_tuple.d_separation is None else fits_tuple.d_separation.separated,
            None if fits_tuple.d_separation is None else fits_tuple.d_separation.conditions,
        )
        for (a, b), fits_tuple in fits_dict.items()
    ]
    df = pd.DataFrame(
        rows, columns=["parent", "child", "slope", "intercept", "r2", "d_separated", "d_sep_cond"]
    )
    return df


class LinearSCM:
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
        *,
        generators: Optional[Mapping[Variable, Generator]] = None,
        data: Optional[pd.DataFrame] = None,
        weights: Optional[Mapping[Tuple[Variable, Variable], float]] = None,
    ) -> None:
        """Prepare a simulation.

        :param graph: The ADMG
        :param generators: Generator functions for each node. If none given, defaults to uniformly
            distributed between -1.0 and 1.0.
        :param data: If given, estimates weights.
        :param weights: Weights for each directed edge. If none given, defaults to uniformly distributed
            weights between -1.0 and 1.0.
        :raises ValueError: if the generators or weights are given and don't match the graph
        """
        self.graph = graph

        if data is not None:
            self.weights = get_single_door(graph, data)
        elif weights is None:
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

    def fix(self, values: Mapping[Variable, float]) -> LinearSCM:
        """Create a new simulation with the given nodes fixed to their values."""

        def _get(node: Variable) -> float:
            return values[node]

        generators = cast(
            Mapping[Variable, Generator],
            {
                node: generator if node not in values else partial(_get, node=node)
                for node, generator in self.generators.items()
            },
        )
        return LinearSCM(graph=self.graph, generators=generators, weights=self.weights)

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
