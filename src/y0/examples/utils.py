"""Utilities for generating examples."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from y0.algorithm.identify import Identification, Query
from y0.dsl import Variable
from y0.graph import NxMixedGraph
from y0.struct import DSeparationJudgement, VermaConstraint

__all__ = [
    "Example",
]


@dataclass
class Example:
    """An example graph packaged with certain pre-calculated data structures."""

    name: str
    reference: str
    graph: NxMixedGraph
    description: str | None = None
    verma_constraints: Sequence[VermaConstraint] | None = None
    conditional_independencies: Sequence[DSeparationJudgement] | None = None
    data: pd.DataFrame | None = None
    identifications: list[dict[str, list[Identification]]] | None = None
    #: Example queries are just to give an idea to a new user
    #: what might be interesting to use in the ID algorithm
    example_queries: list[Query] | None = None
    generate_data: Callable[[int, dict[Variable, float] | None], pd.DataFrame] | None = None

    def generate_ate(
        self,
        *,
        num_samples: int,
        treatment: Variable,
        outcome: Variable,
        treatment_0: float = 0.0,
        treatment_1: float = 1.0,
        **kwargs: Any,
    ) -> float:
        """Calculate the ATE for a single treatment/outcome pair."""
        if self.generate_data is None:
            raise TypeError(f"no generation method provided in example: {self.name}")

        data_1 = self.generate_data(num_samples, {treatment: treatment_1}, **kwargs)
        data_0 = self.generate_data(num_samples, {treatment: treatment_0}, **kwargs)
        return cast(float, data_1.mean()[outcome.name] - data_0.mean()[outcome.name])
