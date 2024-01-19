"""Utilities for generating examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

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
    description: Optional[str] = None
    verma_constraints: Optional[Sequence[VermaConstraint]] = None
    conditional_independencies: Optional[Sequence[DSeparationJudgement]] = None
    data: Optional[pd.DataFrame] = None
    identifications: Optional[list[dict[str, list[Identification]]]] = None
    #: Example queries are just to give an idea to a new user
    #: what might be interesting to use in the ID algorithm
    example_queries: Optional[list[Query]] = None
    generate_data: Optional[Callable[[int, Optional[dict[Variable, float]]], pd.DataFrame]] = None

    def generate_ate(
        self,
        *,
        num_samples: int,
        treatment: Variable,
        outcome: Variable,
        treatment_0: float = 0.0,
        treatment_1: float = 1.0,
        **kwargs,
    ) -> float:
        """Calculate the ATE for a single treatment/outcome pair."""
        if self.generate_data is None:
            raise TypeError(f"no generation method provided in example: {self.name}")

        data_1 = self.generate_data(num_samples, {treatment: treatment_1}, **kwargs)
        data_0 = self.generate_data(num_samples, {treatment: treatment_0}, **kwargs)
        return data_1.mean()[outcome.name] - data_0.mean()[outcome.name]
