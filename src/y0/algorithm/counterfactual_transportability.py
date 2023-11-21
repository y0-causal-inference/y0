"""Implementation of counterfactual transportability."""

from typing import Optional

from y0.dsl import CounterfactualVariable, Intervention

__all__ = [
    "simplify",
]


def simplify(
    event: list[tuple[CounterfactualVariable, Intervention]]
) -> Optional[dict[CounterfactualVariable, Intervention]]:
    """Run algorithm 1."""
    raise NotImplementedError
