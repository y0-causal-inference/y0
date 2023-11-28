"""Implementation of counterfactual transportability."""

from typing import Optional, Union

from y0.dsl import CounterfactualVariable, Intervention, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "simplify",
]


def simplify(
    event: list[tuple[CounterfactualVariable, Intervention]]
) -> Optional[dict[CounterfactualVariable, Intervention]]:
    """Run algorithm 1: the SIMPLIFY algorithm from Correa, Lee, and Bareinboim 2022"""
    raise NotImplementedError

def get_ancestors_of_counterfactual(
    event: tuple[CounterfactualVariable, NxMixedGraph]
) -> set(Union[CounterfactualVariable, Variable]):
    """Gets the ancestors of a counterfactual variable following Correa, Lee, and Bareinboim 2022, Definition 2.1 and Example 2.1
       and returns them as a set of variables that may include counterfactual variables."""
    raise NotImplementedError(f"Unimplemented function: get_ancestors_of_counterfactual")    

## TODO: Add expected inputs and outputs to the below three algorithms
def sigmaTR() -> None:
    """Implements the sigma-TR algorithm from Correa, Lee, and Bareinboim 2022 (Algorithm 4 in Appendix B)"""
    raise NotImplementedError(f"Unimplemented function: sigmaTR")

def ctfTR() -> None:
    """Implements the ctfTR algorithm from Correa, Lee, and Bareinboim 2022 (Algorithm 3)"""
    raise NotImplementedError(f"Unimplemented function: ctfTR")

def ctfTRu() -> None:
    """Implements the ctfTRu algorithm from Correa, Lee, and Bareinboim 2022 (Algorithm 2)"""
    raise NotImplementedError(f"Unimplemented function: ctfTRu")


