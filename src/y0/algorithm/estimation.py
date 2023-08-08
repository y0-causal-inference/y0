"""
for A-fixable, probably "Efficient Generalized AIPW" would be the one to focus on first.
For P-fixable, probably, "Efficient APIPW" would be the one to focus on first. (edited)
The other algorithms are inferior, and are there just to demonstrate that
"Efficient Generalized AIPW" and "Efficient APIPW" is better.

There is also a simple algorithm that can tell you whether a query is A-fixable
or P-fixable, so you know which algorithm to use.
"""

from typing import List, Literal, Optional, Union

from y0.dsl import CounterfactualVariable, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "estimate_causal_effect",
    "is_a_fixable",
    "aipw",
    "is_p_fixable",
    "apipw",
]


def estimate_causal_effect(
    graph: NxMixedGraph,
    treatment: Variable,
    outcome: Variable,
    data,
    *,
    query_type: Literal["ate", "expectation", "probability"],
    conditions: Optional[List[Variable]] = None,
) -> float:
    if query_type == "ate":
        return estimate_ate(
            graph=graph, treatment=treatment, outcome=outcome, data=data, conditions=conditions
        )
    elif query_type == "expectation":
        raise NotImplementedError
    elif query_type == "probability":
        raise NotImplementedError
    else:
        raise TypeError


def estimate_ate(
    graph: NxMixedGraph,
    treatment: Union[Variable, List[Variable]],
    outcome: Union[Variable, List[Variable]],
    data: ...,
    *,
    conditions: Optional[List[Variable]] = None,
) -> float:
    """Estimate the average treatment effect."""
    if conditions is not None:
        raise NotImplementedError("can not yet handle conditional queries")
    if isinstance(treatment, list) or isinstance(outcome, list):
        raise NotImplementedError("can not yet handle multiple treatments nor outcomes")
    if isinstance(treatment, CounterfactualVariable) or isinstance(outcome, CounterfactualVariable):
        raise NotImplementedError("can not yet handle counterfactual treatments nor outcomes")

    ananke_graph = graph.to_admg()
    from ananke.estimation import CausalEffect

    causal_effect = CausalEffect(ananke_graph, treatment.name, outcome.name)
    return causal_effect.compute_effect(data, "eff-aipw")


def is_a_fixable() -> bool:
    pass


def aipw():
    pass


def is_p_fixable() -> bool:
    pass


def apipw():
    pass
