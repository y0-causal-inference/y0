"""Do Calculus."""

from typing import Annotated, Literal

from .conditional_independencies import are_d_separated
from .separation import are_sigma_separated
from ..dsl import Variable
from ..graph import NxMixedGraph
from ..util import InPaperAs

__all__ = [
    "rule_2_of_do_calculus_applies",
]


# TODO implement rule 1 condition
# TODO implement rule 3 condition


def rule_2_of_do_calculus_applies(
    graph: Annotated[NxMixedGraph, InPaperAs("G")],
    *,
    treatments: Annotated[set[Variable], InPaperAs(r"\mathbf{X}")],
    outcomes: Annotated[set[Variable], InPaperAs(r"\mathbf{Y}")],
    conditions: Annotated[set[Variable], InPaperAs(r"\mathbf{Z}")],
    condition: Variable,
    separation_implementation: Literal["d", "sigma"] | None = None,
) -> bool:
    r"""Check if Rule 2 of the Do-Calculus applies to the conditioned variable.

    :param graph: The graph
    :param treatments: The treatments
    :param conditions: The conditions
    :param outcomes: The outcomes
    :param condition: The condition to check
    :param separation_implementation: The separation implementation. Defaults to d
        separation, but can be generalized to sigma separation

    :returns: If rule 2 applies, see below.

    If Rule 2 of the do calculus applies to the conditioned variable, then it can be
    converted to a do variable.

    .. math::

        \newcommand\ci{\perp\!\!\!\perp}
        \newcommand{\ubar}[1]{\underset{\bar{}}{#1}}
        \newcommand{\obar}[1]{\overset{\bar{}}{#1}}
        \text{if } (\exists Z \in \mathbf{Z})(\mathbf{Y} \ci Z | \mathbf{X}, \mathbf{Z}
        - \{Z\})_{G_{\bar{\mathbf{X}}\ubar{Z}}} \\
        \text{then } P(\mathbf{Y}|do(\mathbf{X}),\mathbf{Z}) = P(\mathbf Y|do(\mathbf X), do(Z), \mathbf{Z} - \{Z\})
    """
    reduced_conditions = treatments | (conditions - {condition})
    reduced_graph: Annotated[NxMixedGraph, InPaperAs(r"G_{\bar{x}, \underbar{z}}")] = (
        graph.remove_in_edges(treatments).remove_out_edges(condition)
    )
    if separation_implementation == "d" or separation_implementation is None:
        return all(
            are_d_separated(reduced_graph, outcome, condition, conditions=reduced_conditions)
            for outcome in outcomes
        )
    elif separation_implementation == "sigma":
        return all(
            are_sigma_separated(reduced_graph, outcome, condition, conditions=reduced_conditions)
            for outcome in outcomes
        )
    else:
        raise ValueError(f"Unknown separation implementation: {separation_implementation}")
