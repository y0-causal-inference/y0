"""Do Calculus."""

from typing import Annotated

from .conditional_independencies import are_d_separated
from ..dsl import Variable
from ..graph import NxMixedGraph
from ..util import InPaperAs

__all__ = [
    "rule_2_of_do_calculus_applies",
]


def rule_2_of_do_calculus_applies(
    graph: Annotated[NxMixedGraph, InPaperAs("G")],
    *,
    treatments: Annotated[set[Variable], InPaperAs(r"\mathbf{X}")],
    outcomes: Annotated[set[Variable], InPaperAs(r"\mathbf{Y}")],
    conditions: Annotated[set[Variable], InPaperAs(r"\mathbf{Z}")],
    condition: Variable,
) -> bool:
    r"""Check if Rule 2 of the Do-Calculus applies to the conditioned variable.

    :param graph: The graph
    :param treatments:
    :param conditions:
    :param outcomes:
    :param condition: The condition to check

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
    return all(
        are_d_separated(reduced_graph, outcome, condition, conditions=reduced_conditions)
        for outcome in outcomes
    )
