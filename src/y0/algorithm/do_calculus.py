"""Do Calculus."""

from typing import Annotated

from .conditional_independencies import are_d_separated
from ..dsl import Variable
from ..graph import NxMixedGraph
from ..util import InPaperAs

__all__ = [
    "rule_1_of_do_calculus_applies",
    "rule_2_of_do_calculus_applies",
    "rule_3_of_do_calculus_applies",
    "satisfies_backdoor",
    "satisfies_frontdoor",
]


def rule_1_of_do_calculus_applies(
    graph: NxMixedGraph,
    *,
    treatments: set[Variable],
    outcomes: set[Variable],
    conditions: set[Variable],
    observation: Variable,
) -> bool:
    """Check if Rule 1 of the Do-Calculus applies. Ref: do_calculus.dfy Rule1_InsertDeleteObservation."""
    mutilated = graph.remove_in_edges(treatments)
    return all(
        are_d_separated(mutilated, outcome, observation, conditions=treatments | conditions)
        for outcome in outcomes
    )


def rule_3_of_do_calculus_applies(
    graph: NxMixedGraph,
    *,
    treatments: set[Variable],
    outcomes: set[Variable],
    conditions: set[Variable],
    action: Variable,
) -> bool:
    """Check if Rule 3 of the Do-Calculus applies. Ref: do_calculus.dfy Rule3_InsertDeleteAction."""
    gx = graph.remove_in_edges(treatments)
    ancestors_of_w = gx.ancestors_inclusive(conditions)
    # Only remove in-edges of action if it is NOT an ancestor of any condition
    z_not_anc = {action} - ancestors_of_w
    gxz = gx.remove_in_edges(z_not_anc)
    return all(
        are_d_separated(gxz, outcome, action, conditions=treatments | conditions)
        for outcome in outcomes
    )


def satisfies_backdoor(
    graph: NxMixedGraph,
    *,
    outcomes: set[Variable],
    treatments: set[Variable],
    adjustment: set[Variable],
) -> bool:
    """Check if adjustment set satisfies the backdoor criterion. Ref: do_calculus.dfy BackdoorAdjustment."""
    # (i) No adjustment variable may be a proper descendant of any treatment
    descendants_of_x = graph.descendants_inclusive(treatments)
    if adjustment & (descendants_of_x - treatments):
        return False
    # (ii) Adjustment blocks all backdoor paths (paths through X's parents).
    # Check Y ⊥ X | Z in G with X's outgoing edges removed — this removes causal
    # paths and leaves only backdoor paths, matching Dafny DSep which ignores
    # direct single-step trails between the query nodes.
    g_xout = graph.remove_out_edges(treatments)
    return all(
        are_d_separated(g_xout, outcome, treatment, conditions=adjustment)
        for outcome in outcomes
        for treatment in treatments
    )


def satisfies_frontdoor(
    graph: NxMixedGraph,
    *,
    outcomes: set[Variable],
    treatments: set[Variable],
    mediators: set[Variable],
) -> bool:
    """Check if mediator set satisfies the frontdoor criterion. Ref: do_calculus.dfy FrontdoorCriterion."""
    raise NotImplementedError


# TODO implement rule 1 condition
# TODO implement rule 3 condition


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
    :param treatments: The treatments
    :param conditions: The conditions
    :param outcomes: The outcomes
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
