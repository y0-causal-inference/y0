"""Implementation of the IDC algorithm."""

from collections.abc import Sequence

from .id_std import identify
from .utils import Identification
from ..conditional_independencies import are_d_separated
from ...dsl import Expression, Variable
from ...graph import NxMixedGraph

__all__ = [
    "idc",
]


def idc(
    identification: Identification, *, ordering: Sequence[Variable] | None = None
) -> Expression:
    """Run the IDC algorithm from [shpitser2008]_.

    :param identification: The identification tuple
    :param ordering: A topological ordering of the variables. If not passed, is
        calculated from the directed component of the mixed graph.

    :returns: An expression created by the :func:`identify` algorithm after simplifying
        the original query

    Raises "Unidentifiable" if no appropriate identification can be found.
    """
    for condition in identification.conditions:
        if rule_2_of_do_calculus_applies(
            graph=identification.graph,
            treatments=identification.treatments,
            outcomes=identification.outcomes,
            conditions=identification.conditions,
            condition=condition,
        ):
            return idc(
                identification.exchange_observation_with_action(condition), ordering=ordering
            )

    # Run ID algorithm
    id_estimand = identify(identification.uncondition(), ordering=ordering)
    return id_estimand.normalize_marginalize(identification.outcomes)


def rule_2_of_do_calculus_applies(
    graph: NxMixedGraph,
    *,
    treatments: set[Variable],
    conditions: set[Variable],
    outcomes: set[Variable],
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
    reduced_graph = graph.remove_in_edges(treatments).remove_out_edges(condition)
    return all(
        are_d_separated(reduced_graph, outcome, condition, conditions=reduced_conditions)
        for outcome in outcomes
    )
