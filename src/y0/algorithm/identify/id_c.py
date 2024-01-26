# -*- coding: utf-8 -*-

"""Implementation of the IDC algorithm."""

from .id_std import identify
from .utils import Identification
from ..conditional_independencies import are_d_separated
from ...dsl import Expression, Variable

__all__ = [
    "idc",
    "rule_2_of_do_calculus_applies",
]


def idc(identification: Identification) -> Expression:
    """Run the IDC algorithm from [shpitser2008]_.

    :param identification: The identification tuple
    :returns: An expression created by the :func:`identify` algorithm after simplifying the original query

    Raises "Unidentifiable" if no appropriate identification can be found.
    """
    for condition in identification.conditions:
        if rule_2_of_do_calculus_applies(identification=identification, condition=condition):
            return idc(identification.exchange_observation_with_action(condition))

    # Run ID algorithm
    id_estimand = identify(identification.uncondition())
    return id_estimand.normalize_marginalize(identification.outcomes)


def rule_2_of_do_calculus_applies(identification: Identification, condition: Variable) -> bool:
    r"""Check if Rule 2 of the Do-Calculus applies to the conditioned variable.

    :param identification: The identification tuple
    :param condition: The condition to check
    :returns: If rule 2 applies, see below.

    If Rule 2 of the do calculus applies to the conditioned variable, then it can be converted to a do variable.

    .. math::

        \newcommand\ci{\perp\!\!\!\perp}
        \newcommand{\ubar}[1]{\underset{\bar{}}{#1}}
        \newcommand{\obar}[1]{\overset{\bar{}}{#1}}
        \text{if } (\exists Z \in \mathbf{Z})(\mathbf{Y} \ci Z | \mathbf{X}, \mathbf{Z}
        - \{Z\})_{G_{\bar{\mathbf{X}}\ubar{Z}}} \\
        \text{then } P(\mathbf{Y}|do(\mathbf{X}),\mathbf{Z}) = P(\mathbf Y|do(\mathbf X), do(Z), \mathbf{Z} - \{Z\})
    """
    graph = identification.graph
    treatments = identification.treatments
    conditions = treatments | (identification.conditions - {condition})
    graph_mod = graph.remove_in_edges(treatments).remove_out_edges(condition)
    return all(
        are_d_separated(graph_mod, outcome, condition, conditions=conditions)
        for outcome in identification.outcomes
    )
