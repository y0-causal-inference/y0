"""Implementation of the IDC algorithm."""

from collections.abc import Sequence

from .id_std import identify
from .utils import Identification
from ..do_calculus import rule_2_of_do_calculus_applies
from ...dsl import Expression, Variable

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
