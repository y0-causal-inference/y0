"""An implementation of Cyclic IDC, based off of the extension from ID to IDC."""

from collections.abc import Iterable, Sequence

from .cyclic_id import cyclic_id
from .utils import Identification
from ..do_calculus import rule_2_of_do_calculus_applies
from ...dsl import Expression, Variable
from ...graph import NxMixedGraph

__all__ = ["cyclic_idc"]


def cyclic_idc(
    graph: NxMixedGraph,
    outcomes: Variable | Iterable[Variable],
    interventions: Variable | Iterable[Variable],
    conditions: Variable | Iterable[Variable],
    *,
    ordering: Sequence[Variable] | None = None,
) -> Expression:
    """Run cyclic ID with support for conditions."""
    identification = Identification.from_parts(
        graph=graph,
        outcomes=outcomes,
        treatments=interventions,
        conditions=conditions,
    )
    for condition in identification.conditions:
        if rule_2_of_do_calculus_applies(
            graph=identification.graph,
            treatments=identification.treatments,
            outcomes=identification.outcomes,
            conditions=identification.conditions,
            condition=condition,
            separation_implementation="sigma",
        ):
            modified = identification.exchange_observation_with_action(condition)
            return cyclic_idc(
                graph=modified.graph,
                outcomes=modified.outcomes,
                interventions=modified.treatments,
                conditions=modified.conditions,
                ordering=ordering,
            )

    modified = identification.uncondition()
    id_estimand = cyclic_id(
        graph=modified.graph,
        outcomes=modified.outcomes,
        interventions=modified.treatments,
        ordering=ordering,
    )
    return id_estimand.normalize_marginalize(identification.outcomes)
