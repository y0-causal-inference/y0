# -*- coding: utf-8 -*-

"""Interface to the R causaleffect package via :mod:`rpy2`."""

from __future__ import annotations

import logging
from typing import NamedTuple, Sequence, Tuple, Union

from rpy2 import robjects

from y0.dsl import Expression, Variable
from y0.examples import verma_1
from y0.graph import CausalEffectGraph, NxMixedGraph
from y0.parser import parse_causaleffect
from y0.r_utils import uses_r

logger = logging.getLogger(__name__)


class VermaConstraint(NamedTuple):
    """Represent a Verma constraint output by causaleffect."""

    lhs_cfactor: Expression
    lhs_expr: Expression
    rhs_cfactor: Expression
    rhs_expr: Expression
    variables: Tuple[Variable, ...]

    @classmethod
    def from_element(cls, element) -> VermaConstraint:
        """Extract content from each element in the vector returned by `verma.constraint`.

        :param element: An element in the in the vector returned by `verma.constraint`
        :returns: A Verma constraint tuple for the given element

        .. seealso:: Extracting from R objects https://rpy2.github.io/doc/v3.4.x/html/vector.html#extracting-items
        """
        return cls(
            rhs_cfactor=parse_causaleffect(_extract(element, 'rhs.cfactor')),
            rhs_expr=parse_causaleffect(_extract(element, 'rhs.expr')),
            lhs_cfactor=parse_causaleffect(_extract(element, 'lhs.cfactor')),
            lhs_expr=parse_causaleffect(_extract(element, 'lhs.expr')),
            variables=_parse_vars(element),
        )


def _parse_vars(element) -> Tuple[Variable, ...]:
    _vars = element.rx('vars')
    return tuple(
        Variable(name)
        for name in sorted(_vars[0])
    )


def _extract(element, key):
    return element.rx(key)[0][0]


@uses_r
def r_get_verma_constraints(graph: Union[NxMixedGraph, CausalEffectGraph]) -> Sequence[VermaConstraint]:
    """Calculate the verma constraints on the graph using ``causaleffect``."""
    if isinstance(graph, NxMixedGraph):
        graph = graph.to_causaleffect()
    verma_constraints = robjects.r['verma.constraints']
    return [
        VermaConstraint.from_element(row)
        for row in verma_constraints(graph)
    ]


def _main():
    verma_constraints = r_get_verma_constraints(verma_1)
    print(*verma_constraints, sep='\n')


if __name__ == '__main__':
    _main()
