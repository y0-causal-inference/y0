# -*- coding: utf-8 -*-

"""Data structures."""

from __future__ import annotations

from typing import NamedTuple, Tuple

from .dsl import Expression, Variable

__all__ = [
    'VermaConstraint',
]


class VermaConstraint(NamedTuple):
    """Represent a Verma constraint."""

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
        from .parser import parse_causaleffect
        from .r_utils import _extract, _parse_vars
        print(element)
        return cls(
            rhs_cfactor=parse_causaleffect(_extract(element, 'rhs.cfactor')),
            rhs_expr=parse_causaleffect(_extract(element, 'rhs.expr')),
            lhs_cfactor=parse_causaleffect(_extract(element, 'lhs.cfactor')),
            lhs_expr=parse_causaleffect(_extract(element, 'lhs.expr')),
            variables=_parse_vars(element),
        )
