# -*- coding: utf-8 -*-

"""Data structures."""

from __future__ import annotations

from operator import attrgetter
from typing import Iterable, NamedTuple, Optional, Tuple, Union
from ananke.graphs import SG

from itertools import chain

from .dsl import Expression, Variable, _upgrade_ordering

__all__ = [
    'VermaConstraint',
    'ConditionalIndependency',
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


class ConditionalIndependency(NamedTuple):
    """A conditional independency."""

    left: Variable
    right: Variable
    observations: Tuple[Variable, ...]

    def __repr__(self):
        left = self.left.name
        right = self.right.name
        observations = tuple(obs.name for obs in self.observations)
        return f"ConditionalIndependency('{left}', '{right}', {observations})"

    @property
    def is_canonical(self) -> bool:
        """Return if the conditional independency is canonical."""
        return self.left.name < self.right.name and isinstance(self.observations, tuple)

    @classmethod
    def create(
        cls,
        left: Union[str, Variable],
        right: Union[str, Variable],
        observations: Optional[Iterable[Union[str, Variable]]] = tuple(),
        graph: Optional[SG] = None
    ) -> ConditionalIndependency:
        """Create a canonical conditional independency."""
        
        if isinstance(left, str):
            left = Variable(name=left)
        if isinstance(right, str):
            right = Variable(name=right)
        if left.name > right.name:
            left, right = right, left
            
        observations = set(_upgrade_ordering(observations)) # Remove duplicates, maybe make into Variables
            
        if graph is not None:
            all_edges = [edge 
                         for pth in graph.directed_paths([left.name], [right.name])
                         for edge in pth ]
            permissible_nodes = set(chain(*zip(*all_edges)))
            #permissible = graph.ancestors([left.name,right.name])
            observations = [obs for obs in observations if obs.name in permissible_nodes]
            
        observations = tuple(sorted(set(_upgrade_ordering(observations)), key=attrgetter('name')))  # type: ignore
            
        return cls(left, right, observations)
