# -*- coding: utf-8 -*-

"""An implementation of the simplification algorithm from [tikka2018]_.

.. [tikka2018] Tikka, S., & Karvanen, J. (2018). `Simplifying probabilistic expressions in causal
   inference <https://arxiv.org/abs/1806.07082>`_. arXiv 1806.07082.
"""

from typing import Optional, Sequence, Union

from ananke.graphs import ADMG

from .dsl import Expression, Variable, _upgrade_ordering
from .graph import NxMixedGraph
from .predicates import has_markov_postcondition

__all__ = [
    'simplify',
]


def simplify(
    expression: Expression,
    graph: Union[NxMixedGraph, ADMG],
    ordering: Optional[Sequence[Union[str, Variable]]],
) -> Expression:
    """Simplify an expression.

    :param expression: An unsimplified expression
    :param graph: A graph
    :param ordering: An optional topological ordering. If not provided, one will be generated from the graph.
    :returns: A simplified expression
    :raises ValueError: if the simplified expression does not satisfy the markov postcondition
    """
    simplifier = Simplifier(graph=graph, ordering=_upgrade_ordering(ordering) if ordering is not None else None)
    rv = simplifier.simplify(expression)
    if not has_markov_postcondition(rv):
        raise ValueError('raised if the simplified expression does not satisfy the markov postcondition')
    return rv


class Simplifier:
    """A data structure to support application of the simplify algorithm."""

    #: The constant topological ordering
    ordering: Sequence[Variable]

    def __init__(
        self,
        graph: Union[NxMixedGraph, ADMG],
        ordering: Optional[Sequence[Variable]] = None,
    ) -> None:
        if isinstance(graph, NxMixedGraph):
            self.graph = graph.to_admg()
        elif isinstance(graph, ADMG):
            self.graph = graph
        else:
            raise TypeError

        if ordering is None:
            self.ordering = [Variable(name) for name in self.graph.topological_sort()]
        else:
            self.ordering = ordering

    # The simplify function is implemented inside a class since there is shared state between recursive calls
    #  and this makes it much easier to reference rather than making the simplify function itself have many
    #  arguments
    def simplify(self, expression: Expression) -> Expression:
        """Simplify the expression given the internal topological ordering."""
        raise NotImplementedError
