# -*- coding: utf-8 -*-

"""Interface to the R causaleffect package via :mod:`rpy2`."""

from __future__ import annotations

import logging
from typing import Iterable, NamedTuple, Sequence

from rpy2 import robjects
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects.vectors import StrVector

from y0.graph import NxMixedGraph, figure_1

logger = logging.getLogger(__name__)

CAUSALEFFECT = 'causaleffect'
IGRAPH = 'igraph'

R_REQUIREMENTS = [
    CAUSALEFFECT,
    IGRAPH,
]


def prepare_renv(*, requirements: Iterable[str]) -> None:
    """Ensure the given R packages are installed.

    :param requirements: A list of R packages to ensure are installed

    .. seealso:: https://rpy2.github.io/doc/v3.4.x/html/introduction.html#installing-packages
    """
    # import R's utility package
    utils = importr('utils')

    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

    requirements = [
        requirement
        for requirement in requirements
        if not isinstalled(requirement)
    ]
    if requirements:
        logger.warning('installing R packages: %s', requirements)
        utils.install_packages(StrVector(requirements))


class VermaConstraint(NamedTuple):
    """Represent a Verma constraint output by causaleffect."""

    lhs_cfactor: str
    lhs_expr: str
    rhs_cfactor: str
    rhs_expr: str

    @classmethod
    def from_element(cls, element) -> VermaConstraint:
        """Extract content from each element in the vector returned by `verma.constraint`.

        :param element: An element in the in the vector returned by `verma.constraint`
        :returns: A Verma constraint tuple for the given element

        .. seealso:: Extracting from R objects https://rpy2.github.io/doc/v3.4.x/html/vector.html#extracting-items
        """
        return cls(
            rhs_cfactor=element.rx('rhs.cfactor')[0][0],
            rhs_expr=element.rx('rhs.expr')[0][0],
            lhs_cfactor=element.rx('lhs.cfactor')[0][0],
            lhs_expr=element.rx('lhs.expr')[0][0],
        )


def r_get_verma_constraints(graph) -> Sequence[VermaConstraint]:
    """Calculate the verma constraints on the graph using ``causaleffect``."""
    if isinstance(graph, NxMixedGraph):
        graph = graph.to_causaleffect()
    verma_constraints = robjects.r['verma.constraints']
    return [
        VermaConstraint.from_element(row)
        for row in verma_constraints(graph)
    ]


def _main():
    prepare_renv(requirements=R_REQUIREMENTS)

    importr(CAUSALEFFECT)
    importr(IGRAPH)

    graph = figure_1.to_causaleffect()
    print(graph)

    # Get verma constraints
    # see also: https://rpy2.github.io/doc/v3.4.x/html/vector.html#extracting-items
    verma_constraints = r_get_verma_constraints(graph)
    print(*verma_constraints, sep='\n')


if __name__ == '__main__':
    _main()
