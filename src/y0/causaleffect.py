# -*- coding: utf-8 -*-

"""Interface to the R causaleffect package via :mod:`rpy2`."""

import logging
from typing import Iterable

import pystow
from rpy2 import robjects
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects.vectors import StrVector

from y0.graph import napkin_graph

logger = logging.getLogger(__name__)

CAUSALEFFECT = 'causaleffect'
IGRAPH = 'igraph'

R_REQUIREMENTS = [
    CAUSALEFFECT,
    IGRAPH,
]

Y0_R_LIB_LOC = pystow.get('y0', 'r')


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


def _main():
    prepare_renv(requirements=R_REQUIREMENTS)

    importr(CAUSALEFFECT)
    importr(IGRAPH)

    graph_code = napkin_graph.to_causaleffect_str()
    graph = robjects.r(graph_code)
    print(graph)

    verma_constraints = robjects.r['verma.constraints']
    rv = verma_constraints(graph)
    print(rv[0].rx('rhs.cfactor'))
    print(rv[0].rx('rhs.expr'))
    print(rv[0].rx('lhs.cfactor'))
    print(rv[0].rx('lhs.expr'))


if __name__ == '__main__':
    _main()
