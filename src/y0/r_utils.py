# -*- coding: utf-8 -*-

"""General utilities for :mod:`rpy2`."""

import logging
from functools import lru_cache, wraps
from typing import Iterable, Tuple

from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects.vectors import StrVector

from .dsl import Variable

__all__ = [
    "uses_r",
]

logger = logging.getLogger(__name__)

CAUSALEFFECT = "causaleffect"
IGRAPH = "igraph"
R_REQUIREMENTS = [
    CAUSALEFFECT,
    IGRAPH,
]


def prepare_renv(requirements: Iterable[str]) -> None:
    """Ensure the given R packages are installed.

    :param requirements: A list of R packages to ensure are installed

    .. seealso:: https://rpy2.github.io/doc/v3.4.x/html/introduction.html#installing-packages
    """
    # import R's utility package
    utils = importr("utils")

    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

    uninstalled_requirements = [
        requirement for requirement in requirements if not isinstalled(requirement)
    ]
    if uninstalled_requirements:
        logger.warning("installing R packages: %s", uninstalled_requirements)
        utils.install_packages(StrVector(uninstalled_requirements))

    for requirement in requirements:
        importr(requirement)


@lru_cache(maxsize=1)
def prepare_default_renv() -> bool:
    """Prepare the default R environment."""
    prepare_renv(R_REQUIREMENTS)
    return True


def uses_r(f):
    """Decorate functions that use R."""

    @wraps(f)
    def _wrapped(*args, **kwargs):
        prepare_default_renv()
        return f(*args, **kwargs)

    return _wrapped


def _parse_vars(element) -> Tuple[Variable, ...]:
    _vars = element.rx("vars")
    return tuple(Variable(name) for name in sorted(_vars[0]))


def _extract(element, key):
    return element.rx(key)[0][0]
