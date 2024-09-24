"""General utilities for :mod:`rpy2`."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from functools import lru_cache, wraps
from typing import Any, TypeVar, cast

from rpy2.robjects.packages import InstalledPackage, InstalledSTPackage, importr, isinstalled
from rpy2.robjects.vectors import StrVector

from .dsl import Variable

__all__ = ["uses_r", "prepare_renv", "prepare_default_renv"]

logger = logging.getLogger(__name__)

CAUSALEFFECT = "causaleffect"
IGRAPH = "igraph"
R_REQUIREMENTS = [
    CAUSALEFFECT,
    IGRAPH,
]


T = TypeVar("T")
Func = Callable[..., T]


def prepare_renv(requirements: Iterable[str]) -> list[InstalledSTPackage | InstalledPackage]:
    """Ensure the given R packages are installed.

    :param requirements: A list of R package names to ensure are installed
    :returns: A list of R packages

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

    return [importr(requirement) for requirement in requirements]


@lru_cache(maxsize=1)
def prepare_default_renv() -> bool:
    """Prepare the default R environment."""
    prepare_renv(R_REQUIREMENTS)
    return True


def uses_r(f: Callable[..., T]) -> Callable[..., T]:
    """Decorate functions that use R."""

    @wraps(f)
    def _wrapped(*args: Any, **kwargs: Any) -> T:
        prepare_default_renv()
        return f(*args, **kwargs)

    return _wrapped


def _parse_vars(element: Any) -> tuple[Variable, ...]:
    _vars = element.rx("vars")
    return tuple(Variable(name) for name in sorted(_vars[0]))


def _extract(element: Any, key: str) -> str:
    return cast(str, element.rx(key)[0][0])
