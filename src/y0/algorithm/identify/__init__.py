# -*- coding: utf-8 -*-

"""Identification algorithms."""

from .id_c import idc  # noqa:F401
from .id_std import identify  # noqa:F401
from .utils import Fail, Identification, Query  # noqa:F401

__all__ = [
    "identify",
    "idc",
    "Fail",
    "Query",
    "Identification",
]
