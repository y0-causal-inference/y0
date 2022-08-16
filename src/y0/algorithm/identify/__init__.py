# -*- coding: utf-8 -*-

"""Identification algorithms."""

from .id_c import idc
from .id_std import identify
from .utils import Identification, Query, Unidentifiable

__all__ = [
    # Algorithms
    "identify",
    "idc",
    # Data Structures
    "Query",
    # Exceptions
    "Unidentifiable",
    "Identification",
]
