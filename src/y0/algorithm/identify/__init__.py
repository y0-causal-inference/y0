# -*- coding: utf-8 -*-

"""Identification algorithms."""

from .id_c import idc
from .id_star import id_star
from .id_std import identify
from .utils import Identification, Query, Unidentifiable

__all__ = [
    # Algorithms
    "identify",
    "idc",
    "id_star",
    # Data Structures
    "Query",
    # Exceptions
    "Unidentifiable",
    "Identification",
    # internal methods
    "get_district_domains",
    "domain_of_counterfactual_values",
    "id_star_line_1",
    "is_self_intervened"
] + [f"id_star_line_{i}" for i in [4, 6, 8]]
