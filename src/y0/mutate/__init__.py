# -*- coding: utf-8 -*-

"""Functions that mutate probability expressions."""

from .canonicalize_expr import canonicalize, canonical_expr_equal
from .chain import bayes_expand, chain_expand, fraction_expand

__all__ = [
    "canonicalize",
    "canonical_expr_equal",
    "chain_expand",
    "fraction_expand",
    "bayes_expand",
]
