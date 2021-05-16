# -*- coding: utf-8 -*-

"""Functions that mutate probability expressions."""

from .canonicalize_expr import canonicalize
from .chain import chain_expand, fraction_expand

__all__ = [
    'canonicalize',
    'chain_expand',
    'fraction_expand',
]
