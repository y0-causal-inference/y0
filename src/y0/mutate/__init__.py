# -*- coding: utf-8 -*-

"""Functions that mutate probability expressions."""

from .canonicalize_expr import canonicalize
from .chain import chain_expand, markov_kernel_to_fraction

__all__ = [
    'canonicalize',
    'chain_expand',
    'markov_kernel_to_fraction',
]
