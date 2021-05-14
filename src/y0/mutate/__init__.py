# -*- coding: utf-8 -*-

"""Functions that mutate probability expressions."""

from .canonicalize_expr import canonicalize

__all__ = [
    'canonicalize','conditional_to_fraction'  
]

def conditional_to_fraction( a: Probability ):
    if a.distribution.is_conditioned():
        children, parents = a.distribution.children, a.distribution.parents
        return P(children + parents )/Sum[parents](P(children + parents ))
