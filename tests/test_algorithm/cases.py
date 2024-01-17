# -*- coding: utf-8 -*-

"""Test cases."""

import unittest

from y0.dsl import Expression, get_outcomes_and_treatments
from y0.graph import NxMixedGraph
from y0.mutate import canonicalize

__all__ = ["GraphTestCase"]


class GraphTestCase(unittest.TestCase):
    """Tests parallel worlds and counterfactual graphs."""

    def assert_graph_equal(
        self, expected: NxMixedGraph, actual: NxMixedGraph, msg=None, *, sort: bool = False
    ) -> None:
        """Check the graphs are equal (more nice than the builtin :meth:`NxMixedGraph.__eq__` for testing)."""
        if sort:
            self.assertEqual(
                sorted(set(expected.directed.nodes())),
                sorted(set(actual.directed.nodes())),
                msg=msg,
            )
        else:
            self.assertEqual(set(expected.directed.nodes()), set(actual.directed.nodes()), msg=msg)
        self.assertEqual(set(expected.undirected.nodes()), set(actual.undirected.nodes()), msg=msg)
        self.assertEqual(set(expected.directed.edges()), set(actual.directed.edges()), msg=msg)
        self.assertEqual(
            set(map(frozenset, expected.undirected.edges())),
            set(map(frozenset, actual.undirected.edges())),
            msg=msg,
        )

    def assert_expr_equal(self, expected: Expression, actual: Expression) -> None:
        """Assert that two expressions are the same."""
        expected_outcomes, expected_treatments = get_outcomes_and_treatments(query=expected)
        actual_outcomes, actual_treatments = get_outcomes_and_treatments(query=actual)
        self.assertEqual(expected_treatments, actual_treatments)
        self.assertEqual(expected_outcomes, actual_outcomes)
        ordering = sorted(expected.get_variables(), key=lambda x: str(x))
        expected_canonical = canonicalize(expected, ordering)
        actual_canonical = canonicalize(actual, ordering)
        self.assertEqual(
            expected_canonical,
            actual_canonical,
            msg=f"\nExpected: {str(expected_canonical)}\nActual:   {str(actual_canonical)}",
        )
