
"""Tests for the base_distribution parameter of the cyclic ID algorithm.

This module tests:
1. cyclic_id() with base_distribution parameter - identifies P(Y|do(W)) given
   interventional data P[do(J)](V) as background context.
"""

from tests.test_algorithm import cases
from y0.algorithm.identify.cyclic_id import Unidentifiable, cyclic_id
from y0.dsl import P, X, Y, Z
from y0.graph import NxMixedGraph



class TestInitialDistributionParameter(cases.GraphTestCase):
    """
    Tests for the base_distribution parameter of the cyclic ID algorithm.

    This new parameter added allows the cyclic ID to work with an additional
    "perturbation" in order to see the effect of an additional intervention.
    """

    def test_manual_vs_automatic_graph_mutilation(self):
        """Verify automatic mutilation matches manual graph surgery."""
        # original graph with Z
        graph_with_z = NxMixedGraph.from_edges(directed=[(Z, X), (Z, Y), (X, Y)], undirected=[])

        # Manually mutilated graph (remove Z)
        graph_without_z = NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[])

        # added base_distribution parameter added to cyclic ID
        interventional_dist = P[Z](X, Y, Z)
        result_automatic = cyclic_id(
            graph=graph_with_z,
            outcomes={Y},
            interventions={X},
            base_distribution=interventional_dist,
        )

        # manual method
        result_manual = cyclic_id(
            graph=graph_without_z,
            outcomes={Y},
            interventions={X},
        )

        # if the feature is implemented right, should be equal expressions
        self.assertEqual(
            result_automatic,
            result_manual,
            "Automatic mutilation should match manual graph surgery",
        )

    def test_scc_cycle_breaker_with_interventional_data(self):
        """SCC cycle breaker: X→Y→Z→X unidentifiable, identifiable with P[do(Z)](V)."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z), (Z, X)])

        with self.assertRaises(Unidentifiable):
            cyclic_id(graph, outcomes={Y}, interventions={X})

        interventional_data = P[Z](X, Y, Z)
        result = cyclic_id(
            graph, outcomes={Y}, interventions={X}, base_distribution=interventional_data
        )
        expected = P(X, Y) / P(X)

        self.assert_expr_equal(expected, result)

    def test_mediated_bow_arc_break_with_interventional_data(self):
        """Mediated bow-arc: X→Z→Y with X↔Z unidentifiable, identifiable with P[do(Z)](V)."""
        # X→Z→Y with X↔Z
        graph = NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)], undirected=[(X, Z)])

        with self.assertRaises(Unidentifiable):
            cyclic_id(graph, outcomes={Y}, interventions={X})

        interventional_data = P[Z](X, Y, Z)
        result = cyclic_id(
            graph, outcomes={Y}, interventions={X}, base_distribution=interventional_data
        )

        self.assert_expr_equal(P(Y), result)

    def test_overlapping_interventions_raises_error(self):
        """Verify error when J ∩ W ≠ ∅.

        X appears in both the base_distribution and interventions,
        which can't work. You cannot use P[do(X)] data to identify do(X).
        """
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])

        with self.assertRaises(ValueError) as cm:
            cyclic_id(graph, outcomes={Z}, interventions={X}, base_distribution=P[X](X, Y, Z))

        self.assertIn("must be disjoint", str(cm.exception))
        
    

    def test_identifiable_stays_identifiable_with_base_distribution(self):
        """Identifiable query stays identifiable when base_distribution is added.

        Simple DAG: Z→X→Y (no confounding)
        Without base_distribution: P(Y|do(X)) is identifiable = P(Y|X)
        With base_distribution P[do(Z)](X,Y,Z): should still be identifiable, same result.
        """
        graph = NxMixedGraph.from_edges(directed=[(Z, X), (X, Y)], undirected=[])

        # without base_distribution
        result_without = cyclic_id(graph, outcomes={Y}, interventions={X})

        # with base_distribution - adding Z intervention as background
        result_with = cyclic_id(
            graph, outcomes={Y}, interventions={X}, base_distribution=P[Z](X, Y, Z)
        )

        self.assertIsNotNone(result_with)
        expected = P(X, Y) / P(X)
        
        # both should be identifiable and equal
        self.assert_expr_equal(expected, result_with)
        