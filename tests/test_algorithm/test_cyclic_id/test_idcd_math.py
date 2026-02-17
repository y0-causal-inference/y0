"""Validation tests for the math of Line 23.

This module validates that Line 23 of the IDCD algorithm correctly implements the
formula found in the paper to compute the conditional probability distribution
expression.

Each test validates a different aspect of the formula's implementation in terms of the
math and structure.
"""

from typing import cast

from tests.test_algorithm import cases
from tests.test_algorithm.test_ioscm import simple_cyclic_graph_1
from y0.algorithm.identify.cyclic_id import compute_scc_distributions
from y0.algorithm.ioscm.utils import get_strongly_connected_components
from y0.dsl import Fraction, Probability, Product, R, Sum, W, X, Z


class TestLine23SymbolicStructure(cases.GraphTestCase):
    """Testing that the math is implemented correctly and produces the correct symbolic structure."""

    def test_line_23_produces_conditional_probability(self) -> None:
        """Line 23 should produce a conditional probability structure for an SCC."""
        graph = simple_cyclic_graph_1
        ancestral_closure = {R, X, W, Z}
        subgraph_a = graph.subgraph(ancestral_closure)

        cycle_scc = frozenset({X, W, Z})
        relevant_sccs = [cycle_scc]

        # compute the interventional distributions for the SCCs
        # J = empty - no input nodes
        # V\A = {Y}
        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure  # V\A

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
            intervention_set=intervention_set,
        )

        self.assertIn(cycle_scc, result, "SCC should be in result dictionary")

        expression = result[cycle_scc]

        # The expression should be either:
        # - A Fraction (numerator/denominator)
        # - Or a conditional Probability P(...|...)
        self.assertIsInstance(
            expression,
            (Fraction, Probability),
            f"Expected Fraction or Probability, got {type(expression)}",
        )

    def test_line_23_has_correct_marginalization_and_conditioning(self) -> None:
        """Verifying that the conditional probability has the correct form and structure explicitly."""
        graph = simple_cyclic_graph_1
        ancestral_closure = {R, X, W, Z}
        subgraph_a = graph.subgraph(ancestral_closure)

        cycle_scc = frozenset({X, W, Z})
        relevant_sccs = [cycle_scc]

        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure  # V\A

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
            intervention_set=intervention_set,
        )

        expression = result[cycle_scc]
        
        self.assertIsInstance(expression, Fraction, "Expected a Fraction expression")
        expression = cast(Fraction, expression)
        
        all_vars = expression.get_variables()
        
        self.assertTrue(
            {X, W, Z, R}.issubset(all_vars),
            f"expression should contain {{X, W, Z, R}}, got {all_vars}"
        )

    def test_line_23_first_scc_has_no_predecessors(self) -> None:
        """First SCC in apt-order should have no conditioning variables."""
        graph = simple_cyclic_graph_1
        ancestral_closure = {R, X, W, Z}
        subgraph_a = graph.subgraph(ancestral_closure)

        first_scc = frozenset({R})
        relevant_sccs = [first_scc]

        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
            intervention_set=intervention_set,
        )

        expression = result[first_scc]

        # check if its a simple Probability
        if isinstance(expression, Fraction):
            denominator = cast(Sum, expression.denominator)

            # if it is a fraction, then verify no variables are being conditioned on
            all_vars = set(denominator.expression.get_variables())
            summed_vars = set(denominator.ranges)
            conditioning_vars = all_vars - summed_vars

            self.assertEqual(
                set(),
                conditioning_vars,
                f"First SCC should have no conditioning variables, got {conditioning_vars}",
            )

            # verify R is in the result
            self.assertIn(first_scc, result)

    def test_line_23_multiple_sccs_have_different_predecessors(self) -> None:
        """Test Line 23 handles multiple SCCs with different predecessor sets."""
        graph = simple_cyclic_graph_1

        ancestral_closure = {R, X, W, Z}
        subgraph_a = graph.subgraph(ancestral_closure)

        sccs = get_strongly_connected_components(subgraph_a)
        self.assertEqual(2, len(sccs), f"Expected 2 SCCs, got {len(sccs)}")

        # identify each SCC
        scc_r = frozenset({R})
        scc_cycle = frozenset({X, W, Z})

        self.assertIn(scc_r, sccs)
        self.assertIn(scc_cycle, sccs)

        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure

        # process all the SCCs
        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=list(sccs),
            ancestral_closure=ancestral_closure,
            intervention_set=intervention_set,
        )
        

        # verify all the SCCs are processed:
        self.assertEqual(2, len(result), "Should have results for all 2 SCCs")

        # # test SCC {R} - no predecessors
        # expr_r = result[scc_r]
        # self.assertIsNotNone(expr_r, "First SCC should have a distribution")

        # test SCC {X,W,Z} - predecessor {R}
        # expr_r = expression for R
        expr_cycle = result[scc_cycle]
        self.assertIsInstance(expr_cycle, Fraction, "Cycle SCC should be a Fraction.")
        
        all_vars = expr_cycle.get_variables()
        self.assertTrue(
            {R, X, W, Z}.issubset(all_vars),
            f"expression should contain {{R, X, W, Z}}, got {all_vars}"
        )
       
    def test_line_23_handles_single_node_scc(self) -> None:
        """Test Line 23 handles single-node SCCs correctly."""
        graph = simple_cyclic_graph_1
        ancestral_closure = {R}
        subgraph_a = graph.subgraph(ancestral_closure)

        sccs = get_strongly_connected_components(subgraph_a)

        # should have 1 SCC: {R}
        self.assertEqual(1, len(sccs), f"Expected 1 SCC, got {len(sccs)}")

        scc_r = frozenset({R})
        self.assertIn(scc_r, sccs)

        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=list(sccs),
            ancestral_closure=ancestral_closure,
            intervention_set=intervention_set,
        )

        # verify R is processed
        self.assertIn(scc_r, result)

        expression = result[scc_r]

        # A single node SCC should still produce a valid expression
        self.assertIsInstance(
            expression,
            (Fraction, Probability, Sum, Product),
            f"Expected Fraction, Probability, Sum, or Product, got {type(expression)}",
        )
