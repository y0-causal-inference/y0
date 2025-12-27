"""Tests for cyclic ID algorithm top level function implementation."""

from tests.test_algorithm import cases
from y0.algorithm.identify.id_cyclic import cyclic_id, initialize_district_distribution
from y0.dsl import A, B, C, D, P, Product
from y0.graph import NxMixedGraph


class TestInitializeDistrictDistribution(cases.GraphTestCase):
    """Tests for district initialization (Proposition 9.8)."""

    def test_single_node_cases(self) -> None:
        """Test single node districts return appropriate expressions."""
        parameters = [
            # graph_edges, district, apt_order, expected_expression
            # test 1: No predecessors
            ([(A, B)], {A}, [A, B], P(A), "no predecessors"),
            # test 2: With predecessors
            ([(A, B), (B, C)], {B}, [A, B, C], P(B | A), "with predecessors"),
            # test 3: Multiple predecessors
            ([(A, C), (B, C), (C, D)], {C}, [A, B, C, D], P(C | A, B), "multiple predecessors"),
        ]

        for edges, district, apt_order, expected, description in parameters:
            with self.subTest(msg=description):
                graph = NxMixedGraph.from_edges(directed=edges)
                result = initialize_district_distribution(graph, district, apt_order)
                self.assert_expr_equal(expected, result)

    # ------------------------------------------------------------------------------

    def test_cycle_with_predecessors(self) -> None:
        """Test district containing a cycle with predecessors returns joint conditional."""
        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C), (C, B)])
        district = {B, C}
        apt_order = [A, B, C]

        result = initialize_district_distribution(graph, district, apt_order)
        expected = P(B, C | A)

        self.assert_expr_equal(expected, result)

    def test_multiple_sccs_in_district(self) -> None:
        """Test district with multiple separate SCCs returns the product of their conditionals."""
        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, B), (C, D), (D, C)])
        district = {B, C, D}
        apt_order = [A, B, C, D]

        result = initialize_district_distribution(graph, district, apt_order)

        # SCC {B} has predecessor A: P(B | A)
        # SCC {C, D} has no predecessor: P(C, D)
        expected = Product.safe([P(B | A), P(C, D)])

        self.assert_expr_equal(expected, result)

    def test_self_loop_treated_as_scc(self) -> None:
        """Test that a self-loop is treated as single-node SCC."""
        # B should be treated as single-node SCC with predecessor A
        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, B), (B, C)])
        district = {B}
        apt_order = [A, B, C]

        result = initialize_district_distribution(graph, district, apt_order)
        expected = P(B | A)

        self.assert_expr_equal(expected, result)


# ------------------------------------------------------------------------------
class TestCyclicID(cases.GraphTestCase):
    """Tests for the main cyclic_id function for a top-level algorithm."""

    # testing line 1 - input validation
    def test_input_validation(self) -> None:
        """Test input validation before algorithm preconditions."""
        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C)])

        parameters = [
            # outcomes, interventions, error_pattern, description
            (None, {A}, "outcomes", "outcomes is None"),
            ({B}, None, "interventions", "interventions is None"),
            ([B], {A}, "set", "outcomes is list not set."),
            ({B}, [A], "set", "interventions is list not set."),
        ]

        for outcomes, interventions, error_pattern, description in parameters:
            with self.subTest(msg=description):
                with self.assertRaisesRegex(ValueError, error_pattern):
                    cyclic_id(graph, outcomes, interventions)

    # ---- Testing Line 2 ------------------------------
    def test_precondition_validation(self) -> None:
        """Test line 2 of the main cyclic ID algorithm."""
        # test cases that should raise a Value Error
        invalid_cases = [
            (set(), {A}, "empty|outcomes", "empty outcomes"),
            ({B}, {B}, "disjoint|overlap", "outcomes and interventions overlap"),
            ({D}, {A}, "subset|nodes", "outcomes not in graph"),
            ({B}, {D}, "subset|nodes", "interventions not in graph"),
        ]

        for outcomes, interventions, error_pattern, description in invalid_cases:
            with self.subTest(msg=description):
                graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C)])
                with self.assertRaisesRegex(ValueError, error_pattern):
                    cyclic_id(graph, outcomes, interventions)

        # test cases that should pass the validation check with no error

        valid_cases = [
            ({B}, set(), "empty interventions is valid"),
            ({B}, {A}, "valid outcomes and interventions"),
        ]

        for outcomes, interventions, description in valid_cases:
            with self.subTest(msg=description):
                graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C)])
                try:
                    cyclic_id(graph, outcomes, interventions)
                except ValueError as e:
                    # should only fail if it's a precondition validation error
                    error_msg = str(e).lower()
                    if any(
                        word in error_msg for word in ["subset", "disjoint", "empty", "overlap"]
                    ):
                        self.fail(f"Precondition validation failed unexpectedly: {description}")

    # ----- Testing Line 3 ---------------------------
