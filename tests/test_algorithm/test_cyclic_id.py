"""Tests for cyclic ID algorithm top level function implementation."""

from tests.test_algorithm import cases
from y0.algorithm.identify import Unidentifiable
from y0.algorithm.identify.id_cyclic import cyclic_id, initialize_district_distribution
from y0.dsl import A, B, C, D, E, Expression, P, Product, R, W, X, Y, Z
from y0.graph import NxMixedGraph


class TestInitializeDistrictDistribution(cases.GraphTestCase):
    """Tests for district initialization (Proposition 9.8)."""

    def test_single_node_cases(self) -> None:
        """Test single node districts return appropriate expressions."""
        parameters = [
            # graph_edges, district, apt_order, expected_expression
            # test 1: No predecessors
            (
                [(A, B)],
                {A},
                [A, B],
                P(A),
                "no predecessors",
            ),  # expected simplified expression = P(A)
            # test 2: With predecessors
            (
                [(A, B), (B, C)],
                {B},
                [A, B, C],
                P(A, B) / P(A),
                "with predecessors",
            ),  # expected simplified expression = P(B | A)
            # test 3: Multiple predecessors
            (
                [(A, C), (B, C), (C, D)],
                {C},
                [A, B, C, D],
                P(A, B, C) / P(A, B),
                "multiple predecessors",
            ),  # expected simplified expression = P(C | A, B)
        ]

        for edges, district, apt_order, expected, description in parameters:
            with self.subTest(msg=description):
                graph = NxMixedGraph.from_edges(directed=edges)
                result = initialize_district_distribution(graph, district, apt_order)
                self.assert_expr_equal(expected, result)

    # ------------------------------------------------------------------------------

    def test_cycle_with_predecessors(self) -> None:
        """Test district containing a cycle with predecessors returns joint conditional."""
        # district: {B, C} is one SCC with predecessor A
        # expected: P(B, C | A) = P(A, B, C) / P(A)

        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C), (C, B)])
        district = {B, C}
        apt_order = [A, B, C]

        result = initialize_district_distribution(graph, district, apt_order)
        expected = P(A, B, C) / P(A)

        self.assert_expr_equal(expected, result)

    def test_multiple_sccs_in_one_district(self) -> None:
        # FIXME - this test passes sometimes and then also sometimes fails? I'm not sure why. I think likely a string
        # comparison might be best instead.
        """Test single consolidated district containing multiple SCCs connected by latent confounders."""
        # Consolidated district {B, C, D} has two SCCs: {B} and {C, D}
        # SCC {B}: P(B | A)
        # SCC {C, D}: P(C, D | A, B)
        # Expected = Product of both SCC distributions
        graph = NxMixedGraph.from_edges(
            directed=[(A, B), (B, B), (C, D), (D, C)], undirected=[(B, C)]
        )
        district = {B, C, D}
        apt_order = [A, B, C, D]

        result = initialize_district_distribution(graph, district, apt_order)

        expected = Product.safe([P(A, B) / P(A), P(A, B, C, D) / P(A, B)])

        # expected = product of distributions for each SCC
        self.assert_expr_equal(expected, result, ordering=apt_order)

    def test_two_single_node_sccs_with_confounder(self) -> None:
        # NOTE - I'm not sure if we need this test. Feel free to remove if redundant.
        """Test district with two single-node SCCs connected by latent confounder."""
        # Graph: X ↔ Y (bidirected edge only, no directed edges)
        # SCCs: {X} and {Y} - separate SCCs, one consolidated district
        graph = NxMixedGraph.from_edges(
            directed=[],  # No directed edges
            undirected=[(X, Y)],
        )

        district = {X, Y}
        apt_order = [X, Y]

        result = initialize_district_distribution(graph, district, apt_order)

        # Expected: Joint distribution (no predecessors, ancestral district)
        # According to Lemma 9.7, ancestral district = simple joint
        expected = Product.safe([P(X), P(X, Y) / P(X)])

        self.assert_expr_equal(expected, result)

    def test_single_scc_no_predecessors(self) -> None:
        """Test district with single SCC and no predecessors."""
        graph = NxMixedGraph.from_edges(directed=[(C, D), (D, C)])
        district = {C, D}
        apt_order = [C, D]

        result = initialize_district_distribution(graph, district, apt_order)
        expected = P(C, D)  # expected simplified expression = P(C, D) - simple joint distribution

        self.assert_expr_equal(expected, result)

    def test_self_loop_treated_as_scc(self) -> None:
        """Test that a self-loop is treated as single-node SCC."""
        # B should be treated as single-node SCC with predecessor A
        # Expected: P(B | A) as P(A, B) / P(A)
        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, B), (B, C)])
        district = {B}
        apt_order = [A, B, C]

        result = initialize_district_distribution(graph, district, apt_order)
        expected = P(A, B) / P(A)  # expected simplified expression = P(B | A)

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
            (None, {A}, "Outcomes must be a set.", "outcomes is None"),
            ({B}, None, "Interventions must be a set.", "interventions is None"),
            ([B], {A}, "set", "outcomes is list not set."),
            ({B}, [A], "set", "interventions is list not set."),
        ]
        for outcomes, interventions, error_pattern, description in parameters:
            with self.subTest(msg=description):
                with self.assertRaisesRegex(TypeError, error_pattern):
                    cyclic_id(graph, outcomes, interventions)  # type:ignore

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

    def test_ancestral_closure_integration(self) -> None:
        """Line 3: Verify ancestral closure through the final result structure."""
        # Query: P(Z | do(R))
        # After we remove R, ancestral closure H = {X, Y, Z}
        # Expected: Result contains only variables from H
        graph = NxMixedGraph.from_edges(directed=[(R, X), (X, Y), (Y, X), (Y, Z)])
        outcomes = {Z}
        interventions = {R}

        result = cyclic_id(graph, outcomes, interventions)

        # verify that the result only contains variables from the expected H = {X, Y, Z}
        result_vars = result.get_variables()
        expected_h = {X, Y, Z}
        self.assertTrue(result_vars.issubset(expected_h | {R}))

    def test_intervention_nodes_removed_from_ancestral_closure(self) -> None:
        """Intervention nodes shouldn't appear in the ancestral closure set H."""
        # Query: P(Y | do(R))
        # After removing R, ancestral closure H = {X, Y} - R should not appear in final result
        # Expected: R is not in the ancestral closure set
        graph = NxMixedGraph.from_edges(directed=[(R, X), (X, Y), (Y, X)])
        outcomes = {Y}
        interventions = {R}

        result = cyclic_id(graph, outcomes, interventions)

        # R should not appear in the final result structure
        # (except possible as a conditioning variable from do(R))
        result.get_variables()

        self.assertIsInstance(result, Expression)

    def test_ancestral_closure_computed_with_cycles(self) -> None:
        """Ancestral closure set correctly computed in presence of cycles."""
        # Query: P(D | do(A))
        # Ancestral closure H = {B, C, D} despite cycle between B and C
        # Expected: The algorithm should handle the cyclic ancestral closure.
        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C), (C, B), (C, D)])
        outcomes = {D}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)

    def test_empty_ancestral_closure(self) -> None:
        """Handle case where ancestral closure H is empty after removing interventions."""
        # Query: P(C | do(A))
        # After removing A, H = An({D}) = {D}, which means no ancestors
        # Expected: Algorithm should handle empty ancestral closure.

        graph = NxMixedGraph.from_edges(directed=[(A, B)])
        graph.add_node(C)
        outcomes = {C}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)

    def test_outcomes_equals_ancestral_closure(self) -> None:
        # Query P(D | do(B, C))
        # after removing {B, C}: H = An({D}) = {D}
        # Expected: ancestral closure = outcomes and minimal marginalization is needed.
        """Handles the case where the ancestral closure set = the outcomes. (Outcomes have no other ancestors)."""
        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C)])
        outcomes = {C}
        interventions = {B}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)

    # ------ Testing Line 4 ------------------------------------
    def test_single_consolidated_district(self) -> None:
        """Tests single consolidated district in ancestral closure."""
        # Query = P(D | do(A)) = identifiable
        # Ancestral closure H = {B, C, D} forms single consolidated district
        # Districts: {B, C} and {D}
        # Expected: Algorithm identifies and processes districts

        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, B), (B, C), (C, B), (C, D)])
        outcomes = {D}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)

    def test_multiple_consolidated_districts(self) -> None:
        """Tests multiple consolidated districts in ancestral closure."""
        # Query: P(E | do(A))
        # H = {B, C, D, E}
        # Districts: {B, C} (cycle), {D, E} (cycle)
        # Expected: Algorithm processes multiple separate districts
        graph = NxMixedGraph.from_edges(
            directed=[(A, B), (B, B), (B, C), (C, B), (C, D), (D, E), (E, D)]
        )
        outcomes = {E}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)

    def test_ancestral_closure_forms_single_cycle(self) -> None:
        """Entire ancestral closure set forms one large cycle (single district)."""
        # Query: P(D | do(A))
        # H = {B, C, D} forms one large consolidated district
        # Expected: Identifiable (A external to cycle)
        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C), (C, B), (C, D), (D, C)])
        outcomes = {D}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)

    def test_districts_with_latent_confounders(self) -> None:
        """Consolidated districts with bidirected edges (Latent confounders)."""
        # NOTE - I'm sure there is a way to condense these parameters?

        # Query: P(D | do(A))
        # {B, C} form consolidated district via bidirected edge B ↔ C
        # Districts: {B, C} (confounded), {D} (single node)
        # Expected: Identifiable (A external to district)
        graph = NxMixedGraph.from_edges(directed=[(A, B), (C, D)], undirected=[(B, C)])

        outcomes = {D}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)

    def test_mixed_cycles_and_confounders(self) -> None:
        """Districts formed by both cycles and latent confounders."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (W, X),
                (X, Y),
                (Y, X),
                (Z, W),
            ],
            undirected=[(Y, Z)],
        )

        outcomes = {Z}
        interventions = {W}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)

    # ------ Testing Lines 5-8 --------------------------------------------

    def test_recursive_idcd_call_and_failure_propagation(self) -> None:
        """Testing Lines 5-8 in the algorithm: IDCD initialization, recursive calls, and failure propagation."""
        test_cases = [
            # (graph_edges, outcomes, interventions, should_fail, description)
            # Line 5: Successful IDCD call
            # Query: P(C | do(A))
            # Expected to be Identifiable
            ([(A, B), (B, B), (B, C)], {C}, {A}, False, "identifiable chain"),
            # Line 5: Successful with cycle but external intervention
            # Query: P(D | do(A))
            # Expected: Identifiable
            ([(A, B), (B, C), (C, B), (C, D)], {D}, {A}, False, "identifiable with cycle"),
            # Lines 6-8: IDCD fails, propagates Unidentifiable
            # Query: P(Y | do(X))
            # Expected: Unidentifiable
            ([(X, Y), (Y, X)], {Y}, {X}, True, "unidentifiable - cause and effect in same SCC"),
            # Lines 6-8: Another failure case
            # QUery: P(B | do(A))
            (
                [(A, A), (A, B), (B, A)],
                {B},
                {A},
                True,
                "unidentifiable - intervention in same SCC as outcome",
            ),
        ]

        for edges, outcomes, interventions, should_fail, description in test_cases:
            with self.subTest(msg=description):
                graph = NxMixedGraph.from_edges(directed=edges)

                if should_fail:
                    with self.assertRaises(Unidentifiable):
                        cyclic_id(graph, outcomes, interventions)
                else:
                    result = cyclic_id(graph, outcomes, interventions)
                    self.assertIsInstance(result, Expression)

    # ------ Testing Line 10 --------------------------------------------

    def test_single_district_no_product_wrapper(self) -> None:
        """Single district should reutrn distribution directly without Product wrapper."""
        # Query: P(C | do(A))
        # H = {B, C} forms single consolidated district
        # Expected: Q[H] returned directly without Product

        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C), (C, B)])

        outcomes = {C}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)

        # should be an Expression, but not necessarily wrapped in Product
        self.assertIsInstance(result, Expression)

    def test_multiple_disjoint_districts_product(self) -> None:
        """Multiple districts should return Product of district distributions."""
        # Query: P(D | do(A))
        # H = {B, C, D} contains two districts: {B} (self-loop) and {C} (single)
        # Expected: Tensor product of both district distributions

        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, B), (A, C), (B, D), (C, D)])

        outcomes = {D}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)

        # verifying both B and C appear in result
        result_vars = result.get_variables()
        self.assertIn(B, result_vars)
        self.assertIn(C, result_vars)
        self.assertIn(D, result_vars)

    def test_product_contains_all_ancestral_closure_variables(self) -> None:
        """Product must contain all variables from ancestral closure H which is also prepared for marginalization in Line 11."""
        # Query: P(Z | do(R))
        # H = {X, Y, Z}
        # Expected: Q[H] contains all of {X, Y, Z} for marginalization in Line 11
        graph = NxMixedGraph.from_edges(directed=[(R, X), (X, Y), (Y, X), (Y, Z)])

        outcomes = {Z}
        interventions = {R}

        result = cyclic_id(graph, outcomes, interventions)

        # distribution should contain all variables in H = {X, Y, Z}
        result_vars = result.get_variables()
        expected_h = {X, Y, Z}
        self.assertTrue(expected_h.issubset(result_vars))

    def test_mixed_district_types(self) -> None:
        """Tests districts formed by cycles and latent confounders both included in product."""
        # Query: P(C, Y | do(A))
        # H contains TWO district types:
        #   - {B, C} formed by directed feedback cycle
        #   - {X, Y} formed by bidirected edge (latent confounder)
        # Expected: Product includes both district types

        graph = NxMixedGraph.from_edges(
            directed=[(A, B), (B, C), (C, B), (A, X)], undirected=[(X, Y)]
        )

        outcomes = {C, Y}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)

        # both districts should be represented
        result_vars = result.get_variables()
        self.assertIn(B, result_vars)  # From cycle district
        self.assertIn(C, result_vars)  # From cycle district
        self.assertIn(X, result_vars)  # From confounder district
        self.assertIn(Y, result_vars)  # From confounder district

    # ---- Testing Line 11: Marginalization --------------------------------------------

    def test_no_marginalization_ancestral_closure_equals_outcomes(self) -> None:
        """Tests that no marginalization occurs when ancestral closure H = outcomes Y."""
        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C), (C, B), (C, D)])
        outcomes = {D}
        interventions = {B, C}

        result = cyclic_id(graph, outcomes, interventions)

        # result should be an Expression
        self.assertIsInstance(result, Expression)

        # result should only contain D
        result_vars = result.get_variables()
        self.assertIn(D, result_vars)

    def test_marginalize_single_variable(self) -> None:
        """Tests that line 11 should marginalize out single variable from H (ancestral closure)."""
        # Query: P(C | do(A))
        # H = {B, C}, Y = {C}
        # Should marginalize out {B}
        # Expected: Sum over B

        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, B), (B, C), (C, B)])

        outcomes = {C}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)
        result_vars = result.get_variables()
        self.assertIn(C, result_vars)

    def test_marginalize_multiple_variables(self) -> None:
        """Tests that line 11 should marginalize out multiple variables from H (ancestral closure)."""
        # Query: P(D | do(A))
        # H = {B, C, D}, Y = {D}
        # Should marginalize out {B, C}
        # Expected: Sum over B and C

        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C), (C, B), (C, D)])

        outcomes = {D}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)
        result_vars = result.get_variables()
        self.assertIn(D, result_vars)

    def test_final_result_contains_only_outcomes(self) -> None:
        """Final result after marginalization contains only outcome variables Y."""
        # Query: P(D | do(A))
        # H = {B, C, D}, Y = {D}
        # Should marginalize out {B, C}
        # Expected: Result focuses on D

        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, C), (C, B), (C, D)])
        outcomes = {D}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)

        # After marginalization, D should be in result
        result_vars = result.get_variables()
        self.assertIn(D, result_vars)

    def test_marginalize_with_self_loops(self) -> None:
        """Test marginalizing with self-loops."""
        # Query: P(D | do(A))
        # H = {B, C, D}, Y = {D}
        # Should marginalize out {B, C}
        # Expected: Handles self-loops correctly
        graph = NxMixedGraph.from_edges(directed=[(A, B), (B, B), (B, C), (C, D)])
        outcomes = {D}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)
        result_vars = result.get_variables()
        self.assertIn(D, result_vars)

    def test_marginalize_with_latent_confounders(self) -> None:
        """Test marginalizing with latent confounders."""
        # Query: P(D | do(A))
        # H = {B, C, D}, Y = {D}
        # Should marginalize out {B, C} (confounded district)
        # Expected: Correctly marginalizes confounded variables
        graph = NxMixedGraph.from_edges(directed=[(A, B), (C, D)], undirected=[(B, C)])
        outcomes = {D}
        interventions = {A}

        result = cyclic_id(graph, outcomes, interventions)
        self.assertIsInstance(result, Expression)
        result_vars = result.get_variables()
        self.assertIn(D, result_vars)
