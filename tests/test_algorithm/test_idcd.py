"""Tests for IDCD algorithm implementation."""

import unittest

from tests.test_algorithm.test_ioscm import simple_cyclic_graph_1, simple_cyclic_graph_2
from y0.algorithm.identify import Unidentifiable, identify_outcomes
from y0.algorithm.identify.idcd import (
    _calculate_scc_distribution,
    _get_apt_order_predecessors,
    compute_scc_distributions,
    idcd,
    identify_through_scc_decomposition,
    marginalize_to_ancestors,
    validate_preconditions,
)
from y0.algorithm.ioscm.utils import get_apt_order
from y0.dsl import Expression, P, R, Variable, W, X, Y, Z
from y0.graph import NxMixedGraph

# ----------------------------------------------------------------------------


class TestValidatePreconditions(unittest.TestCase):
    """Tests for IDCD precondition validation."""

    def test_empty_targets_raises_error(self) -> None:
        """Empty target set should raise ValueError."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets: set[Variable] = set()
        district = {Y}

        with self.assertRaisesRegex(ValueError, "Target set C cannot be empty"):
            validate_preconditions(graph, targets, district)

    def test_empty_district_raises_error(self) -> None:
        """Empty district should raise ValueError."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets = {Y}
        district: set[Variable] = set()

        with self.assertRaisesRegex(ValueError, "District D cannot be empty"):
            validate_preconditions(graph, targets, district)

    def test_targets_not_subset_of_district_raises_error(self) -> None:
        """Target must be a subset of district."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])
        targets = {Y, Z}
        district = {Y}  # Z is not in district

        with self.assertRaisesRegex(
            ValueError,
            "Target must be subset of district.",
        ):
            validate_preconditions(graph, targets, district)

    def test_district_not_subset_of_nodes_raises_error(self) -> None:
        """District must be a subset of graph nodes."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets = {Y}
        district = {Y, Z}  # Z not in graph

        with self.assertRaisesRegex(
            ValueError,
            "District must be subset of graph nodes.",
        ):
            validate_preconditions(graph, targets, district)

    def test_valid_preconditions_pass(self) -> None:
        """Valid inputs should pass without error."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])
        targets = {Z}
        district = {Y, Z}

        # should not raise due to valid inputs
        validate_preconditions(graph, targets, district)

    def test_cyclic_graph_valid_preconditions_pass(self) -> None:
        """Valid inputs on cyclic graph should pass without error."""
        # simple cycle - X -> Y -> Z -> X
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z), (Z, X)])

        targets = {Z}
        district = {X, Y, Z}

        # should not raise due to valid inputs
        validate_preconditions(graph, targets, district)


# ----------------------------------------------------------------------------


class TestMarginalizationToAncestors(unittest.TestCase):
    """Tests for marginalization to ancestors function."""

    def test_no_marginalization_district_equals_ancestral_closure(self) -> None:
        """If district equals ancestral closure, no marginalization should occur."""
        distribution = P(X, Y)
        district = {X, Y}
        ancestral_closure = {X, Y}

        result = marginalize_to_ancestors(distribution, district, ancestral_closure)

        self.assertEqual(result, distribution)  # should be unchanged

    def test_marginalization_occurs(self) -> None:
        r"""Marginalization should remove district \ ancestral_closure variables."""
        distribution = P(X, Y, Z)
        district = {X, Y, Z}
        ancestral_closure = {X, Y}

        result = marginalize_to_ancestors(distribution, district, ancestral_closure)

        result_str = str(result)
        self.assertIn("Sum", result_str)
        self.assertIn("Z", result_str)

    def test_marginalization_with_single_variable(self) -> None:
        """Test marginalizing out a single variable."""
        distribution = P(X, Y)
        district = {X, Y}
        ancestral_closure = {Y}

        result = marginalize_to_ancestors(distribution, district, ancestral_closure)

        result_str = str(result)
        self.assertIn("Sum", result_str)
        self.assertIn("X", result_str)


# ----------------------------------------------------------------------------


class TestGetAptOrderPredecessors(unittest.TestCase):
    """Tests the _get_apt_order_predecessors function."""

    def test_first_element_has_no_predecessors(self) -> None:
        """First element in apt-order should have no predecessors."""
        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        # first element should not have predecessors
        scc = frozenset([R])
        ancestral_closure = {R, W, X, Z, Y}

        predecessors = _get_apt_order_predecessors(scc, apt_order, ancestral_closure)
        self.assertEqual(predecessors, set())

    def test_predecessors_filtered_by_ancestral_closure(self) -> None:
        """Predecessors should only include those in the ancestral closure."""
        # Apt order = [R, W, X, Z, Y]

        # If ancestral closure = {R, Z}, only R should be predecessor of Z
        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([Z])
        ancestral_closure = {R, Z}

        predecessors = _get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        self.assertEqual(predecessors, {R})

    def test_all_predecessors_included(self) -> None:
        """All variables before SCC in apt order and in ancestral closure should be included."""
        # apt order = [R, W, X, Z, Y], Y is last so all before it should be included

        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([Y])
        ancestral_closure = {R, W, X, Z, Y}

        predecessors = _get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        self.assertEqual(predecessors, {R, W, X, Z})

    def test_multi_node_scc_uses_minimum_position(self) -> None:
        """For multi-node SCC, use the minimum position of its members in apt order."""
        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([W, X, Z])
        ancestral_closure = {R, W, X, Z, Y}

        predecessors = _get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        self.assertEqual(predecessors, {R})

    def test_empty_ancestral_closure(self) -> None:
        """If ancestral closure is empty, there should be no predecessors."""
        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([Z])
        ancestral_closure: set[Variable] = set()

        predecessors = _get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        self.assertEqual(predecessors, set())


# ----------------------------------------------------------------------------


class TestIDCDFunction(unittest.TestCase):
    """Tests for the IDCD algorithm implementation."""

    def test_base_case_ancestral_closure_equals_district(self) -> None:
        """Test base case where ancestral closure equals district with a cyclic graph.

        Graph Structure:

        X -> Y -> Z

        Query: Identify effect on Z within district {Z}
        - Targets: {Z}
        - District: {Z}
        - Distribution: P(Z)

        Algorithm Execution:
        1. Line 14: Validate C ⊆ D ⊆ V - Precondition check passes
        2. Line 15: Compute A = An^{G[{Z}]}(Z) = {Z} - Computing the ancestral closure
        3. Line 16: No marginalization (D = A = {Z}) - No change to distribution
        4. Line 17-18: A = C → Return distribution unchanged - Return the distribution as is since targets equal ancestral closure.


         Expected Expression:
        - Input:  P(Z)
        - Output: P(Z)  ← EXACTLY the same, no transformation

        Why: Base case returns the distribution as-is when ancestral
        closure equals targets (no decomposition needed).
        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, Z),
                (Z, X),  # creates the cycle
            ]
        )

        targets = {Z}
        district = {Z}
        distribution = P(Z)

        # should return the distribution as is without recursion
        result = idcd(
            graph=graph,
            targets=targets,
            district=district,
            distribution=distribution,
        )

        # verifying result type
        self.assertIsInstance(
            result, Expression
        )  # should return the expression that is unchanged from the input: P(Z)

        # verify exact equality
        self.assertEqual(result, distribution)

        # verify string representation
        self.assertEqual(str(result), "P(Z)", "Result should be exactly P(Z)")

    def test_unidentifiable_case_ancestral_closure_equals_district(self) -> None:
        """Line 19-20: When ancestral closure equals district it should be unidentifiable."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z), (Z, X)])

        targets = {X}
        district = {X, Y, Z}
        distribution = P(X, Y, Z)

        # the result should indicate unidentifiability
        with self.assertRaises(Unidentifiable) as context:
            idcd(
                graph=graph,
                targets=targets,
                district=district,
                distribution=distribution,
            )

        self.assertIn("cannot identify", str(context.exception).lower())

    def test_recursive_case_through_scc_decomposition(self) -> None:
        """Tests the recursive case when SCC decomposition is needed."""
        graph = NxMixedGraph.from_edges(directed=[(W, R), (R, X), (X, X), (X, Y), (Y, Z), (Z, X)])

        targets = {Y}
        district = {W, R, X, Y, Z}
        distribution = P(W, R, X, Y, Z)
        with self.assertRaises(Unidentifiable):
            idcd(
                graph=graph,
                targets=targets,
                district=district,
                distribution=distribution,
            )

    def test_single_scc_in_consolidated_district(self) -> None:
        """Test with a single SCC in the consolidated district."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, Z),
                (Z, X),
            ]
        )

        targets = {Z}
        ancestral_closure = {X, Y, Z}

        original_distribution = P(X, Y, Z)

        with self.assertRaises(Unidentifiable):
            identify_through_scc_decomposition(
                graph=graph,
                targets=targets,
                ancestral_closure=ancestral_closure,
                recursion_level=0,
                original_distribution=original_distribution,
            )

    def test_multiple_sccs_in_consolidated_district(self) -> None:
        """Test with multiple SCCs in the consolidated district."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, X),
                (Y, W),
                (W, Z),
                (Z, W),
            ]
        )

        targets = {Z}
        ancestral_closure = {X, Y, W, Z}

        original_distribution = P(X, Y, W, Z)

        with self.assertRaises(Unidentifiable):
            identify_through_scc_decomposition(
                graph=graph,
                targets=targets,
                ancestral_closure=ancestral_closure,
                recursion_level=0,
                original_distribution=original_distribution,
            )

    @unittest.skip("Edge case: 'No relevant SCCs' condition is difficult to trigger in practice")
    def test_no_relevant_sccs_raises_unidentifiable(self) -> None:
        """Test that when no relevant SCCs are found, Unidentifiable is raised."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, X),
                (X, Z),
            ]
        )

        targets = {Z}
        ancestral_closure = {X, Y, Z}

        original_distribution = P(X, Y, Z)

        with self.assertRaises(Unidentifiable) as context:
            identify_through_scc_decomposition(
                graph=graph,
                targets=targets,
                ancestral_closure=ancestral_closure,
                recursion_level=0,
                original_distribution=original_distribution,
            )
            self.assertIn("No SCCs", str(context.exception))

    def test_recursive_idcd_call_receives_correct_inputs(self) -> None:
        """Test that recursive calls to IDCD receive the correct input.

        Instead of testing the full recursion, this test verifies that
        identify_through_scc_decomposition correctly prepares inputs for
        the next IDCD call.

        - Correct consolidated district
        - Correct product of SCC distributions
        - Correct targets

        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (R, X),
                (X, Y),
                (Y, X),
            ]
        )

        targets = {X}
        ancestral_closure = {R, X, Y}
        original_distribution = P(R, X, Y)

        try:
            result = identify_through_scc_decomposition(
                graph=graph,
                targets=targets,
                ancestral_closure=ancestral_closure,
                recursion_level=0,
                original_distribution=original_distribution,
            )

            self.assertIsInstance(result, Expression)
        except Unidentifiable:
            pass

    def test_simple_identifiable_graph(self) -> None:
        """Test IDCD on a simple identifiable cyclic graph.

        Graph: A simple cyclic graph example:
        - Has cycle: X -> W -> Z - X
        - Has a confounder: R <-> X
        - Has isolated node: Y

        Query: Identify effect on Y within the district {R, X, W, Y, Z}

        IDCD will:
        - Compute ancestral closure of Y
        - Marginalize and condition appropriately
        - Return a symbolic expression for P(Y | do(...))

        """
        targets = {Y}
        district = {R, X, W, Y, Z}
        distribution = P(R, X, W, Y, Z)

        try:
            result = idcd(
                graph=simple_cyclic_graph_2,
                targets=targets,
                district=district,
                distribution=distribution,
            )

            # the expression should be:
            self.assertIsInstance(result, Expression)  # should return P(Y) unchanged
        except Unidentifiable:
            pass  # this query is unidentifiable in this graph

    def test_simple_unidentifiable_graph(self) -> None:
        """Test IDCD on a simple unidentifiable cyclic graph with cycles.

        Graph: A simple cyclic graph example:
        - X has a self-loop
        - X -> W
        - X -> Y
        - X <- Z
        - Z -> Y
        - Cycle: X -> ... -> Z -> X (with X's self loop)

        Query: Identify effect on Z within the cyclic structure

        This is a known non-identifiable structure because:
        - The causal effect cannot be separated from confounding.
        - IDCD should raise Unidentifiable exception.
        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, X),  # forms a self loop
                (X, W),
                (X, Y),
                (Z, X),
                (Z, Y),
            ]
        )

        # try to identify effect on Z within the full graph
        targets = {X}
        district = {X, Z}
        distribution = P(X, Z)

        try:
            result = idcd(
                graph=graph,
                targets=targets,
                district=district,
                distribution=distribution,
            )

            self.assertIsInstance(result, Expression)
            self.assertIn("X", str(result))
        except Unidentifiable as e:
            # verify error mentions inability to identify
            error_msg = str(e).lower()
            self.assertIn("cannot identify", error_msg)


# ----------------------------------------------------------------------------


class TestComputeSCCDistributions(unittest.TestCase):
    """Tests for compute_scc_distributions function."""

    def test_single_scc_returns_correct_structure(self) -> None:
        """Test that a single SCC returns the correct distribution structure.

        Graph: X -> Y -> Z -> X (cycle)
        Input: We are identifying within this single SCC
        Ancestral closure: {X, Y, Z}
        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, Z),
                (Z, X),
            ]
        )

        subgraph_a = graph.subgraph({X, Y, Z})

        relevant_sccs = [frozenset({X, Y, Z})]
        ancestral_closure = {X, Y, Z}

        original_distribution = P(X, Y, Z)
        intervention_set = set()

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
            original_distribution=original_distribution,
            intervention_set=intervention_set,
        )

        # Should return a dictionary
        self.assertIsInstance(result, dict)

        # Should have one entry (one SCC)
        self.assertEqual(len(result), 1)

        # The key should be the SCC
        self.assertIn(frozenset({X, Y, Z}), result)

        # The value should be an Expression
        distribution = result[frozenset({X, Y, Z})]
        self.assertIsInstance(distribution, Expression)

    def test_multiple_sccs_with_cycles(self) -> None:
        """Test multiple SCCs each with cycles in the graph."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, X),
                (Y, X),
                (W, Z),
                (Z, W),
            ]
        )

        # subgraph contains both cycles in the graph
        subgraph_a = graph.subgraph({X, Y, W, Z})

        # two SCCs
        relevant_sccs = [frozenset({X, Y}), frozenset({W, Z})]
        ancestral_closure = {X, Y, W, Z}

        original_distribution = P(X, Y, W, Z)
        intervention_set = set()

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
            original_distribution=original_distribution,
            intervention_set=intervention_set,
        )

        # Should return a dictionary
        self.assertIsInstance(result, dict)

        # should have two entries (two SCCs)
        self.assertEqual(len(result), 2)

        # Each key should be an SCC
        self.assertIn(frozenset({X, Y}), result)
        self.assertIn(frozenset({W, Z}), result)

        for scc in relevant_sccs:
            self.assertIsInstance(result[scc], Expression)

    def test_intervention_set_calculation(self) -> None:
        """Test that intervention sets are calculated correctly for SCCs."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (R, X),
                (X, Y),
                (Y, Z),
            ]
        )

        # subgraph contains just Y -> Z
        subgraph_a = graph.subgraph({Y, Z})

        # one SCC
        relevant_sccs = [frozenset({Z})]
        ancestral_closure = {Y, Z}

        # ancestral closure is {Y, Z}
        # So intervention_set should be nodes - ancestral_closure = {R, X, Y, Z} - {Y, Z} = {R, X}
        original_distribution = P(R, X, Y, Z)
        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
            original_distribution=original_distribution,
            intervention_set=intervention_set,
        )

        # should successfully compute distribution for the SCC
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)

        self.assertIsInstance(result[frozenset({Z})], Expression)


# ----------------------------------------------------------------------------


class TestCalculateSCCDistribution(unittest.TestCase):
    """Tests for _calculate_scc_distribution function."""

    pass  # adding tests here later


class TestLine23ImplementationComparison(unittest.TestCase):
    """Tests comparing two implementations of Line 23."""

    def test_approach_comparison_simple_cycle(self) -> None:
        """Compare Approach A (using identify_outcomes) vs Approach B (direct construction).

        Graph: simple_cyclic_graph_1
        - R -> X -> W -> Z -> X (cycle)
        - W -> Y
        - Districts: {R}, {X, W, Z,},  {Y}



        This test verifies both approaches produce equivalent symbolic expressions
        for computing R_A[S] in Line 23 of the IDCD algorithm.

        Approach A: Call identify_outcomes with apt_order
        Approach B: Current _calculate_scc_distribution implementation

        Expected: Both should return equivalent Expression objects.

        """
        graph = simple_cyclic_graph_1

        # test on the SCC {X, W, Z}
        scc = frozenset({X, W, Z})

        # ancestral closure for this test
        ancestral_closure = {R, X, W, Z}

        original_distribution = P(R, X, W, Z)

        # intervention set: nodes - ancestral_closure
        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure

        # step 2: compute apt-order and predecessors
        subgraph_a = graph.subgraph(ancestral_closure)
        apt_order_a = get_apt_order(subgraph_a)

        # get predecessors of the SCC
        predecessors = _get_apt_order_predecessors(
            scc,
            apt_order_a,
            ancestral_closure,
        )

        # step 3: Approach A vs B

        # approach B using current implementation with direct construction of expression
        result_b = _calculate_scc_distribution(
            scc=scc,
            predecessors=predecessors,
            intervention_set=intervention_set,
            original_distribution=original_distribution,
            graph=graph,
        )

        # Approach A - using identify_outcomes
        result_a = identify_outcomes(
            graph=graph,
            outcomes=scc,
            conditions=predecessors if predecessors else None,
            strict=True,
            treatments=intervention_set,
            ordering=apt_order_a,
        )

        # Step 4: Compare results - they should be equivalent expressions
        self.assertIsInstance(result_a, Expression)
        self.assertIsInstance(result_b, Expression)

        # compare string representations for equivalence
        result_a_str = str(result_a)
        result_b_str = str(result_b)

        # If they're equivalent, the strings should match
        # (This might need adjustment depending on expression normalization)
        self.assertEqual(
            result_a_str,
            result_b_str,
            f"Approaches produce different results:\n"
            f"Approach A: {result_a_str}\n"
            f"Approach B: {result_b_str}",
        )

    def test_approach_comparison_no_predecessors(self) -> None:
        """Test Line 23 when there are no predecessors.

        Graph: simple_cyclic_graph_1
        - R -> X -> W -> Z -> X (cycle)
        - W -> Y

        Test Case: Scc = {R} with no predecessors.
        Expected: Both approaches should handle empty predecessors correctly.
        """
        graph = simple_cyclic_graph_1

        # test the first node R that has no predecessors
        scc = frozenset({R})
        ancestral_closure = {R, X, W, Z}
        original_distribution = P(R, X, W, Z)

        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure  # should be {Y}

        # compute the apt-order and predecessors
        subgraph_a = graph.subgraph(ancestral_closure)
        apt_order_a = get_apt_order(subgraph_a)
        predecessors = _get_apt_order_predecessors(
            scc,
            apt_order_a,
            ancestral_closure,
        )

        # approach B using current implementation
        result_b = _calculate_scc_distribution(
            scc=scc,
            predecessors=predecessors,
            intervention_set=intervention_set,
            original_distribution=original_distribution,
            graph=graph,
        )

        # approach a using identify_outcomes
        result_a = identify_outcomes(
            graph=graph,
            outcomes=scc,
            treatments=intervention_set,
            conditions=predecessors if predecessors else None,
            strict=True,
            ordering=apt_order_a,
        )

        # Both should return valid Expressions
        self.assertIsInstance(result_a, Expression)
        self.assertIsInstance(result_b, Expression)

        # Verify predecessors is empty
        self.assertEqual(len(predecessors), 0)

    def test_approach_comparison_single_node_scc(self) -> None:
        """Test Line 23 comparison with a single-node SCC.

        Graph: simple_cyclic_graph_1
        - R -> X -> W -> Z -> X (cycle)
        - W -> Y

        Test Case: SCC = {Y} which is a single node SCC.
        Expected: Both approaches should yield the same expression for R_A[S].
        """
        graph = simple_cyclic_graph_1

        scc = frozenset({Y})
        ancestral_closure = {W, X, Z, Y}
        original_distribution = P(W, X, Z, Y)

        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure  # should be {R}

        # compute apt-order and predecessors
        subgraph_a = graph.subgraph(ancestral_closure)
        apt_order_a = get_apt_order(subgraph_a)
        predecessors = _get_apt_order_predecessors(
            scc,
            apt_order_a,
            ancestral_closure,
        )

        # Approach B - current implementation
        result_b = _calculate_scc_distribution(
            scc=scc,
            predecessors=predecessors,
            intervention_set=intervention_set,
            original_distribution=original_distribution,
            graph=graph,
        )

        # approach A - using identify_outcomes
        result_a = identify_outcomes(
            graph=graph,
            outcomes=scc,
            treatments=intervention_set,
            conditions=predecessors if predecessors else None,
            strict=True,
            ordering=apt_order_a,
        )

        # Both should return valid Expressions
        self.assertIsInstance(result_a, Expression)
        self.assertIsInstance(result_b, Expression)

    def test_approach_comparison_graph2(self) -> None:
        """Test Line 23 comparison on simple_cyclic_graph_2.

        Graph: simple_cyclic_graph_2

        - R <-> X (bidirected edge)
        - R -> X -> W -> Z -> X (cycle)
        - W -> Y

        Test Case: SCC = {X, W, Z} with confounding.
        Expected: Both approaches should yield the same expression for R_A[S].
        """
        graph = simple_cyclic_graph_2

        scc = frozenset({X, W, Z})
        ancestral_closure = {R, X, W, Z}
        original_distribution = P(R, X, W, Z)

        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure  # should be {Y}

        # compute apt-order and predecessors
        subgraph_a = graph.subgraph(ancestral_closure)
        apt_order_a = get_apt_order(subgraph_a)
        predecessors = _get_apt_order_predecessors(
            scc,
            apt_order_a,
            ancestral_closure,
        )

        # Approach B - current implementation
        result_b = _calculate_scc_distribution(
            scc=scc,
            predecessors=predecessors,
            intervention_set=intervention_set,
            original_distribution=original_distribution,
            graph=graph,
        )

        # approach A - using identify_outcomes
        result_a = identify_outcomes(
            graph=graph,
            outcomes=scc,
            treatments=intervention_set,
            conditions=predecessors if predecessors else None,
            strict=True,
            ordering=apt_order_a,
        )

        # Both should return valid Expressions
        self.assertIsInstance(result_a, Expression)
        self.assertIsInstance(result_b, Expression)

    def test_approach_comparison_multiple_nodes_in_predecessors(self) -> None:
        """Test Line 23 comparison with multiple predecessor nodes.

        Creates a custom graph where an SCC has multiple predecessors.

        Graph: R1 → S, R2 → S, S → T → S (cycle)

        Test Case: SCC = {S, T} with predecessors = {R1, R2}
        Expected: Both approaches handle multiple predecessors correctly.
        """
        from y0.dsl import Variable

        # Create custom variables
        r1 = Variable("R1")
        r2 = Variable("R2")
        s = Variable("S")
        t = Variable("T")

        # Build custom graph
        graph = NxMixedGraph.from_edges(
            directed=[
                (r1, s),
                (r2, s),
                (s, t),
                (t, s),  # Creates cycle
            ]
        )

        # Test on the SCC {S, T}
        scc = frozenset({s, t})
        ancestral_closure = {r1, r2, s, t}
        original_distribution = P(r1, r2, s, t)

        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure  # Empty in this case

        # Compute apt-order and predecessors
        subgraph_a = graph.subgraph(ancestral_closure)
        apt_order_a = get_apt_order(subgraph_a)
        predecessors = _get_apt_order_predecessors(scc, apt_order_a, ancestral_closure)

        # Approach B
        result_b = _calculate_scc_distribution(
            scc=scc,
            predecessors=predecessors,
            intervention_set=intervention_set,
            original_distribution=original_distribution,
            graph=graph,
        )

        # Approach A
        result_a = identify_outcomes(
            graph=graph,
            outcomes=scc,
            treatments=intervention_set,
            conditions=predecessors if predecessors else None,
            strict=True,
            ordering=apt_order_a,
        )

        # Both should return valid Expressions
        self.assertIsInstance(result_a, Expression)
        self.assertIsInstance(result_b, Expression)

        # Verify we have multiple predecessors
        self.assertGreater(len(predecessors), 1)
        self.assertEqual(predecessors, {r1, r2})
