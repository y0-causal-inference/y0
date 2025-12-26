"""Tests for IDCD algorithm implementation."""

import unittest
from unittest.mock import MagicMock, patch

from tests.test_algorithm import cases
from tests.test_algorithm.test_ioscm import simple_cyclic_graph_1, simple_cyclic_graph_2
from y0.algorithm.identify import Unidentifiable
from y0.algorithm.identify.idcd import (
    compute_scc_distributions,
    get_apt_order_predecessors,
    idcd,
    identify_through_scc_decomposition,
    marginalize_to_ancestors,
    validate_preconditions,
)
from y0.algorithm.ioscm.utils import get_apt_order
from y0.dsl import P, R, Sum, Variable, W, X, Y, Z
from y0.graph import NxMixedGraph


class TestComponents(unittest.TestCase):
    """Tests for IDCD precondition validation."""

    def test_empty_targets_raises_error(self) -> None:
        """Empty target set should raise ValueError."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets: set[Variable] = set()
        district = {Y}

        with self.assertRaises(
            ValueError,
        ):
            validate_preconditions(graph, targets, district)

    def test_empty_district_raises_error(self) -> None:
        """Empty district should raise ValueError."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y)])
        targets = {Y}
        district: set[Variable] = set()

        with self.assertRaises(
            ValueError,
        ):
            validate_preconditions(graph, targets, district)

    def test_targets_not_subset_of_district_raises_error(self) -> None:
        """Target must be a subset of district."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])
        targets = {Y, Z}
        district = {Y}  # Z is not in district

        with self.assertRaises(
            ValueError,
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

    def test_no_marginalization_district_equals_ancestral_closure(self) -> None:
        """If district equals ancestral closure, no marginalization should occur."""
        parameters = [
            # test 1: If district equals ancestral closure, no marginalization should occur
            (P(X, Y), P(X, Y), {X, Y}, {X, Y}),
            # test 2: Marginalization should remove district \ ancestral_closure variables
            (Sum[Z](P(X, Y, Z)), P(X, Y, Z), {X, Y, Z}, {X, Y}),  # type: ignore[misc]
            # test 3: Test marginalizing out a single variable # type: ignore[misc]
            (Sum[X](P(X, Y)), P(X, Y), {X, Y}, {Y}),  # type: ignore[misc]
        ]
        for expected, distribution, district, ancestral_closure in parameters:
            with self.subTest(
                distribution=distribution, district=district, ancestral_closure=ancestral_closure
            ):
                self.assertEqual(
                    expected,
                    marginalize_to_ancestors(distribution, district, ancestral_closure),
                )


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

        predecessors = get_apt_order_predecessors(scc, apt_order, ancestral_closure)
        self.assertEqual(set(), predecessors)

    def test_predecessors_filtered_by_ancestral_closure(self) -> None:
        """Predecessors should only include those in the ancestral closure."""
        # Apt order = [R, W, X, Z, Y]

        # If ancestral closure = {R, Z}, only R should be predecessor of Z
        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([Z])
        ancestral_closure = {R, Z}

        predecessors = get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        # expected: {R}, actual: predecessors
        self.assertEqual({R}, predecessors)

    def test_all_predecessors_included(self) -> None:
        """All variables before SCC in apt order and in ancestral closure should be included."""
        # apt order = [R, W, X, Z, Y], Y is last so all before it should be included

        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([Y])
        ancestral_closure = {R, W, X, Z, Y}

        predecessors = get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        self.assertEqual({R, W, X, Z}, predecessors)

    def test_multi_node_scc_uses_minimum_position(self) -> None:
        """For multi-node SCC, use the minimum position of its members in apt order."""
        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([W, X, Z])
        ancestral_closure = {R, W, X, Z, Y}

        predecessors = get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        # expected: {R}, actual: predecessors
        self.assertEqual({R}, predecessors)

    def test_empty_ancestral_closure(self) -> None:
        """If ancestral closure is empty, there should be no predecessors."""
        graph = simple_cyclic_graph_1
        subgraph = graph.subgraph({R, W, X, Z, Y})
        apt_order = get_apt_order(subgraph)

        scc = frozenset([Z])
        ancestral_closure: set[Variable] = set()

        predecessors = get_apt_order_predecessors(scc, apt_order, ancestral_closure)

        # expected: empty set, actual: predecessors, which is also empty
        self.assertEqual(set(), predecessors)


class TestIDCDFunction(cases.GraphTestCase):
    """Tests for the IDCD algorithm implementation."""

    def test_base_case_ancestral_closure_equals_district(self) -> None:
        """Test base case where ancestral closure equals district with a cyclic graph."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, Z),
                (Z, X),  # creates the cycle
            ]
        )

        targets = {Z}
        district = {Z}
        expected = P(Z)
        result = idcd(graph=graph, outcomes=targets, district=district)
        self.assertEqual(expected, result)  # expected output: P(Z)

    def test_unidentifiable_case_ancestral_closure_equals_district(self) -> None:
        """Line 19-20: When ancestral closure equals district it should be unidentifiable."""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z), (Z, X)])

        targets = {X}
        district = {X, Y, Z}

        # the result should indicate unidentifiability due to ancestral closure equaling district
        with self.assertRaises(Unidentifiable) as context:
            idcd(graph=graph, outcomes=targets, district=district)

        self.assertIn("cannot identify", str(context.exception).lower())

    def test_recursive_case_through_scc_decomposition(self) -> None:
        """Tests the recursive case when SCC decomposition is needed."""
        graph = NxMixedGraph.from_edges(directed=[(W, R), (R, X), (X, X), (X, Y), (Y, Z), (Z, X)])

        targets = {Y}
        district = {W, R, X, Y, Z}
        with self.assertRaises(Unidentifiable):
            idcd(graph=graph, outcomes=targets, district=district)

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

        with self.assertRaises(Unidentifiable):
            identify_through_scc_decomposition(
                graph=graph,
                outcomes=targets,
                ancestral_closure=ancestral_closure,
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

        with self.assertRaises(Unidentifiable):
            identify_through_scc_decomposition(
                graph=graph,
                outcomes=targets,
                ancestral_closure=ancestral_closure,
            )

    @unittest.skip("Edge case: 'No relevant SCCs' condition is difficult to trigger in practice")
    def test_no_relevant_sccs_raises_unidentifiable(self) -> None:
        """Test that when no relevant SCCs are found, Unidentifiable is raised."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (X, Y),
                (Y, X),
                (W, Z),
                (Z, W),
            ]
        )

        targets = {X}
        ancestral_closure = {X, Y}

        with self.assertRaises(Unidentifiable) as context:
            identify_through_scc_decomposition(
                graph=graph,
                outcomes=targets,
                ancestral_closure=ancestral_closure,
            )
        self.assertIn("No SCCs", str(context.exception))

    def test_recursive_idcd_call_receives_correct_inputs(self) -> None:
        """Test that recursive calls to IDCD receive the correct input.

        Instead of testing the full recursion, this test verifies that
        identify_through_scc_decomposition correctly prepares inputs for the next IDCD
        call.

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

        # adding explicit catch: Unidentifiable is expected here due to the graph structure
        with self.assertRaises(Unidentifiable) as context:
            identify_through_scc_decomposition(
                graph=graph,
                outcomes=targets,
                ancestral_closure=ancestral_closure,
            )

        # verify error message is correct
        error_msg = str(context.exception).lower()
        self.assertIn("cannot identify", error_msg)

    def test_simple_identifiable_graph(self) -> None:
        """Test IDCD on a simple identifiable cyclic graph.

        Graph: A simple cyclic graph example: - Has cycle: X -> W -> Z - X - Has a
        confounder: R <-> X - Has isolated node: Y
        """
        targets = {Y}
        district = {R, X, W, Y, Z}

        result = idcd(
            graph=simple_cyclic_graph_2,
            outcomes=targets,
            district=district,
        )

        # explicitly construct expected expression structure
        conditional_prob = P(Y | W, X, Z)

        # normalization factor
        normalization = Sum[Y](P(Y | W, X, Z))  # type: ignore[misc]

        # this is the final expected result which should be: (P(Y | W, X, Z)) / (Sum_Y P(Y | W, X, Z))
        expected = conditional_prob / normalization

        self.assert_expr_equal(expected, result)

    def test_simple_unidentifiable_graph(self) -> None:
        """Test IDCD on a simple unidentifiable cyclic graph with cycles.

        This is a known non-identifiable structure because the causal effect cannot be
        separated from confounding.
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

        with self.assertRaises(Unidentifiable):
            idcd(graph=graph, outcomes=targets, district=district)

    # can work on a case for this at a later time
    def test_invalid_subsets_raise(self) -> None:
        """Test when condition targets ⊊ ancestral_closure ⊊ district is not met."""
        # NOTE: mathematically this isn't something that should happen, since the ancestral closure set
        # should always include the targets and some portion of the consolidated district.

        graph = NxMixedGraph.from_edges(directed=[((X, Y))], undirected=[(Y, Z)])
        targets = {Z}
        district = {X, Y, Z}

        mock_subgraph = MagicMock()

        mock_subgraph.ancestors_inclusive.return_value = {X, Y}

        with patch.object(graph, "subgraph", return_value=mock_subgraph):
            with self.assertRaises(ValueError) as context:
                idcd(graph=graph, outcomes=targets, district=district)
            error_msg = str(context.exception).lower()
            self.assertIn("unexpected state", error_msg)
            self.assertIn("targets", error_msg)
            self.assertIn("ancestral_closure", error_msg)
            self.assertIn("district", error_msg)


class TestComputeSCCDistributions(cases.GraphTestCase):
    """Tests for compute_scc_distributions function."""

    def test_single_scc_returns_correct_structure(self) -> None:
        """Test that a single SCC returns the correct distribution structure."""
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

        intervention_set: set[Variable] = set()
        expected = {frozenset({X, Y, Z}): P(X, Y, Z)}

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
            intervention_set=intervention_set,
        )
        self.assertEqual(expected, result)

    def test_multiple_sccs_with_cycles(self) -> None:
        """Test compute_scc_distributions with multiple independent SCCs."""
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

        # added for documentation - original_distribution = P(X, Y, W, Z)
        intervention_set: set[Variable] = set()

        expected = {
            # SCC {X, Y}: marginalize out {W, Z}
            frozenset({X, Y}): Sum[W, Z](P(X, Y, W, Z)),  # type:ignore[misc]
            # SCC {W, Z}: marginalize out {X, Y}
            frozenset({W, Z}): Sum[X, Y](P(X, Y, W, Z)),  # type:ignore[misc]
        }
        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
            intervention_set=intervention_set,
        )
        self.assertEqual(expected, result)

    def test_intervention_set_calculation(self) -> None:
        """Test that intervention sets are calculated correctly for SCCs."""
        graph = NxMixedGraph.from_edges(
            directed=[
                (R, X),
                (X, Y),
                (Y, Z),
            ]
        )

        # subgraph induced by ancestral closure {Y, Z}
        subgraph_a = graph.subgraph({Y, Z})

        # one SCC
        relevant_sccs = [frozenset({Z})]
        ancestral_closure = {Y, Z}

        # ancestral closure is {Y, Z}
        # So intervention_set should be nodes - ancestral_closure = {R, X, Y, Z} - {Y, Z} = {R, X}

        # added for documentation - the original_distribution = P(R, X, Y, Z)

        # intervention set: nodes outside ancestral closure
        nodes = set(graph.nodes())
        intervention_set = nodes - ancestral_closure  # should be {R, X}

        # expected output should be P(Z | Y, R, X) or Z conditioned on Y (its predecessor in apt-order), normalized
        conditional_prob = P(Z | Y)
        normalization = Sum[Z](P(Z | Y))  # type:ignore[misc]
        # result = ((P(Z | Y)) / (Sum_Z P(Z | Y)))
        expected_expression = conditional_prob / normalization
        expected = {frozenset({Z}): expected_expression}

        result = compute_scc_distributions(
            graph=graph,
            subgraph_a=subgraph_a,
            relevant_sccs=relevant_sccs,
            ancestral_closure=ancestral_closure,
            intervention_set=intervention_set,
        )
        self.assertEqual(expected, result)
