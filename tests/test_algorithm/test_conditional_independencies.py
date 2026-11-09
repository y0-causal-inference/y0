"""Test getting conditional independencies (and related)."""

from __future__ import annotations

import typing
import unittest
from collections.abc import Callable, Iterable
from functools import partial
from itertools import combinations, groupby, pairwise

from tests import requires_pgmpy
from y0.algorithm.conditional_independencies import (
    are_d_separated,
    get_conditional_independencies,
)
from y0.dsl import AA, B, C, D, E, F, G, Variable, X, Y
from y0.examples import (
    Example,
    d_separation_example,
    examples,
    frontdoor_backdoor_example,
    frontdoor_example,
)
from y0.graph import NxMixedGraph, iter_moral_links
from y0.struct import CITestTuple, DSeparationJudgement
from y0.util.combinatorics import powerset


def _legacy_judgement_grouper(judgement: DSeparationJudgement) -> tuple[Variable, Variable]:
    """Group legacy judgements by queried pair."""
    return judgement.left, judgement.right


def _legacy_topological_policy(
    judgement: DSeparationJudgement, order: list[Variable]
) -> tuple[int, int]:
    """Apply the legacy topological tie-breaker."""
    return (
        len(judgement.conditions),
        sum(order.index(v) for v in judgement.conditions),
    )


def _legacy_get_topological_policy(
    graph: NxMixedGraph,
) -> Callable[[DSeparationJudgement], tuple[int, int]]:
    """Build the legacy topological policy for the given graph."""
    order = list(graph.topological_sort())
    return partial(_legacy_topological_policy, order=order)


def _legacy_minimal(
    judgements: Iterable[DSeparationJudgement],
    graph: NxMixedGraph,
) -> set[DSeparationJudgement]:
    """Reproduce the legacy minimal-judgement reduction."""
    policy = _legacy_get_topological_policy(graph)
    judgements = sorted(judgements, key=_legacy_judgement_grouper)
    return {min(vs, key=policy) for _, vs in groupby(judgements, _legacy_judgement_grouper)}


def _legacy_d_separations(
    graph: NxMixedGraph,
    *,
    max_conditions: int | None = None,
) -> Iterable[DSeparationJudgement]:
    """Reproduce the legacy exhaustive separator search."""
    vertices = set(graph.nodes())
    for a, b in combinations(vertices, 2):
        for conditions in powerset(vertices - {a, b}, stop=max_conditions):
            judgement = are_d_separated(graph, a, b, conditions=conditions)
            if judgement.separated:
                yield judgement
                break


def _legacy_get_conditional_independencies(
    graph: NxMixedGraph,
    *,
    max_conditions: int | None = None,
) -> set[DSeparationJudgement]:
    """Reproduce legacy conditional independencies for regression comparison."""
    return _legacy_minimal(
        _legacy_d_separations(graph, max_conditions=max_conditions),
        graph,
    )


def _build_separator_search_graph(width: int = 4, depth: int = 3) -> NxMixedGraph:
    """Build a small dense layered graph that stresses separator search."""
    directed = []
    layers = []
    for layer_index in range(depth):
        layer = [f"L{layer_index}_{node_index}" for node_index in range(width)]
        layers.append(layer)
    for source_layer, target_layer in pairwise(layers):
        for source in source_layer:
            for target in target_layer:
                directed.append((source, target))
    return NxMixedGraph.from_str_edges(directed=directed)


def _build_ci_throughput_graph(components: int = 4, chain_length: int = 4) -> NxMixedGraph:
    """Build disconnected chains that yield many implied independencies."""
    directed = []
    for component_index in range(components):
        prefix = f"C{component_index}"
        for node_index in range(chain_length - 1):
            directed.append((f"{prefix}_{node_index}", f"{prefix}_{node_index + 1}"))
    return NxMixedGraph.from_str_edges(directed=directed)


class TestDSeparation(unittest.TestCase):
    """Test the d-separation utility."""

    def test_mit_example(self) -> None:
        """Test checking D-separation on the MIT example."""
        graph = d_separation_example.graph

        self.assertFalse(are_d_separated(graph, AA, B, conditions=[D, F]))
        self.assertTrue(are_d_separated(graph, AA, B))
        self.assertTrue(are_d_separated(graph, D, E, conditions=[C]))
        self.assertFalse(are_d_separated(graph, AA, B, conditions=[C]))
        self.assertFalse(are_d_separated(graph, D, E))
        self.assertFalse(are_d_separated(graph, D, E, conditions=[AA, B]))
        self.assertFalse(are_d_separated(graph, G, G, conditions=[C]))

    def test_examples(self) -> None:
        """Check that example conditional independencies are d-separations and conditions (if present) are required.

        This test is using convenient examples to ensure that the d-separation algorithm
        isn't just always returning true or false.
        """
        testable = (
            example for example in examples if example.conditional_independencies is not None
        )

        for example in testable:
            with self.subTest(name=example.name):
                for ci in example.conditional_independencies:
                    self.assertIn(ci.left, example.graph)
                    self.assertIn(ci.right, example.graph)
                    judgement = are_d_separated(
                        example.graph, ci.left, ci.right, conditions=ci.conditions
                    )
                    self.assertTrue(
                        judgement,
                        msg=f"Expected d-separation not found in {example.name}",
                    )
                    if ci.conditions:
                        self.assertFalse(
                            are_d_separated(example.graph, ci.left, ci.right),
                            msg="Unexpected d-separation",
                        )

    def test_moral_links(self) -> None:
        """Test adding 'moral links' (part of the d-separation algorithm).

        This test covers several cases around moral links to ensure that they are added
        when needed.
        """
        graph = NxMixedGraph.from_str_edges(
            nodes=("a", "b", "c"),
            directed=[("a", "b"), ("b", "c")],
        )
        links: set[frozenset[Variable]] = {frozenset(e) for e in iter_moral_links(graph)}
        self.assertEqual(set(), links, msg="Unexpected moral links added.")

        graph = NxMixedGraph.from_str_edges(
            nodes=("a", "b", "c"),
            directed=[("a", "c"), ("b", "c")],
        )
        links = {frozenset(e) for e in iter_moral_links(graph)}
        self.assertEqual(
            {frozenset([Variable("a"), Variable("b")])},
            links,
            msg="Moral links not as expected in single-link case.",
        )

        graph = NxMixedGraph.from_str_edges(
            nodes=("a", "b", "aa", "bb", "c"),
            directed=[("a", "c"), ("b", "c"), ("aa", "c"), ("bb", "c")],
        )
        links = {frozenset(e) for e in iter_moral_links(graph)}
        self.assertEqual(
            {
                frozenset(e)
                for e in [
                    (Variable("a"), Variable("b")),
                    (Variable("a"), Variable("aa")),
                    (Variable("a"), Variable("bb")),
                    (Variable("aa"), Variable("b")),
                    (Variable("aa"), Variable("bb")),
                    (Variable("b"), Variable("bb")),
                ]
            },
            links,
            msg="Moral links not as expected in multi-link case.",
        )

        graph = NxMixedGraph.from_str_edges(
            nodes=("a", "b", "c", "d", "e"),
            directed=[("a", "c"), ("b", "c"), ("c", "e"), ("d", "e")],
        )
        links = {frozenset(e) for e in iter_moral_links(graph)}
        self.assertEqual(
            {
                frozenset(e)
                for e in [(Variable("a"), Variable("b")), (Variable("c"), Variable("d"))]
            },
            links,
            msg="Moral links not as expected in multi-site case.",
        )


class TestGetConditionalIndependencies(unittest.TestCase):
    """Test getting conditional independencies."""

    def assert_example_has_judgements(self, example: Example) -> None:
        """Assert that the example is consistent w.r.t. D-separations."""
        if example.conditional_independencies is None:
            self.fail(msg="no conditional independencies were found")
        self.assert_has_judgements(
            graph=example.graph,
            judgements=example.conditional_independencies,
        )

    def assert_judgement_types(self, judgements: Iterable[DSeparationJudgement]) -> None:
        """Assert all judgmenets have the right types."""
        self.assertTrue(
            all(
                (
                    isinstance(judgement.left, Variable)
                    and isinstance(judgement.right, Variable)
                    and all(isinstance(c, Variable) for c in judgement.conditions)
                )
                for judgement in judgements
            )
        )

    def assert_has_judgements(
        self, graph: NxMixedGraph, judgements: Iterable[DSeparationJudgement]
    ) -> None:
        """Assert that the graph has the correct conditional independencies.

        :param graph: the graph to test
        :param judgements: the set of expected conditional independencies
        """
        self.assertTrue(all(isinstance(node, Variable) for node in graph.nodes()))
        self.assert_judgement_types(judgements)

        asserted_judgements = set(judgements)
        self.assertIsNotNone(asserted_judgements, "Expected independencies is empty.")

        observed_judgements = get_conditional_independencies(graph)
        self.assertIsNotNone(observed_judgements, "Observed independencies is empty.")
        self.assert_judgement_types(asserted_judgements)

        self.assertTrue(
            all(judgement.is_canonical for judgement in observed_judgements),
            msg="one or more of the returned DSeparationJudgement instances are not canonical",
        )

        self.assert_valid_judgements(graph, asserted_judgements)
        self.assert_valid_judgements(graph, observed_judgements)

        expected_pairs = {(judgement.left, judgement.right) for judgement in asserted_judgements}
        observed_pairs = {(judgement.left, judgement.right) for judgement in observed_judgements}
        self.assertEqual(
            expected_pairs,
            observed_pairs,
            "Judgements do not find same separable pairs",
        )

        def _get_match(
            ref: DSeparationJudgement, options: set[DSeparationJudgement]
        ) -> DSeparationJudgement | None:
            """Find a judgement that has the same left/right pair as the reference judgement."""
            for alt in options:
                if ref.left == alt.left and ref.right == alt.right:
                    return alt
            return None

        for judgement in asserted_judgements:
            with self.subTest(name=judgement):
                matching = _get_match(judgement, observed_judgements)
                if matching is None:
                    raise self.fail("No matching judgement found.")
                self.assertGreaterEqual(
                    len(judgement.conditions),
                    len(matching.conditions),
                    msg="Observed conditional independence more complicated than reference.",
                )

    def assert_valid_judgements(
        self, graph: NxMixedGraph, judgements: set[DSeparationJudgement]
    ) -> None:
        """Check that a set of judgments are valid with respect to a graph."""
        self.assertIsInstance(graph, NxMixedGraph)

        for judgement in judgements:
            self.assertTrue(
                all(isinstance(condition, Variable) for condition in judgement.conditions)
            )
            self.assertTrue(
                are_d_separated(
                    graph,
                    judgement.left,
                    judgement.right,
                    conditions=judgement.conditions,
                ),
                msg=f"Conditional independency is not a d-separation in {graph}",
            )

        pairs = [(judgement.left, judgement.right) for judgement in judgements]
        self.assertEqual(len(pairs), len(set(pairs)), "Duplicate left/right pair observed")

    def test_examples(self) -> None:
        """Test getting the conditional independencies from the example graphs."""
        testable = (
            example for example in examples if example.conditional_independencies is not None
        )

        for example in testable:
            with self.subTest(name=example.name):
                self.maxDiff = None
                self.assert_example_has_judgements(example)

    @requires_pgmpy
    def test_ci_test_continuous(self) -> None:
        """Test conditional independency test on continuous data."""
        if frontdoor_example.generate_data is None:
            raise self.fail()
        data = frontdoor_example.generate_data(500)  # continuous
        judgement = DSeparationJudgement(
            left=X,
            right=Y,
            separated=...,  # type:ignore
            conditions=(),
        )
        test_result_bool = judgement.test(data, method="pearson", boolean=True)
        self.assertIsInstance(test_result_bool, bool)

        test_result_tuple: CITestTuple = judgement.test(data, method="pearson", boolean=False)
        self.assertIsInstance(test_result_tuple, CITestTuple)
        self.assertIsNone(test_result_tuple.dof)

        # Test that an error is thrown if using a discrete test on continuous data
        with self.assertRaises(ValueError):
            judgement.test(data, method="chi-square", boolean=True)

    @requires_pgmpy
    def test_ci_test_discrete(self) -> None:
        """Test conditional independency test on discrete data."""
        from pgmpy.estimators import CITests

        if frontdoor_backdoor_example.generate_data is None:
            raise self.fail()
        data = frontdoor_backdoor_example.generate_data(500)  # discrete
        judgement = DSeparationJudgement(
            left=X,
            right=Y,
            separated=...,  # type:ignore
            conditions=(),
        )
        for method in typing.get_args(CITests):
            test_result_bool = judgement.test(data, method=method, boolean=True)
            self.assertIsInstance(test_result_bool, bool)

            test_result_tuple: CITestTuple = judgement.test(data, method=method, boolean=False)
            self.assertIsInstance(test_result_tuple, CITestTuple)
            self.assertIsNotNone(test_result_tuple.dof)

        # Test that an error is thrown if using a continous test on discrete data
        with self.assertRaises(ValueError):
            judgement.test(data, method="pearson", boolean=True)

    def test_optimized_separator_search_matches_legacy(self) -> None:
        """Test optimized separator search preserves legacy pair coverage and condition complexity."""
        for graph in [
            _build_separator_search_graph(),
            _build_ci_throughput_graph(),
        ]:
            with self.subTest(graph=graph):
                observed = get_conditional_independencies(graph, max_conditions=2)
                expected = _legacy_get_conditional_independencies(graph, max_conditions=2)

                observed_pairs = {
                    (judgement.left.name, judgement.right.name) for judgement in observed
                }
                expected_pairs = {
                    (judgement.left.name, judgement.right.name) for judgement in expected
                }
                self.assertEqual(expected_pairs, observed_pairs)

                observed_by_pair = {
                    (judgement.left.name, judgement.right.name): judgement for judgement in observed
                }
                for judgement in expected:
                    pair = (judgement.left.name, judgement.right.name)
                    self.assertLessEqual(
                        len(observed_by_pair[pair].conditions),
                        len(judgement.conditions),
                    )
