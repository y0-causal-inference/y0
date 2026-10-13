"""Test getting conditional independencies (and related)."""

from __future__ import annotations

import random
import typing
import unittest
from collections import defaultdict
from collections.abc import Iterable

import networkx as nx

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


def _m_separated_by_paths(
    graph: NxMixedGraph, left: Variable, right: Variable, conditions: set[Variable]
) -> bool:
    """Decide m-separation by enumerating every path between two nodes.

    This follows the definition of m-separation in Richardson, T. (2003). Markov
    Properties for Acyclic Directed Mixed Graphs. Scandinavian Journal of Statistics,
    30(1), 145-157 (https://doi.org/10.1111/1467-9469.00323). A path is m-connecting given
    the conditions if every non-collider on it is not in the conditions and every collider
    on it is in the conditions or is an ancestor of one of them. A node is a collider on a
    path when both edges next to it have an arrowhead at it: ``A -> B`` has an arrowhead
    at ``B`` and ``A <-> B`` has one at each end. Two nodes are m-separated when no path
    between them is m-connecting.
    """
    # each entry is (neighbor, arrowhead at this node, arrowhead at the neighbor)
    edges: defaultdict[Variable, list[tuple[Variable, bool, bool]]] = defaultdict(list)
    for u, v in graph.directed.edges():
        edges[u].append((v, False, True))
        edges[v].append((u, True, False))
    for u, v in graph.undirected.edges():
        edges[u].append((v, True, True))
        edges[v].append((u, True, True))
    ancestors = set(conditions).union(*(nx.ancestors(graph.directed, c) for c in conditions))

    def _blocked(node: Variable, collider: bool) -> bool:
        return node not in ancestors if collider else node in conditions

    def _connected(node: Variable, head_at_node: bool, visited: set[Variable]) -> bool:
        for neighbor, head_here, head_there in edges[node]:
            if neighbor in visited or (node != left and _blocked(node, head_at_node and head_here)):
                continue
            if neighbor == right or _connected(neighbor, head_there, visited | {neighbor}):
                return True
        return False

    return not _connected(left, False, {left})


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

    def test_bidirected_collider(self) -> None:
        """Test that conditioning on a collider entered by a bidirected edge opens the path.

        In ``V0 -> V1 <-> V2``, the node ``V1`` is a collider, so ``V0`` and ``V2`` are
        separated marginally and connected given ``V1`` (m-separation, as defined in
        Richardson (2003), https://doi.org/10.1111/1467-9469.00323). The same holds along
        a chain of bidirected edges, where linking only the two ends of one bidirected edge
        is not enough.
        """
        v0, v1, v2, v3, v4 = (Variable(f"V{i}") for i in range(5))
        graph = NxMixedGraph.from_edges(directed=[(v0, v1)], undirected=[(v1, v2)])
        self.assertTrue(are_d_separated(graph, v0, v2).separated)
        self.assertFalse(are_d_separated(graph, v0, v2, conditions=[v1]).separated)

        graph = NxMixedGraph.from_edges(
            directed=[(v0, v1), (v4, v3)], undirected=[(v1, v2), (v2, v3)]
        )
        self.assertFalse(are_d_separated(graph, v0, v4, conditions=[v1, v2, v3]).separated)
        # V2 is a collider that is neither conditioned on nor an ancestor of a condition
        self.assertTrue(are_d_separated(graph, v0, v4, conditions=[v1, v3]).separated)

    def test_random_against_paths(self) -> None:
        """Test d-separation against a brute-force path search on random ADMGs."""
        rng = random.Random(0)  # noqa:S311
        for _ in range(300):
            nodes = [Variable(f"V{i}") for i in range(rng.randint(3, 6))]
            directed = [
                (u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :] if rng.random() < 0.35
            ]
            undirected = [
                (u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :] if rng.random() < 0.25
            ]
            graph = NxMixedGraph.from_edges(nodes=nodes, directed=directed, undirected=undirected)
            left, right = rng.sample(nodes, 2)
            conditions = {n for n in nodes if n not in (left, right) and rng.random() < 0.4}
            with self.subTest(
                directed=directed,
                undirected=undirected,
                left=left,
                right=right,
                conditions=conditions,
            ):
                self.assertEqual(
                    _m_separated_by_paths(graph, left, right, conditions),
                    are_d_separated(graph, left, right, conditions=conditions).separated,
                )

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
