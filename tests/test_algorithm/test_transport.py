"""Unit tests for transport."""

import unittest

from y0.algorithm.transport import (
    TransportQuery,
    TRSOQuery,
    activate_domain_and_interventions,
    create_transport_diagram,
    get_nodes_to_transport,
    identify_target_outcomes,
    surrogate_to_transport,
    transport_variable,
    trso,
    trso_line1,
    trso_line2,
    trso_line3,
    trso_line4,
    trso_line6,
    trso_line9,
    trso_line10,
)
from y0.dsl import (
    PP,
    TARGET_DOMAIN,
    X1,
    X2,
    Y1,
    Y2,
    Expression,
    Fraction,
    Pi1,
    Pi2,
    Pi3,
    PopulationProbability,
    Probability,
    Product,
    Sum,
    Variable,
    W,
    X,
    Y,
    Z,
    Zero,
)
from y0.examples import tikka_trso_figure_8_graph as tikka_trso_figure_8
from y0.graph import NxMixedGraph
from y0.mutate import canonicalize, fraction_expand

tikka_trso_figure_8_transport = {
    X1: [Pi1],
    Y2: [Pi1],
    X2: [Pi2],
}

graph_1 = NxMixedGraph.from_edges(
    undirected=[(X1, Y1), (Z, W), (Z, X2)],
    directed=[
        (X1, Y1),
        (X1, Y2),
        (W, Y1),
        (W, Y2),
        (Z, Y1),
        (Z, X2),
        (X2, Y2),
        (Z, Y2),
        (transport_variable(X1), X1),
        (transport_variable(Y2), Y2),
    ],
)
graph_2 = NxMixedGraph.from_edges(
    undirected=[(X1, Y1), (Z, W), (Z, X2)],
    directed=[
        (X1, Y1),
        (X1, Y2),
        (W, Y1),
        (W, Y2),
        (Z, Y1),
        (Z, X2),
        (X2, Y2),
        (Z, Y2),
        (transport_variable(X2), X2),
    ],
)


class _TestCase(unittest.TestCase):
    def assert_expr_equal(self, expected: Expression, actual: Expression) -> None:
        """Assert that two expressions are the same."""
        ordering = tuple(expected.get_variables())
        expected_canonical = canonicalize(expected, ordering)
        actual_canonical = canonicalize(actual, ordering)
        self.assertEqual(
            expected_canonical,
            actual_canonical,
            msg=f"\nExpected: {str(expected_canonical)}\nActual:   {str(actual_canonical)}",
        )


class TestTransport(_TestCase):
    """Test surrogate outcomes and transportability."""

    def test_create_transport_diagram(self):
        """Test that we can create the transport diagram correctly."""
        graph_pi1 = create_transport_diagram(graph=tikka_trso_figure_8, nodes_to_transport={X1, Y2})
        graph_pi2 = create_transport_diagram(graph=tikka_trso_figure_8, nodes_to_transport={X2})

        self.assertEqual(graph_1, graph_pi1)
        self.assertEqual(graph_2, graph_pi2)

    def test_get_nodes_to_transport(self):
        """Test that we can correctly find the nodes which should have transport nodes."""
        expected = {X1, Y2}
        actual = get_nodes_to_transport(
            surrogate_interventions=X1, surrogate_outcomes=Y1, graph=tikka_trso_figure_8
        )
        self.assertEqual(actual, expected)
        expected = {X2}
        actual = get_nodes_to_transport(
            surrogate_interventions={X2}, surrogate_outcomes={Y2}, graph=tikka_trso_figure_8
        )
        self.assertEqual(actual, expected)

        # Test for multiple vertices in interventions and surrogate_outcomes
        expected = {X1, X2, Y1}
        actual = get_nodes_to_transport(
            surrogate_interventions={X2, X1}, surrogate_outcomes={Y2, W}, graph=tikka_trso_figure_8
        )
        self.assertEqual(actual, expected)

    def test_surrogate_to_transport(self):
        """Test that surrogate_to_transport correctly converts to a transport query."""
        target_outcomes = {Y1, Y2}
        target_interventions = {X1, X2}
        surrogate_outcomes = {Pi1: {Y1}, Pi2: {Y2}}
        surrogate_interventions = {Pi1: {X1}, Pi2: {X2}}

        actual = surrogate_to_transport(
            target_outcomes=target_outcomes,
            target_interventions=target_interventions,
            graph=tikka_trso_figure_8,
            surrogate_outcomes=surrogate_outcomes,
            surrogate_interventions=surrogate_interventions,
        )

        expected = TransportQuery(
            target_interventions=target_interventions,
            target_outcomes=target_outcomes,
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8,
                Pi1: graph_1,
                Pi2: graph_2,
            },
            domains={Pi1, Pi2},
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
            target_experiments=set(),
        )
        self.assertEqual(actual, expected)
        extra_surrogate_outcomes = {Pi1: {Y1}, Pi2: {Y2}, Pi3: {Y2}}
        missing_surrogate_interventions = {Pi1: {Y2}}
        with self.assertRaises(ValueError):
            surrogate_to_transport(
                target_outcomes=target_outcomes,
                target_interventions=target_interventions,
                graph=tikka_trso_figure_8,
                surrogate_outcomes=extra_surrogate_outcomes,
                surrogate_interventions=surrogate_interventions,
            )
        with self.assertRaises(ValueError):
            surrogate_to_transport(
                target_outcomes=target_outcomes,
                target_interventions=target_interventions,
                graph=tikka_trso_figure_8,
                surrogate_outcomes=surrogate_outcomes,
                surrogate_interventions=missing_surrogate_interventions,
            )

    def test_add_active_interventions(self):
        """Test that interventions are added correctly."""
        expected_1 = fraction_expand(PP[Pi1][X1](Y1 | W, Z))
        expected_2 = fraction_expand(PP[Pi2][X2](Y2 | W, Z, X1))
        test_1 = fraction_expand(PP[Pi1](Y1 | W, Z))
        test_2 = fraction_expand(PP[Pi2](Y2 | W, Z, X1))
        # expected = canonicalize(
        #     Sum.safe(
        #         Product.safe([expected_part1, expected_part2, expected_part3]),
        #         (W, Z),
        #     )
        # )
        # expected = Sum[Y1](PP[Pi1][X1](W, Y1, Z)
        actual_1 = activate_domain_and_interventions(test_1, {X1}, Pi1)
        self.assert_expr_equal(expected_1, actual_1)
        actual_2 = activate_domain_and_interventions(test_2, {X2}, Pi2)
        self.assert_expr_equal(expected_2, actual_2)

        expected_3 = Sum.safe(expected_1, (W, Z))
        test_3 = Sum.safe(test_1, (W, Z))
        actual_3 = activate_domain_and_interventions(test_3, {X1}, Pi1)
        self.assert_expr_equal(expected_3, actual_3)

        # TODO fix this test
        # expected_1b = fraction_expand(PP[Pi1][X2](Y1 | W, Z))
        # expected_4 = Product.safe((expected_1b, expected_2))
        # test_4 = Product.safe((test_1, test_2))
        # actual_4 = activate_domain_and_interventions(test_4, X2)
        # self.assert_expr_equal(expected_4, actual_4)

    def test_trso_line1(self):
        """Test that trso_line 1 returns the correct expression."""
        # triggers line 1
        outcomes = {Y1, Y2}
        domain_graph = tikka_trso_figure_8
        expression = PP[TARGET_DOMAIN](domain_graph.nodes())

        expected = Sum.safe(expression, {W, X1, X2, Z})
        actual = trso_line1(
            outcomes,
            expression,
            domain_graph,
        )
        self.assert_expr_equal(expected, actual)

    def test_trso_line2(self):
        """Test that trso _line2 correctly modifies the query."""
        query_part1 = TRSOQuery(
            target_interventions={X1, X2, Y1, Y2},
            target_outcomes={Z, W},
            expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8,
                Pi1: graph_1,
                Pi2: graph_2,
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

        # actual_part1 = trso(query_part1)
        # triggers line 2
        graph = query_part1.graphs[query_part1.domain]
        outcome_ancestors = graph.ancestors_inclusive(query_part1.target_outcomes)
        actual_part1 = trso_line2(query_part1, outcome_ancestors)

        outcomes_anc = {W, Z}
        outcomes_anc_pi1 = graph_1.ancestors_inclusive(query_part1.target_outcomes)
        outcomes_anc_pi2 = graph_2.ancestors_inclusive(query_part1.target_outcomes)
        new_query_part1 = TRSOQuery(
            target_interventions=set(),
            target_outcomes={W, Z},
            expression=canonicalize(PP[TARGET_DOMAIN]((W, Z))),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8.subgraph(outcomes_anc),
                Pi1: graph_1.subgraph(outcomes_anc_pi1),
                Pi2: graph_2.subgraph(outcomes_anc_pi2),
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

        expected_part1 = new_query_part1
        self.assertEqual(actual_part1, expected_part1)

        # this is the simplified form of the expression
        # Maybe to be done in some future implementation
        # expected3 = PP[TARGET_DOMAIN]({W,Z})
        # actual_part1_simplified = actual_part1.simplified
        # self.assertEqual(expected3, actual_part1_simplified)

        query_part2 = TRSOQuery(
            target_interventions={X1, X2, Z, W, Y2},
            target_outcomes={Y1},
            expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8,
                Pi1: graph_1,
                Pi2: graph_2,
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

        # actual_part2 = trso(query_part2)
        # should trigger line 2
        graph = query_part2.graphs[query_part2.domain]
        outcome_ancestors = graph.ancestors_inclusive(query_part2.target_outcomes)
        actual_part2 = trso_line2(query_part2, outcome_ancestors)
        # triggers line 2

        outcomes_anc = {X1, W, Z, Y1}
        outcomes_anc_pi1 = graph_1.ancestors_inclusive(query_part2.target_outcomes)
        outcomes_anc_pi2 = graph_2.ancestors_inclusive(query_part2.target_outcomes)

        new_query_part2 = TRSOQuery(
            target_interventions={X1, W, Z},
            target_outcomes={Y1},
            expression=canonicalize(PP[TARGET_DOMAIN](X1, W, Z, Y1)),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8.subgraph(outcomes_anc),
                Pi1: graph_1.subgraph(outcomes_anc_pi1),
                Pi2: graph_2.subgraph(outcomes_anc_pi2),
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

        expected_part2 = new_query_part2
        self.assertEqual(actual_part2, expected_part2)

        query_3 = TRSOQuery(
            target_interventions={X1, X2, Z, W, Y2},
            target_outcomes={Y1},
            expression=Sum.safe(PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()), (W, Z)),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8,
                Pi1: graph_1,
                Pi2: graph_2,
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

        graph = query_3.graphs[query_3.domain]
        outcome_ancestors = graph.ancestors_inclusive(query_3.target_outcomes)
        actual_3 = trso_line2(query_3, outcome_ancestors)

        outcomes_anc = {X1, W, Z, Y1}
        outcomes_anc_pi1 = graph_1.ancestors_inclusive(query_3.target_outcomes)
        outcomes_anc_pi2 = graph_2.ancestors_inclusive(query_3.target_outcomes)
        expected_3 = TRSOQuery(
            target_interventions={X1, W, Z},
            target_outcomes={Y1},
            expression=Sum.safe(
                Sum.safe(PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()), (W, Z)), (X2, Y2)
            ),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8.subgraph(outcomes_anc),
                Pi1: graph_1.subgraph(outcomes_anc_pi1),
                Pi2: graph_2.subgraph(outcomes_anc_pi2),
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

        self.assertEqual(actual_3, expected_3)

    def test_trso_line3(self):
        """Test that trso_line3 correctly modifies the query."""
        transportability_diagram_line3 = NxMixedGraph.from_edges(
            directed=[
                (W, X),
                (X, Y),
                (X, Z),
                (Z, Y),
            ],
        )

        query = TRSOQuery(
            target_interventions={X},
            target_outcomes={Y},
            expression=PP[TARGET_DOMAIN](transportability_diagram_line3.nodes()),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: transportability_diagram_line3,
            },
            surrogate_interventions={},
        )

        # target_interventions_overbar = transportability_diagram_line3.remove_in_edges(
        #     query.target_interventions
        # )
        # additional_interventions = (
        #     transportability_diagram_line3.nodes()
        #     - query.target_interventions
        #     - target_interventions_overbar.ancestors_inclusive(query.target_outcomes)
        # )
        additional_interventions = {W}

        new_query = TRSOQuery(
            target_interventions={X, W},
            target_outcomes={Y},
            expression=PP[TARGET_DOMAIN](transportability_diagram_line3.nodes()),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: transportability_diagram_line3,
            },
            surrogate_interventions={},
        )
        expected = new_query

        actual = trso_line3(query=query, additional_interventions=additional_interventions)
        self.assertEqual(actual, expected)

        self.assertIsNotNone(trso(query))
        self.assertIsNotNone(trso(actual))

    def test_trso_line4(self):
        """Test that trso_line4 builds a dictionary of components and modified queries."""
        query = TRSOQuery(
            target_interventions={X1, X2},
            target_outcomes={Y1, Y2},
            expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8,
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

        districts_without_interventions = tikka_trso_figure_8.remove_nodes_from(
            query.target_interventions
        ).districts()

        expected = {
            frozenset([Y2]): TRSOQuery(
                target_interventions={X1, X2, Z, W, Y1},
                target_outcomes={Y2},
                expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
                active_interventions=set(),
                domain=TARGET_DOMAIN,
                domains={Pi1, Pi2},
                graphs={
                    TARGET_DOMAIN: tikka_trso_figure_8,
                },
                surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
            ),
            frozenset([Y1]): TRSOQuery(
                target_interventions={X1, X2, Z, W, Y2},
                target_outcomes={Y1},
                expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
                active_interventions=set(),
                domain=TARGET_DOMAIN,
                domains={Pi1, Pi2},
                graphs={
                    TARGET_DOMAIN: tikka_trso_figure_8,
                },
                surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
            ),
            frozenset([W, Z]): TRSOQuery(
                target_interventions={X1, X2, Y1, Y2},
                target_outcomes={W, Z},
                expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
                active_interventions=set(),
                domain=TARGET_DOMAIN,
                domains={Pi1, Pi2},
                graphs={
                    TARGET_DOMAIN: tikka_trso_figure_8,
                },
                surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
            ),
        }

        actual = trso_line4(
            query,
            districts_without_interventions,
        )

        self.assertEqual(expected, actual)

    def test_trso_line6(self):
        """Test that trso_line6 builds a dictionary of domains and modified queries."""
        query = TRSOQuery(
            target_interventions={X1, Z, W},
            target_outcomes={Y1},
            expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8,
                Pi1: graph_1,
                Pi2: graph_2,
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

        actual = trso_line6(
            query,
        )
        new_transportability_diagram = query.graphs[Pi1].subgraph(
            query.graphs[Pi1].nodes()
            - query.surrogate_interventions[Pi1].intersection(query.target_interventions)
        )
        expected = {
            Pi1: TRSOQuery(
                target_interventions={Z, W},
                target_outcomes={Y1},
                expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
                active_interventions={X1},
                domain=Pi1,
                domains={Pi1, Pi2},
                graphs={
                    TARGET_DOMAIN: tikka_trso_figure_8,
                    Pi1: new_transportability_diagram,
                    Pi2: graph_2,
                },
                surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
            )
        }
        self.assertEqual(expected, actual)

        outcomes_anc = {X1, W, Z, Y1}
        query_part2 = TRSOQuery(
            target_interventions={X1, W, Z},
            target_outcomes={Y1},
            expression=Sum.safe(PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()), {X2, Y2}),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8.subgraph(outcomes_anc),
                Pi1: graph_1,
                Pi2: graph_2,
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

        actual_part2 = trso_line6(
            query_part2,
        )

        new_transportability_diagram = query_part2.graphs[Pi1].subgraph(
            query_part2.graphs[Pi1].nodes()
            - query_part2.surrogate_interventions[Pi1].intersection(
                query_part2.target_interventions
            )
        )
        expected_part2 = {
            Pi1: TRSOQuery(
                target_interventions={Z, W},
                target_outcomes={Y1},
                expression=Sum.safe(PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()), {X2, Y2}),
                active_interventions={X1},
                domain=Pi1,
                domains={Pi1, Pi2},
                graphs={
                    TARGET_DOMAIN: tikka_trso_figure_8.subgraph(outcomes_anc),
                    Pi1: new_transportability_diagram,
                    Pi2: graph_2,
                },
                surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
            )
        }
        self.assertEqual(expected_part2, actual_part2)

    def test_trso_line9(self):
        """Test that trso_line3 returns the correct expression."""
        line9_query1 = TRSOQuery(
            target_interventions={W, Z, X1},
            target_outcomes={Y2},
            expression=PP[Pi2](W, X1, Y2, Z),
            active_interventions={X2},
            domain=Pi2,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8.subgraph({W, X1, Y2, Z}),
                Pi1: graph_1.subgraph({W, X1, Y2, Z}),
                Pi2: graph_2.subgraph({W, X1, Y2, Z}),
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )
        district1 = {Y2}
        line9_actual1 = trso_line9(line9_query1, district1)
        line9_expected1 = PP[Pi2](W, X1, Y2, Z) / Sum.safe(PP[Pi2](W, X1, Y2, Z), (Y2,))
        self.assertEqual(line9_expected1, line9_actual1)

        line9_query2 = TRSOQuery(
            target_interventions={W, Z},
            target_outcomes={Y1},
            expression=PP[Pi2](W, Y1, Z),
            active_interventions={X1},
            domain=Pi1,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8.subgraph({W, Y1, Z}),
                Pi1: graph_1.subgraph({W, Y1, Z}),
                Pi2: graph_2.subgraph({W, Y1, Z}),
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )
        district2 = {Y1}
        line9_actual2 = trso_line9(line9_query2, district2)
        line9_expected2 = PP[Pi2](W, Y1, Z) / Sum.safe(PP[Pi2](W, Y1, Z), (Y1,))
        self.assertEqual(line9_expected2, line9_actual2)

        line9_query3 = TRSOQuery(
            target_interventions={W, Z},
            target_outcomes={Y1},
            expression=Zero(),
            active_interventions={X1},
            domain=Pi1,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8.subgraph({W, Y1, Z}),
                Pi1: graph_1.subgraph({W, Y1, Z}),
                Pi2: graph_2.subgraph({W, Y1, Z}),
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )
        district3 = {Y1}
        with self.assertRaises(RuntimeError):
            trso_line9(line9_query3, district3)

    def test_trso_line10(self):
        """Test that trso_line10 correctly modifies the query."""
        # pass
        line10_query1 = TRSOQuery(
            target_interventions={W, Z, X1},
            target_outcomes={Y2},
            expression=PP[Pi2](W, X1, Y2, Z),
            active_interventions={X2},
            domain=Pi2,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8.subgraph({W, X1, Y2, Z}),
                Pi1: graph_1.subgraph({W, X1, Y2, Z}),
                Pi2: graph_2.subgraph({W, X1, Y2, Z}),
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )
        district1 = {Y2}
        new_surrogate_interventions = dict()
        line10_actual1 = trso_line10(line10_query1, district1, new_surrogate_interventions)

        ordering = list(line10_query1.graphs[Pi2].topological_sort())
        i = ordering.index(Y2)
        pre_node = set(ordering[:i])
        prob = Probability.safe(Y2 | pre_node)
        new_expression = PopulationProbability(population=Pi2, distribution=prob.distribution)

        line10_expected1 = TRSOQuery(
            target_interventions=set(),
            target_outcomes={Y2},
            expression=new_expression,
            active_interventions={X2},
            domain=Pi2,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8.subgraph({W, X1, Y2, Z}),
                Pi1: graph_1.subgraph({W, X1, Y2, Z}),
                Pi2: graph_2.subgraph({Y2}),
            },
            surrogate_interventions=dict(),
        )

        self.assertEqual(line10_expected1, line10_actual1)


class TestIntegration(_TestCase):
    """Test integration over the whole workflow."""

    def test_transport_variable(self):
        """Test that transport nodes will not take inappropriate inputs."""
        with self.assertRaises(TypeError):
            transport_variable(Y1 @ -X2)

    def test_trso_1(self):
        """Test that trso returns the correct expression."""
        query_part1 = TRSOQuery(
            target_interventions={X1, X2, Y1, Y2},
            target_outcomes={Z, W},
            expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8,
                Pi1: graph_1,
                Pi2: graph_2,
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

        actual_part1 = trso(query_part1)

        graph = query_part1.graphs[query_part1.domain]
        outcome_ancestors = graph.ancestors_inclusive(query_part1.target_outcomes)
        new_query_part1 = trso_line2(query_part1, outcome_ancestors)
        new_graph = new_query_part1.graphs[new_query_part1.domain]
        expected_part1 = canonicalize(
            trso_line1(new_query_part1.target_outcomes, new_query_part1.expression, new_graph)
        )
        self.assertEqual(expected_part1, actual_part1)
        expected2 = canonicalize(PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes() - {X1, X2, Y1, Y2}))
        self.assertEqual(expected2, actual_part1)

        # this is the simplified form of the expression
        # Maybe to be done in some future implementation
        # expected3 = PP[TARGET_DOMAIN]({W,Z})
        # actual_part1_simplified = actual_part1.simplified
        # self.assertEqual(expected3, actual_part1_simplified)

    def test_trso_2(self):
        """Test that trso returns the correct expression."""
        query_part2 = TRSOQuery(
            target_interventions={X1, X2, Z, W, Y2},
            target_outcomes={Y1},
            expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8,
                Pi1: graph_1,
                Pi2: graph_2,
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )
        actual_part2 = trso(query_part2)
        expected_part2_conditional = canonicalize(
            PP[Pi1](Y1 @ X1, Z @ X1, W @ X1).conditional((Z @ X1, W @ X1))
        )
        expected_part2_magic_p = PP[Pi1][X1](Y1 | W, Z)
        expected_part2_full = PP[Pi1][X1](W, Y1, Z) / Sum[Y1](PP[Pi1][X1](W, Y1, Z))

        self.assertIsInstance(actual_part2, Fraction)
        self.assert_expr_equal(expected_part2_full.numerator, actual_part2.numerator)
        self.assert_expr_equal(expected_part2_full, actual_part2)
        self.assert_expr_equal(expected_part2_conditional, actual_part2)
        self.assert_expr_equal(fraction_expand(expected_part2_magic_p), actual_part2)

    def test_trso_3(self):
        """Test that trso returns the correct expression."""
        query_part3 = TRSOQuery(
            target_interventions={X1, X2, Z, W, Y1},
            target_outcomes={Y2},
            expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8,
                Pi1: graph_1,
                Pi2: graph_2,
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )
        actual_part3 = trso(query_part3)
        expected_part3_conditional = canonicalize(PP[Pi2][X2](Y2, X1, Z, W).conditional((X1, Z, W)))
        expected_part3_magic_p = PP[Pi2][X2](Y2 | W, Z, X1)
        expected_part3_full = PP[Pi2](W @ -X2, X1 @ -X2, Y2 @ -X2, Z @ -X2) / Sum[Y2](
            PP[Pi2](W @ -X2, X1 @ -X2, Y2 @ -X2, Z @ -X2)
        )
        self.assert_expr_equal(expected_part3_full, actual_part3)
        self.assert_expr_equal(expected_part3_conditional, actual_part3)
        self.assert_expr_equal(fraction_expand(expected_part3_magic_p), actual_part3)

    def test_trso_4(self):
        """Test that trso returns the correct expression."""
        query = TRSOQuery(
            target_interventions={X1, X2},
            target_outcomes={Y1, Y2},
            expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
            active_interventions=set(),
            domain=TARGET_DOMAIN,
            domains={Pi1, Pi2},
            graphs={
                TARGET_DOMAIN: tikka_trso_figure_8,
                Pi1: graph_1,
                Pi2: graph_2,
            },
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )

        actual = trso(query)
        expected_part1 = PP[TARGET_DOMAIN](W, Z)
        expected_part2 = PP[Pi1][X1](W, Y1, Z) / Sum[Y1](PP[Pi1][X1](W, Y1, Z))
        expected_part3 = PP[Pi2](W @ -X2, X1 @ -X2, Y2 @ -X2, Z @ -X2) / Sum[Y2](
            PP[Pi2](W @ -X2, X1 @ -X2, Y2 @ -X2, Z @ -X2)
        )

        expected = canonicalize(
            Sum.safe(
                Product.safe([expected_part1, expected_part2, expected_part3]),
                (W, Z),
            )
        )
        self.assert_expr_equal(expected, actual)

    def test_trso_5(self):
        """Test that trso returns the correct expression."""
        # TODO we should find a way to accomplish this with a transport call
        # This test triggers pillow_has_transport on line 10 and returns None
        target_interventions = {W, Z}
        target_outcomes = {Y1}
        surrogate_interventions = {Pi1: {X1}, Pi2: {X2}}
        surrogate_outcomes = {Pi1: {Y1}, Pi2: {Y2}}
        initial_expression = PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes())
        transport_query = surrogate_to_transport(
            graph=tikka_trso_figure_8,
            target_outcomes=target_outcomes,
            target_interventions=target_interventions,
            surrogate_outcomes=surrogate_outcomes,
            surrogate_interventions=surrogate_interventions,
        )

        transport_z = transport_variable(Z)
        hacked_graph = NxMixedGraph.from_edges(
            undirected=[(X1, Y1), (Y1, W), (Z, X2), (Z, W)],
            directed=[
                (X1, Y1),
                (X1, Y2),
                (W, Y1),
                (W, Y2),
                (Z, Y1),
                (Z, W),
                (Z, X2),
                (X2, Y2),
                (Z, Y2),
                (transport_z, Z),
            ],
        )
        trso_query = TRSOQuery(
            target_interventions={Z, W},
            target_outcomes=transport_query.target_outcomes,
            expression=initial_expression,
            active_interventions={X1},
            domain=TARGET_DOMAIN,
            domains=transport_query.domains,
            graphs=transport_query.graphs,
            surrogate_interventions=transport_query.surrogate_interventions,
        )
        trso_query.graphs[trso_query.domain] = hacked_graph
        self.assertIsNone(trso(trso_query))

    def test_transport_1(self):
        """Test that transport returns the correct expression."""
        expected_part1 = PP[TARGET_DOMAIN](W, Z)
        self.assertIsInstance(expected_part1, PopulationProbability)
        expected_part2 = fraction_expand(PP[Pi1][X1](Y1 | W, Z))
        self.assertIsInstance(expected_part2, Fraction)
        self.assertIsInstance(expected_part2.numerator, PopulationProbability)
        self.assertIsInstance(expected_part2.denominator, PopulationProbability)
        expected_part3 = fraction_expand(PP[Pi2][X2](Y2 | W, Z, X1))
        expected_estimand = canonicalize(
            Sum.safe(
                Product.safe([expected_part1, expected_part2, expected_part3]),
                (W, Z),
            )
        )
        actual_estimand = identify_target_outcomes(
            graph=tikka_trso_figure_8,
            target_outcomes={Y1, Y2},
            target_interventions={X1, X2},
            surrogate_outcomes={Pi1: {Y1}, Pi2: {Y2}},
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )
        self.assert_expr_equal(expected_estimand, actual_estimand)

    def test_transport_2(self):
        """Test that transport returns the correct expression."""
        # This test triggers part of line 11 in trso (district length of 1)
        new_graph = tikka_trso_figure_8.subgraph(tikka_trso_figure_8.nodes() - {X1})
        new_graph.add_undirected_edge(W, Y1)
        actual_11 = identify_target_outcomes(
            graph=new_graph,
            target_outcomes={Y1},
            target_interventions={Z, W},
            surrogate_outcomes={Pi1: {Y1}, Pi2: {Y2}},
            surrogate_interventions={Pi1: {X2}, Pi2: {X2}},
        )
        self.assertIsNone(actual_11)

    def test_transport_3(self):
        """Test that transport returns the correct expression."""
        # This triggers triggers not implemented error on line 9
        # Now it triggers value error
        new_graph = tikka_trso_figure_8.subgraph(tikka_trso_figure_8.nodes() - {X1})
        with self.assertRaises(ValueError):
            identify_target_outcomes(
                graph=new_graph,
                target_outcomes={Z, Y1, W},
                target_interventions={Y1},
                surrogate_outcomes={Pi1: {Y1}, Pi2: {Y2}},
                surrogate_interventions={Pi1: {X2}, Pi2: {X2}},
            )

    def test_transport_4(self):
        """Test that transport returns the correct expression."""
        # This triggers line 10.
        # TODO it fails on the next recursive loop, would be better to find an example that doesn't fail.
        grpah = NxMixedGraph.from_edges(
            undirected=[(X1, Y1), (Y1, W), (Z, X2)],
            directed=[
                (X1, Y1),
                (X1, Y2),
                (W, Y1),
                (W, Y2),
                (Z, Y1),
                (Z, X2),
                (X2, Y2),
                (Z, Y2),
            ],
        )
        estimand = identify_target_outcomes(
            graph=grpah,
            target_outcomes={Y1},
            target_interventions={W, Z},
            surrogate_outcomes={Pi1: {Y1}, Pi2: {Y2}},
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )
        self.assertIsNone(estimand)

    def test_transport_5(self):
        """Test that transport returns the correct expression."""
        with self.assertRaises(ValueError):
            identify_target_outcomes(
                graph=tikka_trso_figure_8,
                target_outcomes={Y1},
                target_interventions={W, Z},
                surrogate_outcomes={Pi1: {Y1}, Pi2: {Y2}},
                surrogate_interventions={Pi1: {X1}, Pi2: {X2}, Pi3: {X}},
            )

    def test_transport_6(self):
        """Test that transport returns the correct expression."""
        # This test triggers if expression is None:  continue block after line 6
        graph = NxMixedGraph.from_edges(
            undirected=[(X1, Y1), (Z, W), (Z, X2), (Y2, X1)],
            directed=[
                (X1, Y1),
                (X1, Y2),
                (W, Y1),
                (W, Y2),
                (Z, Y1),
                (Z, X2),
                (X2, Y2),
                (Z, Y2),
            ],
        )
        estimand = identify_target_outcomes(
            graph,
            target_outcomes={Y1, Y2},
            target_interventions={X1, X2},
            surrogate_outcomes={Pi1: {Y1}, Pi2: {Y2}},
            surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
        )
        self.assertIsNone(estimand)


class TestHighLevel(unittest.TestCase):
    """Tests on more high-level applications."""

    def test_grades(self):
        """Test grades scenario."""
        c, s, g, j = (Variable("C"), Variable("S"), Variable("G"), Variable("J"))
        graph = NxMixedGraph.from_edges(
            directed=[(c, s), (s, g), (g, j)],
            undirected=[(s, c), (j, c)],
        )
        estimand = identify_target_outcomes(
            graph=graph,
            target_outcomes={j},
            target_interventions={c},
            surrogate_outcomes={Pi1: {s}, Pi2: {j}},
            surrogate_interventions={Pi1: {c}, Pi2: {g}},
        )
        self.assertIsNotNone(estimand)
