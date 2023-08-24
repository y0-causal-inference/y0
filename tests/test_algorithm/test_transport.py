"""Unit tests for transport."""

from tests.test_algorithm import cases
from y0.algorithm.transport import (
    TARGET_DOMAIN,
    TransportQuery,
    TRSOQuery,
    get_nodes_to_transport,
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
from y0.dsl import PP, Y1, Y2, Expression, Pi1, Pi2, Product, Sum, Variable, W, X, Y, Z
from y0.graph import NxMixedGraph
from y0.mutate import canonicalize

X1, X2 = Variable("X1"), Variable("X2")

# Figure 8 from https://arxiv.org/abs/1806.07172
tikka_trso_figure_8 = NxMixedGraph.from_edges(
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
    ],
)

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


class TestTransport(cases.GraphTestCase):
    """Test surrogate outcomes and transportability."""

    # def test_transport_figure_8(self):
    #     """Test transportability from figure 8."""
    #     actual = transport(
    #         tikka_trso_figure_8,
    #         tikka_trso_figure_8_transport,
    #         treatments=[X1, X2],
    #         outcomes=[Y1, Y2],
    #     )
    #     # Query is also P[X1, X2](Y1, Y2)
    #     expected = ...

    #     # TODO probably need to canonicalize both of these
    #     self.assertEqual(expected, actual)

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

    def test_get_nodes_to_transport(self):
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

    def test_trso_line1(self):
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

    # def test_trso_line2(self):
    #     query_part1 = TRSOQuery(
    #         target_interventions={X1, X2, Y1, Y2},
    #         target_outcomes={Z, W},
    #         expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
    #         active_interventions=set(),
    #         domain=TARGET_DOMAIN,
    #         domains={Pi1, Pi2},
    #         graphs={
    #             TARGET_DOMAIN: tikka_trso_figure_8,
    #             Pi1: graph_1,
    #             Pi2: graph_2,
    #         },
    #         surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
    #     )

    #     # actual_part1 = trso(query_part1)
    #     # triggers line 2
    #     graph = query_part1.graphs[query_part1.domain]
    #     outcome_ancestors = graph.ancestors_inclusive(query_part1.target_outcomes)
    #     actual_part1 = trso_line2(query_part1, outcome_ancestors)

    #     outcomes_anc = {W, Z}
    #     new_query_part1 = TRSOQuery(
    #         target_interventions=set(),
    #         target_outcomes={W, Z},
    #         expression=Sum.safe(PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()), {X1, Y1, Y2, X2}),
    #         active_interventions=set(),
    #         domain=TARGET_DOMAIN,
    #         domains={Pi1, Pi2},
    #         graphs={
    #             TARGET_DOMAIN: tikka_trso_figure_8.subgraph(outcomes_anc),
    #             Pi1: graph_1,
    #             Pi2: graph_2,
    #         },
    #         surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
    #     )

    #     expected_part1 = new_query_part1
    #     self.assertEqual(actual_part1.expression, expected_part1.expression)
    #     self.assertEqual(actual_part1, expected_part1)

    #     # this is the simplified form of the expression
    #     # Maybe to be done in some future implementation
    #     # expected3 = PP[TARGET_DOMAIN]({W,Z})
    #     # actual_part1_simplified = actual_part1.simplified
    #     # self.assertEqual(expected3, actual_part1_simplified)

    #     query_part2 = TRSOQuery(
    #         target_interventions={X1, X2, Z, W, Y2},
    #         target_outcomes={Y1},
    #         expression=PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()),
    #         active_interventions=set(),
    #         domain=TARGET_DOMAIN,
    #         domains={Pi1, Pi2},
    #         graphs={
    #             TARGET_DOMAIN: tikka_trso_figure_8,
    #             Pi1: graph_1,
    #             Pi2: graph_2,
    #         },
    #         surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
    #     )

    #     # actual_part2 = trso(query_part2)
    #     # should trigger line 2
    #     graph = query_part2.graphs[query_part2.domain]
    #     outcome_ancestors = graph.ancestors_inclusive(query_part2.target_outcomes)
    #     actual_part2 = trso_line2(query_part2, outcome_ancestors)
    #     # triggers line 2

    #     outcomes_anc = {X1, W, Z, Y1}
    #     new_query_part2 = TRSOQuery(
    #         target_interventions={X1, W, Z},
    #         target_outcomes={Y1},
    #         expression=Sum.safe(PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes()), {X2, Y2}),
    #         active_interventions=set(),
    #         domain=TARGET_DOMAIN,
    #         domains={Pi1, Pi2},
    #         graphs={
    #             TARGET_DOMAIN: tikka_trso_figure_8.subgraph(outcomes_anc),
    #             Pi1: graph_1,
    #             Pi2: graph_2,
    #         },
    #         surrogate_interventions={Pi1: {X1}, Pi2: {X2}},
    #     )

    #     expected_part2 = new_query_part2
    #     self.assertEqual(actual_part2.expression, expected_part2.expression)
    #     self.assertEqual(actual_part2, expected_part2)

    def test_trso_line3(self):
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
        pass

    def test_trso_line10(self):
        pass

    def test_trso(self):
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
        expected_part1 = trso_line1(
            new_query_part1.target_outcomes, new_query_part1.expression, new_graph
        )

        self.assertEqual(expected_part1, actual_part1)
        expected2 = PP[TARGET_DOMAIN](tikka_trso_figure_8.nodes() - {X1, X2, Y1, Y2})
        self.assertEqual(expected2, actual_part1)

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
        actual_part2 = trso(query_part2)
        expected_part2 = PP[TARGET_DOMAIN](Y1, Z, W).conditional((Z, W))
        # TODO there is still a missing piece here (do[x1])
        self.assert_expr_equal(expected_part2, actual_part2)

        # The path here should be
        # line2, line6, line10, return from line7
        # The actual path is
        # line2, line6, line9, return from line7
        # X1 appears to be getting dropped erroneuosly

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
        expected_part3 = PP[TARGET_DOMAIN](Y2, X1, Z, W).conditional((X1, Z, W))
        # TODO there is still a missing piece here (do[x2])
        self.assert_expr_equal(expected_part3, actual_part3)

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
        expected_part2 = PP[TARGET_DOMAIN](Y1, Z, W).conditional((Z, W))
        expected_part3 = PP[TARGET_DOMAIN](Y2, X1, Z, W).conditional((X1, Z, W))

        expected = Sum.safe(
            Product.safe([expected_part1, expected_part2, expected_part3]),
            (W, Z),
        )
        self.assert_expr_equal(expected, actual)
