# -*- coding: utf-8 -*-

"""Tests for the ``IDC*`` algorithm."""

from tests.test_algorithm import cases
from y0.algorithm.identify.id_c_star import idc_star_line_2
from y0.dsl import D, P, Variable, W, X, Y, Z
from y0.examples import figure_9a, figure_9c

d, w, x, y, z = -D, -W, -X, -Y, -Z


class TestIDCStar(cases.GraphTestCase):
    """Tests for the ``IDC*`` algorithm."""

    def test_idc_star_line_2(self):
        r"""Test line 2 of the IDC* algorithm.

        Construct the counterfactual graph Figure 9(c) where the corresponding modified query
        is :math:`P(Y_{x} = y|X= x',Z=z,D=d)`
        """
        input_event = {Y @ -x: x}
        input_conditional = {X: +x, Z: -z, D: -d}
        input_graph = figure_9a.graph
        actual_graph, actual_query = idc_star_line_2(input_graph, input_event, input_conditional)
        expected_graph = figure_9c.graph
        expected_query = P(D, X, Y @ ~X, Z)
        self.assert_expr_equal(expected=expected_query, actual=actual_query)
        self.assert_graph_equal(expected=expected_graph, actual=actual_graph)

    def test_idc_star_line_4(self):
        r"""Test line 4 of the IDC* algorithm.

        Check that line 4 or IDC* works correctly moves :math:`Z, D` (with
        :math:`D` being redundant due to graph structure) to the
        subscript of :math:`Y_\mathbf{x}`, to obtain :math:`P(Y_{X',Z} | X )`,
        and calls IDC* with this query recursively.
        """
        input_query = P(Y @ ~X | X, Z, D)
        expected_output_query = P(Y @ (~X, Z) | X)
        new_delta = {X, Z, D}
        new_event = {Y @ ~X}
        graph = figure_9c.graph
        for counterfactual in [Z, D]:
            # self.assertTrue(are_d_separated(graph.remove_outgoing_edges_from( {counterfactual} ), counterfactual, new_event))
            counterfactual_value = Variable(counterfactual.name)
            parents = new_delta - {counterfactual}
            children = {g.intervene(counterfactual_value) for g in new_event}
            # self.assert_expr_equal( P( Y @ {X, counterfactual}  | new_event - {counterfactual}), P(children | parents))
