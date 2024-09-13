"""Tests for Tian and Pearl's Identify algorithm.

.. [huang08a] https://link.springer.com/article/10.1007/s10472-008-9101-x.
.. [correa20a] https://proceedings.neurips.cc/paper/2020/file/7b497aa1b2a83ec63d1777a88676b0c2-Paper.pdf.
.. [correa22a] https://proceedings.mlr.press/v162/correa22a/correa22a.pdf.
.. [tikka20a] https://github.com/santikka/causaleffect/blob/master/R/compute.c.factor.R.
.. [tikka20b] https://github.com/santikka/causaleffect/blob/master/R/identify.R.
.. [tian03a] https://ftp.cs.ucla.edu/pub/stat_ser/R290-L.pdf.
"""

import logging

from tests.test_algorithm import cases
from y0.algorithm.tian_id import (
    compute_ancestral_set_q_value,
    compute_c_factor,
    compute_c_factor_conditioning_on_topological_predecessors,
    compute_c_factor_marginalizing_over_topological_successors,
    compute_q_value_of_variables_with_low_topological_ordering_indices,
    identify_district_variables,
)
from y0.algorithm.transport import transport_variable
from y0.dsl import (
    PP,
    TARGET_DOMAIN,
    W1,
    W2,
    W3,
    W4,
    W5,
    X1,
    X2,
    Fraction,
    One,
    P,
    Pi1,
    Pi2,
    Product,
    R,
    Sum,
    W,
    X,
    Y,
    Z,
    Zero,
)
from y0.graph import NxMixedGraph

# From [correa22a]_, Figure 2a.
figure_2a_graph = NxMixedGraph.from_edges(
    directed=[
        (Z, X),
        (Z, Y),
        (X, Y),
        (X, W),
        (W, Y),
    ],
    undirected=[(Z, X), (W, Y)],
)

# From [correa20a]_, Figure 1b.
soft_interventions_figure_1b_graph = NxMixedGraph.from_edges(
    directed=[
        (X1, Z),
        (X1, X2),
        (X1, Y),
        (X2, Y),
        (transport_variable(Y), Y),
    ],
    undirected=[],
)

# From [correa20a]_, Figure 2a.
soft_interventions_figure_2a_graph = NxMixedGraph.from_edges(
    directed=[
        (R, Z),
        (W, X),
        (X, Z),
        (Z, Y),
    ],
    undirected=[
        (R, Y),
        (W, R),
        (W, X),
        (W, Z),
        (X, Y),
    ],
)

# From [correa20a]_, Figure 2d.
soft_interventions_figure_2d_graph = NxMixedGraph.from_edges(
    directed=[
        (R, Z),
        (X, Z),
        (W, X),
        (Z, Y),
        (transport_variable(W), W),
    ],
    undirected=[
        (R, Y),
        (W, R),
        (W, X),
        (X, Y),
    ],
)

# From [correa20a]_, Figure 3, and corresponding to pi* with the intervention sigma* not applied.
soft_interventions_figure_3_graph = NxMixedGraph.from_edges(
    directed=[
        (R, W),
        (W, X),
        (X, Z),
        (Z, Y),
        (X, Y),
    ],
    undirected=[
        (R, Z),
        (W, Y),
        (W, X),
        (R, X),
    ],
)

tian_pearl_figure_9a_graph = NxMixedGraph.from_edges(
    directed=[
        (W1, W2),
        (W2, X),
        (W3, W4),
        (W4, X),
        (X, Y),
    ],
    undirected=[
        (W1, W3),
        (W3, W5),
        (W4, W5),
        (W2, W3),
        (W1, X),
        (W1, Y),
    ],
)


logger = logging.getLogger(__name__)


class TestIdentify(cases.GraphTestCase):
    """Test the IDENTIFY algorithm (Algorithm 5 of [correa22a]_).

    Source: The example on page 7 of [correa20a]_ (using Figure 1d).
    Note that [correa20a]_ and [correa22a]_ use the same specification of the IDENTIFY
    algorithm. Both papers have a typo on Line 4. See [huang08a]_ for the correct
    version of Line 4.
    """

    def test_identify_preprocessing(self):
        """Test the preprocessing checks in this implementation of IDENTIFY."""
        # Raises a TypeError because the graph has only one C-component: {R,X,W,Z,Y} and our input district T
        # is a subset and therefore not a C-component.
        self.assertRaises(
            KeyError,
            identify_district_variables,
            input_variables=frozenset({Y, R}),
            input_district=frozenset({R, X, W, Z}),
            district_probability=PP[TARGET_DOMAIN](R, W, X, Y, Z),
            graph=soft_interventions_figure_3_graph,
            topo=list(soft_interventions_figure_3_graph.topological_sort()),
        )
        # Raises a KeyError because a variable in T is not in the topo.
        self.assertRaises(
            KeyError,
            identify_district_variables,
            input_variables=frozenset({X, Z}),
            input_district=frozenset({R, X, W, Z}),
            district_probability=PP[TARGET_DOMAIN](R, W, X, Y, Z),
            graph=soft_interventions_figure_3_graph,
            topo=[W, X, Z, Y],
        )
        # Raises a TypeError because G_{X,Y,Z} has two districts and there should be at most one.
        # {X,Y} happen to be in two different districts of G_{X,Y,Z}.
        self.assertRaises(
            TypeError,
            identify_district_variables,
            input_variables=frozenset({X, Y}),
            input_district=frozenset({X, Y, Z}),
            district_probability=PP[TARGET_DOMAIN](R, W, X, Y, Z),
            graph=soft_interventions_figure_3_graph,
            topo=list(soft_interventions_figure_3_graph.topological_sort()),
        )
        # Raises a TypeError because the input district probability has an unrecognized format.
        test_4_identify_input_variables = {Z}  # A
        test_4_identify_input_district = {Z}  # B
        # test_4_district_probability = PP[Population("pi1")](Z | X1)  # Q
        self.assertRaises(
            TypeError,
            identify_district_variables,
            input_variables=frozenset(test_4_identify_input_variables),
            input_district=frozenset(test_4_identify_input_district),
            district_probability=[],
            graph=soft_interventions_figure_1b_graph,
            topo=list(soft_interventions_figure_1b_graph.topological_sort()),
        )
        self.assertRaises(
            TypeError,
            identify_district_variables,
            input_variables=frozenset(test_4_identify_input_variables),
            input_district=frozenset(test_4_identify_input_district),
            district_probability=One(),
            graph=soft_interventions_figure_1b_graph,
            topo=list(soft_interventions_figure_1b_graph.topological_sort()),
        )
        self.assertRaises(
            TypeError,
            identify_district_variables,
            input_variables=frozenset(test_4_identify_input_variables),
            input_district=frozenset(test_4_identify_input_district),
            district_probability=Zero(),
            graph=soft_interventions_figure_1b_graph,
            topo=list(soft_interventions_figure_1b_graph.topological_sort()),
        )

    def test_identify_1(self):
        """Test Line 2 of Algorithm 5 of [correa22a]_.

        This tests the case where A == C.
        """
        # π_star = Pi_star = Variable(f"π*")
        test_1_identify_input_variables = {Z}  # A
        test_1_identify_input_district = {Z}  # B

        # @cthoyt @JZ The next two commented-out lines produce a mypy error:
        test_1_district_probability = PP[Pi1](Z | X1)
        # error: Type application targets a non-generic function or class  [misc]
        # test_transport.py uses a similar syntax and does not trigger the error,
        #   so I'm probably missing something simple.
        # test_1_district_probability = PP[pi1](Z | X1)  # Q
        result = identify_district_variables(
            input_variables=frozenset(test_1_identify_input_variables),
            input_district=frozenset(test_1_identify_input_district),
            district_probability=test_1_district_probability,
            graph=soft_interventions_figure_1b_graph,
            topo=list(soft_interventions_figure_1b_graph.topological_sort()),
        )
        logger.warning("Result of identify() call for test_identify_1 is " + result.to_latex())
        self.assert_expr_equal(result, PP[Pi1](Z | X1))

    def test_identify_2(self):
        """Test Line 3 of Algorithm 5 of [correa22a]_.

        This tests the case where A == T.
        Sources: a modification of the example following Theorem 2 in [correa20a]_
        and the paragraph at the end of section 4 in [correa20a]_.
        """
        result1 = identify_district_variables(
            input_variables=frozenset({R, Y}),
            input_district=frozenset({W, R, X, Z, Y}),
            district_probability=PP[TARGET_DOMAIN](
                W, R, X, Z, Y
            ),  # This is a c-factor if the input variables comprise a c-component
            graph=soft_interventions_figure_2a_graph,
            topo=list(soft_interventions_figure_2a_graph.topological_sort()),
        )
        logger.warning("Result of identify() call for test_identify_2 part 1 is " + str(result1))
        self.assertIsNone(result1)
        result2 = identify_district_variables(
            input_variables=frozenset({Z, R}),
            input_district=frozenset({R, X, W, Z}),
            district_probability=PP[TARGET_DOMAIN](R, W, X, Z),
            graph=soft_interventions_figure_3_graph.subgraph(vertices={R, Z, X, W}),
            topo=list(soft_interventions_figure_3_graph.topological_sort()),
        )
        self.assertIsNone(result2)

    def test_identify_3(self):
        """Test Lines 4-7 of Algorithm 5 of [correa22a]_.

        Source: the example in section 4 of [correa20a]_, which returns FAIL (i.e., None).
        """
        test_3_identify_input_variables = {R, X}
        test_3_identify_input_district = {R, X, W, Y}
        test_3_district_probability = PP[Pi1]((Y, W)).conditional([R, X, Z]) * PP[Pi1](R, X)
        result1 = identify_district_variables(
            input_variables=frozenset(test_3_identify_input_variables),
            input_district=frozenset(test_3_identify_input_district),
            district_probability=test_3_district_probability,
            graph=soft_interventions_figure_2d_graph,
            topo=list(soft_interventions_figure_2d_graph.topological_sort()),
        )
        logger.warning("Result of identify() call for test_identify_3 is " + str(result1))
        self.assertIsNone(result1)
        result2 = identify_district_variables(
            input_variables=frozenset({Z, R}),
            input_district=frozenset({R, X, W, Y, Z}),
            district_probability=PP[TARGET_DOMAIN](R, W, X, Y, Z),
            graph=soft_interventions_figure_3_graph,
            topo=list(soft_interventions_figure_3_graph.topological_sort()),
        )
        logger.warning("Result of identify() call for test_identify_3 is " + str(result2))
        self.assertIsNone(result2)

    def test_identify_4(self):
        """Further test Lines 4-7 of Algorithm 5 of [correa22a]_.

        Source: the example from page 29 of [tian03a]_.

        Note: Tian and Pearl provide a simpler result that is due to using probabilistic
        axioms to simplify the formula. This result is what we get when running Santikka's
        implementation of identify in their R package, Causal Effect ([tikka20b]_), and is more
        complex in its structure but easier to code for an initial Python implementation.
        """
        result_piece_1 = Product.safe(
            [
                P(W1),
                P(W3 | W1),
                P(W2 | (W3, W1)),
                P(X | (W1, W3, W2, W4)),
                P(Y | (W1, W3, W2, W4, X)),
            ]
        )
        result_piece_2_part_1 = Fraction(
            Sum.safe(Sum.safe(result_piece_1, [W3]), [W2, X, Y]), One()
        )  # Q[W1]/Q[\emptyset]
        result_piece_2_part_2 = Fraction(
            Sum.safe(Sum.safe(result_piece_1, [W3]), [Y]),
            Sum.safe(Sum.safe(result_piece_1, [W3]), [X, Y]),
        )  # Q[X]/Q[W2]
        result_piece_2_part_3 = Fraction(
            Sum.safe(result_piece_1, [W3]), Sum.safe(Sum.safe(result_piece_1, [W3]), [Y])
        )  # Q[Y]/Q[X]
        result_piece_2 = Product.safe(
            [result_piece_2_part_1, result_piece_2_part_2, result_piece_2_part_3]
        )
        expected_result = Fraction(
            Sum.safe(result_piece_2, [W1]),
            Sum.safe(Sum.safe(result_piece_2, [W1]), [Y]),
        )  # Q[X,Y]/Q[X]
        result_4 = identify_district_variables(
            input_variables=frozenset({Y}),
            input_district=frozenset({X, Y, W1, W2, W3, W4, W5}),
            district_probability=P(W1, W2, W3, W4, W5, X, Y),
            graph=tian_pearl_figure_9a_graph,
            topo=list(tian_pearl_figure_9a_graph.topological_sort()),
        )
        self.assert_expr_equal(result_4, expected_result)

    def test_identify_4_with_population_probabilities(self):
        """Further test Lines 4-7 of Algorithm 5 of [correa22a]_.

        Source: the example from page 29 of [tian03a]_, requiring all probabilities
        be specified as population probabilities.
        """
        result_piece_1 = Product.safe(
            [
                PP[Pi1](W1),
                PP[Pi1](W3 | W1),
                PP[Pi1](W2 | (W3, W1)),
                PP[Pi1](X | (W1, W3, W2, W4)),
                PP[Pi1](Y | (W1, W3, W2, W4, X)),
            ]
        )
        result_piece_2_part_1 = Fraction(
            Sum.safe(Sum.safe(result_piece_1, [W3]), [W2, X, Y]), One()
        )  # Q[W1]/Q[\emptyset]
        result_piece_2_part_2 = Fraction(
            Sum.safe(Sum.safe(result_piece_1, [W3]), [Y]),
            Sum.safe(Sum.safe(result_piece_1, [W3]), [X, Y]),
        )  # Q[X]/Q[W2]
        result_piece_2_part_3 = Fraction(
            Sum.safe(result_piece_1, [W3]), Sum.safe(Sum.safe(result_piece_1, [W3]), [Y])
        )  # Q[Y]/Q[X]
        result_piece_2 = Product.safe(
            [result_piece_2_part_1, result_piece_2_part_2, result_piece_2_part_3]
        )
        expected_result = Fraction(
            Sum.safe(result_piece_2, [W1]),
            Sum.safe(Sum.safe(result_piece_2, [W1]), [Y]),
        )  # Q[X,Y]/Q[X]
        result_4 = identify_district_variables(
            input_variables=frozenset({Y}),
            input_district=frozenset({X, Y, W1, W2, W3, W4, W5}),
            district_probability=PP[Pi1](W1, W2, W3, W4, W5, X, Y),
            graph=tian_pearl_figure_9a_graph,
            topo=list(tian_pearl_figure_9a_graph.topological_sort()),
        )
        logger.warning("Result from identify_district_variables: " + result_4.to_latex())
        logger.warning("  Expected result: " + expected_result.to_latex())
        self.assert_expr_equal(result_4, expected_result)


class TestComputeCFactor(cases.GraphTestCase):
    """Test the "compute_c_factor" subroutine of Tian and Pearl's identify algorithm as implemented by [tikka20a].

    This subroutine applies Lemma 1 and Lemma 4 of [tian03a]_.
    """

    expected_result_1 = Product.safe(
        [P(W1), P(W3 | W1), P(W2 | (W3, W1)), P(X | (W1, W3, W2, W4)), P(Y | (W1, W3, W2, W4, X))]
    )
    expected_result_2_part_1 = Fraction(
        Sum.safe(Sum.safe(expected_result_1, [W3]), [W2, X, Y]), One()
    )  # Q[W1]/Q[\emptyset]
    expected_result_2_part_2 = Fraction(
        Sum.safe(Sum.safe(expected_result_1, [W3]), [Y]),
        Sum.safe(Sum.safe(expected_result_1, [W3]), [X, Y]),
    )  # Q[X]/Q[W2]
    expected_result_2_part_3 = Fraction(
        Sum.safe(expected_result_1, [W3]), Sum.safe(Sum.safe(expected_result_1, [W3]), [Y])
    )  # Q[Y]/Q[X]
    expected_result_2 = Product.safe(
        [expected_result_2_part_1, expected_result_2_part_2, expected_result_2_part_3]
    )

    def test_compute_c_factor_1(self):
        """First test of the compute C factor subroutine, based on the example on page 29 of [tian03a]."""
        result_1 = compute_c_factor(
            district=[Y, W1, W3, W2, X],
            subgraph_variables=[X, W4, W2, W3, W1, Y],
            subgraph_probability=P(W1, W2, W3, W4, X, Y),
            graph_topo=list(tian_pearl_figure_9a_graph.topological_sort()),
        )
        self.assert_expr_equal(result_1, self.expected_result_1)

    def test_compute_c_factor_2(self):
        """Second test of the compute C factor subroutine, based on the example on page 29 of [tian03a]."""
        result_2 = compute_c_factor(
            district=[W1, X, Y],
            subgraph_variables=[W1, W2, X, Y],
            subgraph_probability=Sum.safe(self.expected_result_1, [W3]),
            graph_topo=list(tian_pearl_figure_9a_graph.topological_sort()),
        )
        self.assert_expr_equal(result_2, self.expected_result_2)

    def test_compute_c_factor_3(self):
        """Third test of the compute C factor subroutine, based on the example on page 29 of [tian03a]."""
        result_3 = compute_c_factor(
            district=[Y],
            subgraph_variables=[X, Y],
            subgraph_probability=Sum.safe(self.expected_result_2, [W1]),
            graph_topo=list(tian_pearl_figure_9a_graph.topological_sort()),
        )
        expected_result_3 = Fraction(
            Sum.safe(self.expected_result_2, [W1]),
            Sum.safe(Sum.safe(self.expected_result_2, [W1]), [Y]),
        )  # Q[X,Y]/Q[X]
        self.assert_expr_equal(result_3, expected_result_3)

    def test_compute_c_factor_4(self):
        """Fourth test of the Compute C Factor function.

        Source: [tian03a], the example in section 4.6.
        """
        topo = list(tian_pearl_figure_9a_graph.topological_sort())
        district = [W1, W3, W2, X, Y]
        subgraph_variables = [W1, W3, W2, W4, X, Y]
        subgraph_probability = P(W1, W3, W2, W4, X, Y)
        expected_result_4 = (
            P(Y | [W1, W2, W3, W4, X])
            * P(X | [W1, W2, W3, W4])
            * P(W2 | [W1, W3])
            * P(W3 | W1)
            * P(W1)
        )
        result_4 = compute_c_factor(
            district=district,
            subgraph_variables=subgraph_variables,
            subgraph_probability=subgraph_probability,
            graph_topo=topo,
        )
        self.assert_expr_equal(result_4, expected_result_4)
        # TODO: As a test, have Q condition on variables in the C factor and see what happens
        #       when you apply Lemma 4(ii) but especially Lemma 1.
        # TODO: Currently when the input Q is an instance of Sum, we immediately
        #       apply Lemma 4(ii). And that's because in theory we should never
        #       have a sum applied to a simple probability. Make sure we can't
        #       have Q set to something like $Sum_{W3}{W4 | W3}$.

    def test_compute_c_factor_5(self):
        """Fifth test of the Compute C Factor function.

        Testing Lemma 1 as called from _compute_c_factor,
        conditioning on a variable as part of the input Q value for the graph.
        Source: [tian03a], the example in section 4.6.
        """
        topo = list(tian_pearl_figure_9a_graph.topological_sort())
        district = [W1, W3, W2, X, Y]
        subgraph_variables = [W1, W3, W2, W4, X, Y]
        subgraph_probability = P(W1, W3, W2, W4, X, Y | W5)
        expected_result_5 = (
            P(Y | [W1, W2, W3, W4, X, W5])
            * P(X | [W1, W2, W3, W4, W5])
            * P(W2 | [W1, W3, W5])
            * P(W3 | [W1, W5])
            * P(W1 | W5)
        )
        result_5 = compute_c_factor(
            district=district,
            subgraph_variables=subgraph_variables,
            subgraph_probability=subgraph_probability,
            graph_topo=topo,
        )
        self.assert_expr_equal(result_5, expected_result_5)

    def test_compute_c_factor_5_with_population_probabilities(self):
        """Fifth test of the Compute C Factor function, using population probabilities.

        Testing Lemma 1 as called from _compute_c_factor,
        conditioning on a variable as part of the input Q value for the graph.
        Source: [tian03a], the example in section 4.6.
        """
        topo = list(tian_pearl_figure_9a_graph.topological_sort())
        district = [W1, W3, W2, X, Y]
        subgraph_variables = [W1, W3, W2, W4, X, Y]
        subgraph_probability = PP[Pi2](W1, W3, W2, W4, X, Y | W5)
        expected_result_5 = (
            PP[Pi2](Y | [W1, W2, W3, W4, X, W5])
            * PP[Pi2](X | [W1, W2, W3, W4, W5])
            * PP[Pi2](W2 | [W1, W3, W5])
            * PP[Pi2](W3 | [W1, W5])
            * PP[Pi2](W1 | W5)
        )
        result_5 = compute_c_factor(
            district=district,
            subgraph_variables=subgraph_variables,
            subgraph_probability=subgraph_probability,
            graph_topo=topo,
        )
        logger.warning(
            "In test_compute_c_factor_5_with_population_probabilities: expected_result = "
            + expected_result_5.to_latex()
        )
        logger.warning(
            "In test_compute_c_factor_5_with_population_probabilities: result = "
            + result_5.to_latex()
        )
        self.assert_expr_equal(result_5, expected_result_5)

    def test_compute_c_factor_6(self):
        """Sixth test of the Compute C Factor function.

        Here we test a case in which the input probability is neither a product, sum, fraction,
        nor a simple probability.
        Source: derivative from [tian03a], the example in section 4.6.
        """
        # TODO: Discuss whether we want identify_district_variables() and _compute_c_factor()
        # to handle expressions of type One, Zero, or QFactor.
        topo = list(tian_pearl_figure_9a_graph.topological_sort())
        district = [W1, W3, W2, X, Y]
        subgraph_variables = [W1, W3, W2, W4, X, Y]
        subgraph_probability = One()
        self.assertRaises(
            TypeError,
            compute_c_factor,
            district=district,
            subgraph_variables=subgraph_variables,
            subgraph_probability=subgraph_probability,
            graph_topo=topo,
        )


class TestComputeCFactorConditioningOnTopologicalPredecessors(cases.GraphTestCase):
    """Test the use of Lemma 1, part (i), of [tian03a]_ to compute a C factor."""

    def test_compute_c_factor_conditioning_on_topological_predecessors_part_1(self):
        """First test of Lemma 1, part (i) (Equation 37 in [tian03a]_.

        Source: The example on p. 30 of [Tian03a]_, run initially through [tikka20a]_.
        """
        topo = [W1, W3, W2, W4, X, Y]
        part_1_graph = tian_pearl_figure_9a_graph.subgraph([Y, X, W1, W2, W3, W4])
        result_1 = compute_c_factor_conditioning_on_topological_predecessors(
            district=[Y, W1, W3, W2, X],
            topo=topo,
            graph_probability=part_1_graph.joint_probability(),
        )
        self.assert_expr_equal(
            result_1,
            Product.safe(
                [
                    P(W1),
                    P(W3 | W1),
                    P(W2 | (W3, W1)),
                    P(X | (W1, W3, W2, W4)),
                    P(Y | (W1, W3, W2, W4, X)),
                ]
            ),
        )
        # District contains no variables
        self.assertRaises(
            TypeError,
            compute_c_factor_conditioning_on_topological_predecessors,
            district=[],
            topo=topo,
            graph_probability=part_1_graph.joint_probability(),
        )
        # District variable not in topo set
        self.assertRaises(
            KeyError,
            compute_c_factor_conditioning_on_topological_predecessors,
            district=[Y, W1, W3, W2, X, Z],
            topo=topo,
            graph_probability=part_1_graph.joint_probability(),
        )

    def test_compute_c_factor_conditioning_on_topological_predecessors_part_2(self):
        """Second test of Lemma 1, part (i) (Equation 37 in [tian03a]_.

        This one handles a graph_probability conditioning on variables.
        Source: The example on p. 30 of [Tian03a]_, run initially through [tikka20a]_.
        """
        # working with tian_pearl_figure_9a_graph.subgraph([Y, X, W1, W2, W3, W4])
        topo = [W1, W3, W2, W4, X, Y]
        result_1 = compute_c_factor_conditioning_on_topological_predecessors(
            district=[Y, W1, W3, W2, X],
            topo=topo,
            graph_probability=P(Y, X, W1, W2, W3, W4 | W5),
        )
        self.assert_expr_equal(
            result_1,
            Product.safe(
                [
                    P(W1 | W5),
                    P(W3 | (W1, W5)),
                    P(W2 | (W3, W1, W5)),
                    P(X | (W1, W3, W2, W4, W5)),
                    P(Y | (W1, W3, W2, W4, X, W5)),
                ]
            ),
        )


class TestComputeCFactorMarginalizingOverTopologicalSuccessors(cases.GraphTestCase):
    """Test the use of Lemma 4, part (ii), of [tian03a]_ to compute a C factor."""

    result_piece = Product.safe(
        [
            P(W1),
            P(W3 | W1),
            P(W2 | (W3, W1)),
            P(X | (W1, W3, W2, W4)),
            P(Y | (W1, W3, W2, W4, X)),
        ]
    )
    expected_result_1_part_1 = Fraction(
        Sum.safe(Sum.safe(result_piece, [W3]), [W2, X, Y]), One()
    )  # Q[W1]/Q[\emptyset]
    expected_result_1_part_2 = Fraction(
        Sum.safe(Sum.safe(result_piece, [W3]), [Y]), Sum.safe(Sum.safe(result_piece, [W3]), [X, Y])
    )  # Q[X]/Q[W2]
    expected_result_1_part_3 = Fraction(
        Sum.safe(result_piece, [W3]), Sum.safe(Sum.safe(result_piece, [W3]), [Y])
    )  # Q[Y]/Q[X]
    #
    # A future version of Y0 could improve simplification of mathematical expressions so that
    # a test using the version of "expected_result_1" commented out below would also pass.
    # expected_result_1_num = Product.safe(
    #    [
    #        Sum.safe(Sum.safe(result_piece, [W3]), [W2, X, Y]),
    #        Sum.safe(result_piece, [W3]),
    #        Sum.safe(Sum.safe(result_piece, [W3]), [Y]),
    #    ]
    # )
    # expected_result_1_den = Product.safe(
    #    [
    #        # Sum.safe(Sum.safe(result_piece, [W3]),[W1, W2, X, Y]),
    #        One(),
    #        Sum.safe(Sum.safe(result_piece, [W3]), [X, Y]),
    #        Sum.safe(Sum.safe(result_piece, [W3]), [Y]),
    #    ]
    # )
    # expected_result_1 = Fraction(expected_result_1_num, expected_result_1_den)
    expected_result_1 = Product.safe(
        [expected_result_1_part_1, expected_result_1_part_2, expected_result_1_part_3]
    )
    expected_result_2_num = Sum.safe(expected_result_1, [W1])
    expected_result_2_den = Sum.safe(Sum.safe(expected_result_1, [W1]), [Y])
    expected_result_2 = Fraction(expected_result_2_num, expected_result_2_den)

    # Same thing, but with population probabilities
    result_piece_pp = Product.safe(
        [
            PP[Pi1](W1),
            PP[Pi1](W3 | W1),
            PP[Pi1](W2 | (W3, W1)),
            PP[Pi1](X | (W1, W3, W2, W4)),
            PP[Pi1](Y | (W1, W3, W2, W4, X)),
        ]
    )
    expected_result_1_part_1_pp = Fraction(
        Sum.safe(Sum.safe(result_piece_pp, [W3]), [W2, X, Y]), One()
    )  # Q[W1]/Q[\emptyset]
    expected_result_1_part_2_pp = Fraction(
        Sum.safe(Sum.safe(result_piece_pp, [W3]), [Y]),
        Sum.safe(Sum.safe(result_piece_pp, [W3]), [X, Y]),
    )  # Q[X]/Q[W2]
    expected_result_1_part_3_pp = Fraction(
        Sum.safe(result_piece_pp, [W3]), Sum.safe(Sum.safe(result_piece_pp, [W3]), [Y])
    )  # Q[Y]/Q[X]
    expected_result_1_pp = Product.safe(
        [expected_result_1_part_1_pp, expected_result_1_part_2_pp, expected_result_1_part_3_pp]
    )
    expected_result_2_num_pp = Sum.safe(expected_result_1_pp, [W1])
    expected_result_2_den_pp = Sum.safe(Sum.safe(expected_result_1_pp, [W1]), [Y])
    expected_result_2_pp = Fraction(expected_result_2_num_pp, expected_result_2_den_pp)

    def test_compute_c_factor_marginalizing_over_topological_successors_part_1(self):
        """First test of Lemma 4, part (ii) (Equations 71 and 72 in [tian03a]_.

        Source: The example on p. 30 of [Tian03a]_, run initially through [tikka20a]_.
        """
        result = compute_c_factor_marginalizing_over_topological_successors(
            district={W1, X, Y},
            graph_probability=Sum.safe(self.result_piece, [W3]),
            topo=list(tian_pearl_figure_9a_graph.subgraph({W1, W2, X, Y}).topological_sort()),
        )
        logger.warning(
            "In first test of Lemma 4(ii): expecting this result: " + str(self.expected_result_1)
        )
        self.assert_expr_equal(result, self.expected_result_1)

    def test_compute_c_factor_marginalizing_over_topological_successors_part_2(self):
        """Second test of Lemma 4, part (ii) (Equations 71 and 72 in [tian03a]_.

        Source: The example on p. 30 of [Tian03a]_, run initially through [tikka20a]_.
        """
        logger.warning(
            "In second test of Lemma 4(ii): expecting this result: " + str(self.expected_result_2)
        )
        logger.warning("Expected_result_1 = " + str(self.expected_result_1))
        result = compute_c_factor_marginalizing_over_topological_successors(
            district={Y},
            graph_probability=Sum.safe(self.expected_result_1, [W1]),
            topo=list(tian_pearl_figure_9a_graph.subgraph({X, Y}).topological_sort()),
        )
        self.assert_expr_equal(result, self.expected_result_2)

    def test_compute_c_factor_marginalizing_over_topological_successors_part_3(self):
        """First test of Equations 71 and 72 in [tian03a]_ using population probabilities.

        Source: The example on p. 30 of [Tian03a]_, run initially through [tikka20a]_.
        """
        result = compute_c_factor_marginalizing_over_topological_successors(
            district={W1, X, Y},
            graph_probability=Sum.safe(self.result_piece_pp, [W3]),
            topo=list(tian_pearl_figure_9a_graph.subgraph({W1, W2, X, Y}).topological_sort()),
        )
        logger.warning(
            "In first test of Lemma 4(ii): expecting this result: "
            + self.expected_result_1_pp.to_latex()
        )
        self.assert_expr_equal(result, self.expected_result_1_pp)

    def test_compute_c_factor_marginalizing_over_topological_successors_part_4(self):
        """Second test of Equations 71 and 72 in [tian03a]_ using population probabilities.

        Source: The example on p. 30 of [Tian03a]_, run initially through [tikka20a]_.
        """
        logger.warning(
            "In second test of Lemma 4(ii): expecting this result: "
            + self.expected_result_2.to_latex()
        )
        logger.warning("Expected_result_1 = " + self.expected_result_1_pp.to_latex())
        result = compute_c_factor_marginalizing_over_topological_successors(
            district={Y},
            graph_probability=Sum.safe(self.expected_result_1_pp, [W1]),
            topo=list(tian_pearl_figure_9a_graph.subgraph({X, Y}).topological_sort()),
        )
        logger.warning("Expected result = " + self.expected_result_2_pp.to_latex())
        self.assert_expr_equal(result, self.expected_result_2_pp)


class TestComputeQValueOfVariablesWithLowTopologicalOrderingIndices(cases.GraphTestCase):
    """Test the use of Equation 72 in Lemma 1, part (ii), of [tian03a]_."""

    def test_compute_q_value_of_variables_with_low_topological_ordering_indices_part_1(self):
        """First test of Equation 72 in [tian03a]_.

        Source: RJC's mind.
        """
        topo = list(figure_2a_graph.subgraph({Z, X, Y, W}).topological_sort())
        result = compute_q_value_of_variables_with_low_topological_ordering_indices(
            vertex=W,
            graph_probability=P(Y | W, X, Z) * P(W | X, Z) * P(X | Z) * P(Z),
            topo=topo,
        )
        self.assert_expr_equal(
            result, Sum.safe(P(Y | W, X, Z) * P(W | X, Z) * P(X | Z) * P(Z), [Y])
        )
        # Variable not in the graph
        self.assertRaises(
            KeyError,
            compute_q_value_of_variables_with_low_topological_ordering_indices,
            vertex={R},
            graph_probability=P(Y | W, X, Z) * P(W | X, Z) * P(X | Z) * P(Z),
            topo=topo,
        )

    def test_compute_q_value_of_variables_with_low_topological_ordering_indices_part_2(self):
        r"""Second test of Equation 72 in [tian03a]_, checking $Q[H^{(0)}]=Q[\emptyset]$.

        Source: RJC's mind.
        """
        topo = list(figure_2a_graph.subgraph({Z, X, Y, W}).topological_sort())
        result = compute_q_value_of_variables_with_low_topological_ordering_indices(
            vertex=None,
            graph_probability=P(Y | W, X, Z) * P(W | X, Z) * P(X | Z) * P(Z),
            topo=topo,
        )
        self.assert_expr_equal(result, One())


class TestComputeAncestralSetQValue(cases.GraphTestCase):
    """Test the use of Lemma 3 (i.e., Equation 69) of [tian03a]_ to compute a C factor."""

    def test_compute_ancestral_set_q_value_part_1(self):
        """First test of Lemma 3 in [tian03a]_ (Equation 69).

        Source: The example on p. 30 of [Tian03a]_, run initially through [tikka20a]_.
        """
        topo = [W1, W3, W5, W2, W4, X, Y]
        # Q_T = Q[{W1, W2, W3, X, Y}]
        subgraph_probability = Product.safe(
            [
                P(W1),
                P(W3 | W1),
                P(W2 | (W3, W1)),
                P(X | (W1, W3, W2, W4)),
                P(Y | (W1, W3, W2, W4, X)),
            ]
        )
        # The ancestors of {Y} in Figure 9(c) of [tian03a]_
        ancestral_set = {W1, W2, X, Y}
        subgraph_variables = {W1, W2, W3, X, Y}  # T in Figure 9(c) of [tian03a]_
        result_1 = compute_ancestral_set_q_value(
            ancestral_set=ancestral_set,
            subgraph_variables=subgraph_variables,
            subgraph_probability=subgraph_probability,
            graph_topo=topo,
        )
        expected_result_1 = Sum.safe(subgraph_probability, [W3])
        self.assert_expr_equal(expected_result_1, result_1)
