"""Tests for counterfactual transportability. See https://proceedings.mlr.press/v162/correa22a/correa22a.pdf."""

import logging
import unittest

from y0.algorithm.counterfactual_transportability import (
    do_ctf_factor_factorization,
    get_ancestors_of_counterfactual,
    get_ctf_factor_query,
    get_ctf_factors,
    is_ctf_factor_form,
    make_selection_diagram,
    minimize,
    same_district,
    simplify,
)
from y0.algorithm.transport import transport_variable
from y0.dsl import (
    CounterfactualVariable,
    Expression,
    Intervention,
    P,
    Sum,
    Variable,
    W,
    X,
    Y,
    Z,
    Zero,
    get_outcomes_and_treatments,
)
from y0.graph import NxMixedGraph
from y0.mutate import canonicalize

# From Correa, Lee, and Bareinboim 2022.
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

logger = logging.getLogger(__name__)


class TestGetAncestorsOfCounterfactual(unittest.TestCase):
    """Test getting the ancestors of a counterfactual.

    This test follows Correa, Lee, and Bareinboim 2022, Definition 2.1 and Example 2.1.
    """

    # TODO: It seems important to be able to represent Y_{x_0}, not just Y_{X}.
    # (That is, the value of Y were we to intervene on X and set its value to x0).
    # That's not yet implemented with Y0's architecture.

    def test_example_2_1_1(self):
        """Test the first result of Example 2.1.

        Note we test whether we can get Z back, Z is just a Variable so
        it's a test of the Union in the return value for the tested function.
        """
        test1_in = CounterfactualVariable(
            name="Y", star=None, interventions=(Intervention(name="X", star=False),)
        )  # Y @ -X
        # test1_in = (Y @ -X)
        test1_out = {
            CounterfactualVariable(
                name="Y", star=None, interventions=(Intervention(name="X", star=False),)
            ),
            CounterfactualVariable(
                name="W", star=None, interventions=(Intervention(name="X", star=False),)
            ),
            Variable(name="Z"),
        }
        result = get_ancestors_of_counterfactual(event=test1_in, graph=figure_2a_graph)
        self.assertTrue(variable in test1_out for variable in result)
        self.assertTrue(test1_in in result)  # Every outcome is its own ancestor

    def test_example_2_1_2(self):
        """Test the second result of Example 2.1."""
        test2_in = CounterfactualVariable(
            name="W",
            star=None,
            interventions=(Intervention(name="Y", star=False), Intervention(name="Z", star=False)),
        )  # W @ -Y @ -Z
        test2_out = {
            CounterfactualVariable(
                name="W", star=None, interventions=(Intervention(name="Z", star=False),)
            ),
            CounterfactualVariable(
                name="X", star=None, interventions=(Intervention(name="Z", star=False),)
            ),
        }
        result = get_ancestors_of_counterfactual(event=test2_in, graph=figure_2a_graph)
        self.assertTrue(variable in test2_out for variable in result)
        self.assertTrue((W @ -Z) in result)  # The outcome, (W @ -Y @ -Z), simplifies to this
        self.assertFalse(test2_in in result)

    def test_example_2_1_3(self):
        """Test the third result of Example 2.1."""
        test3_in = CounterfactualVariable(
            name="Y", star=None, interventions=(Intervention(name="W", star=False),)
        )  # Y @ -W
        test3_out = {
            CounterfactualVariable(
                name="Y", star=None, interventions=(Intervention(name="W", star=False),)
            ),
            Variable(name="X"),
            Variable(name="Z"),
        }
        result = get_ancestors_of_counterfactual(event=test3_in, graph=figure_2a_graph)
        self.assertTrue(variable in test3_out for variable in result)
        self.assertTrue(test3_in in result)  # Every outcome is its own ancestor

    def test_4(self):
        """Test passing in a variable with no interventions.

        The problem reduces to just getting ancestors. Source: out of Richard's head.
        """
        test4_in = Variable(name="Z", star=None)
        test4_out = {Variable(name="Z")}
        result = get_ancestors_of_counterfactual(event=test4_in, graph=figure_2a_graph)
        self.assertTrue(variable in test4_out for variable in result)
        self.assertTrue(test4_in in result)  # Every outcome is its own ancestor

    def test_5(self):
        """Test passing in a variable intervening on itself.

        The problem reduces to just getting ancestors. Source: out of Richard's head.
        """
        test5_in = CounterfactualVariable(
            name="Y", star=None, interventions=(Intervention(name="Y", star=False),)
        )
        test5_out = {
            CounterfactualVariable(
                name="Y", star=None, interventions=(Intervention(name="Y", star=False),)
            )
        }
        result = get_ancestors_of_counterfactual(event=test5_in, graph=figure_2a_graph)
        self.assertTrue(variable in test5_out for variable in result)
        self.assertTrue(test5_in in result)  # Every outcome is its own ancestor


class TestSimplify(unittest.TestCase):
    """Test the simplify algorithm from counterfactual transportability."""

    # TODO: Incorporate a test involving counterfactual unnesting.

    def test_inconsistent_1(self):
        """Test simplifying an inconsistent event.

        Correa et al. specify the output should be 0 if the counterfactual event
        is guaranteed to have probability 0. Source: Richard's mind.
        """
        event = [(Y @ -X, -Y), (Y @ -X, +Y)]
        self.assertEquals(simplify(event), Zero())

    def test_inconsistent_2(self):
        """Second test for simplifying an inconsistent event. Source: Richard's mind."""
        event = [(Y @ -Y, +Y)]
        self.assertEquals(simplify(event), Zero())

    def test_redundant_1(self):
        """First test for simplifying an event with redundant subscripts. Source: Richard's mind."""
        event = [(Y @ -X, -Y), (Y @ -X, -Y)]
        result = simplify(event)
        self.assertEqual(result, [(Y @ -X, -Y)])

    def test_redundant_2(self):
        """Second test for simplifying an event with redundant subscripts. Source: Richard's mind."""
        event = [(Y @ -X, -Y), (Y @ -X, -Y), (X @ -W, -X)]
        self.assertEqual(simplify(event), [(Y @ -X, -Y), (X @ -W, -X)])

    def test_redundant_3(self):
        """Third test for simplifying an event with redundant subscripts.

        (Y @ (-Y,-X), -Y) reduces to (Y @ -Y, -Y) via line 1 of the SIMPLIFY algorithm.
        Source: Jeremy's mind.
        """
        event = [
            (Y @ (-Y, -X), -Y),
            (Y @ -Y, -Y),
            (X @ -W, -X),
        ]
        self.assertEqual(simplify(event), [(Y @ -Y, -Y), (X @ -W, -X)])

    def test_redundant_4(self):
        """Fourth test for simplifying an event with redundant subscripts. Source: Richard's mind."""
        event = [
            (Y @ -Y, -Y),
            (Y @ -Y, -Y),
        ]
        self.assertIsNone(simplify(event))


class TestIsCtfFactorForm(unittest.TestCase):
    """Test whether a set of counterfactual variables are all in counterfactual factor form."""

    # TODO: Incorporate a test involving counterfactual unnesting.

    def test_is_ctf_factor_form(self):
        """From Example 3.3 of Correa, Lee, and Barenboim 2022."""
        event1 = [(Y @ (-Z, -W, -X)), (W @ -X)]  # (Y @ -Z @ -W @ -X)
        self.assertTrue(is_ctf_factor_form(event=event1, graph=figure_2a_graph))

        event2 = [(W @ X), (Z)]
        self.assertTrue(is_ctf_factor_form(event=event2, graph=figure_2a_graph))

        event3 = [(Y @ (-Z, -W)), (W @ -X)]
        self.assertFalse(is_ctf_factor_form(event=event3, graph=figure_2a_graph))

        # Y has parents, so they should be intervened on but are not
        event4 = [(Y)]
        self.assertFalse(is_ctf_factor_form(event=event4, graph=figure_2a_graph))

        # Z has no parents, so this variable is also a ctf-factor
        event5 = [(Z)]
        self.assertTrue(is_ctf_factor_form(event=event5, graph=figure_2a_graph))

        # Z is not a parent of W, so the second counterfactual variable is not a ctf-factor,
        # because it is not a valid counterfactual variable
        event6 = [(Y @ (-Z, -W)), (W @ (-X, Z))]
        self.assertFalse(is_ctf_factor_form(event=event6, graph=figure_2a_graph))


class TestMakeSelectionDiagram(unittest.TestCase):
    """Test the results of creating a selection diagram that is an amalgamation of domain selection diagrams."""

    def test_make_selection_diagram(self):
        """Create Figure 2(b) of Correa, Lee, and Barenboim 2022 from Figures 3(a) and 3(b)."""
        selection_nodes = dict({1: set({Z}), 2: set({W})})
        selection_diagram = make_selection_diagram(
            selection_nodes=selection_nodes, graph=figure_2a_graph
        )
        expected_selection_diagram = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (Z, Y),
                (X, Y),
                (X, W),
                (W, Y),
                (
                    transport_variable(Z),
                    Z,
                ),  # How do we indicate with a superscript that this is from domain 1?
                (
                    transport_variable(W),
                    W,
                ),  # How do we indicate with a superscript that this is from domain 2?
            ],
            undirected=[(Z, X), (W, Y)],
        )
        self.assertEquals(selection_diagram, expected_selection_diagram)


class TestMinimize(unittest.TestCase):
    r"""Test minimizing a set of counterfactual variables.

    Source: last paragraph in Section 4 of Correa, Lee, and Barenboim 2022, before Section 4.1.
    Mathematical expression: ||\mathbf Y_*|| = {||Y_{\mathbf x}|| | Y_{\mathbf x}} \elementof \mathbf Y_*}, and
    ||Y_{\mathbf x}|| = Y_{\mathbf t}, where \mathbf T = \mathbf X \intersect An(Y)_{G_{\overline{\mathbf X}}}}.
    (The math syntax is not necessarily cannonical LaTeX.)
    """

    def test_minimize_1(self):
        """Test the minimize function sending in a single counterfactual variable. Source: out of Richard's head."""
        minimize_graph_1 = NxMixedGraph.from_edges(
            directed=[
                (X, W),
                (W, Y),
            ],
            undirected=[(X, Y)],
        )
        minimize_test1_in = [(Y @ (-W, -X))]
        minimize_test1_out = [(Y @ -W)]
        self.assertEquals(minimize_test1_out, minimize(minimize_test1_in, minimize_graph_1))

    def test_minimize_2(self):
        """Test the minimize function for multiple counterfactual variables. Source: out of Richard's head."""
        minimize_graph_2 = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (X, W),
                (W, Y),
            ],
            undirected=[(X, Y)],
        )
        minimize_test2_in = [(Y @ (-W, -X, -Z)), (W @ (-X, -Z))]
        minimize_test2_out = [(Y @ -W), (W @ -Z)]
        self.assertEquals(minimize_test2_out, minimize(minimize_test2_in, minimize_graph_2))


class TestSameDistrict(unittest.TestCase):
    """Test whether a set of counterfactual variables are in the same district (c-component)."""

    def test_same_district_1(self):
        """Test #1 for whether a set of counterfactual variables are in the same district (c-component).

        Source: out of Richard's head.
        """
        same_district_test1_in = [(Y @ (-X, -W, -Z)), (W @ (-X))]
        self.assertTrue(same_district(same_district_test1_in, figure_2a_graph))

    def test_same_district_2(self):
        """Test #2 for whether a set of counterfactual variables are in the same district (c-component).

        Source: out of Richard's head.
        """
        same_district_test2_in = [(Z), (X @ -Z)]
        self.assertTrue(same_district(same_district_test2_in, figure_2a_graph))

    def test_same_district_3(self):
        """Test #3 for whether a set of counterfactual variables are in the same district (c-component).

        Source: out of Richard's head.
        """
        same_district_test3_in = [(Y @ -Y), (Z), (X @ -Z)]
        self.assertFalse(same_district(same_district_test3_in, figure_2a_graph))


# TODO: Convert all Ctfs to Counterfactual
class TestGetCtfFactors(unittest.TestCase):
    """Test the GetCtfFactors function in counterfactual_transportability.py.

    This is one step in the ctf-factor factorization process. Here we want
    to check that we can separate a joint probability distribution of ctf-factors
    into a set of joint probability distributions that we'll later multiply
    together as per Equation 15 in Correa, Lee, and Barenboim 2022.
    """

    def test_get_ctf_factors(self):
        """Test factoring a set of counterfactual variables by district (c-component).

        Source: Example 4.2 of Correa, Lee, and Barenboim 2022. Note that
                we're not testing the full example, just computing factors
                for the query.
        """
        # TODO: Add more tests.
        get_ctf_factors_test_1_in = [(Y @ (-X, -W, -Z)), (W @ -X), (X @ -Z), (Z)]
        get_ctf_factors_test_1_expected = {((Y @ (-X, -W, -Z)), (W @ -X)), ((X @ -Z), (Z))}
        self.assertSetEqual(
            get_ctf_factors_test_1_expected,
            get_ctf_factors(event=get_ctf_factors_test_1_in, graph=figure_2a_graph),
        )


class TestEquation11(unittest.TestCase):
    """Test deriving a query of ctf-factors from a counterfactual query, following Equation 11 in Correa et al. (2022).

    This is one step in the ctf-factor factorization process. We get syntax inspiration from Line 1 of the ID algorithm.
    """

    def test_equation_11(self):
        """Test deriving a query of ctf factors from a counterfactual query.

        Source: Equation 14 of Correa et al. (2022).
        """
        # Already in ctf-factor form
        test_equation_11_in_1 = [(Y @ (-X, -W, -Z)), (W @ -X), (X @ -Z), (Z)]

        # get_ctf_factor_query() should convert this expression to ctf-factor form
        test_equation_11_in_2 = [(Y @ (-X)), (W @ -X), (X @ -Z), (Z)]

        test_equation_11_expected = Sum.safe(
            P([(Y @ (-X, -W, -Z)), (W @ -X), (X @ -Z), (Z)]), [Z, W]
        )
        self.assert_expr_equal(
            get_ctf_factor_query(event=test_equation_11_in_1, graph=figure_2a_graph),
            test_equation_11_expected,
        )
        self.assert_expr_equal(
            get_ctf_factor_query(event=test_equation_11_in_2, graph=figure_2a_graph),
            test_equation_11_expected,
        )


class TestCtfFactorFactorization(unittest.TestCase):
    """Test factorizing the counterfactual factors corresponding to a query, as per Example 4.2 of Correa et al. (2022).

    This puts together getting the ancestral set of a query, getting the ctf-factors for each element of the set,
    and factorizing the resulting joint distribution according to the C-components of the graph.
    """

    # TODO: Go to cases.GraphTestCase for the function and inherit from there.
    def assert_expr_equal(self, expected: Expression, actual: Expression) -> None:
        """Assert that two expressions are the same. This code is from test_id_alg.py."""
        # TODO: Check with Jeremy: we may wish to move the code into a utils file
        #       in the tests/test_algorithm folder, and shared by test_id_alg.py and this file.
        expected_outcomes, expected_treatments = get_outcomes_and_treatments(query=expected)
        actual_outcomes, actual_treatments = get_outcomes_and_treatments(query=actual)
        self.assertEqual(expected_treatments, actual_treatments)
        self.assertEqual(expected_outcomes, actual_outcomes)
        ordering = tuple(expected.get_variables())
        expected_canonical = canonicalize(expected, ordering)
        actual_canonical = canonicalize(actual, ordering)
        self.assertEqual(
            expected_canonical,
            actual_canonical,
            msg=f"\nExpected: {str(expected_canonical)}\nActual:   {str(actual_canonical)}",
        )

    def test_ctf_factor_factorization(self):
        """Test counterfactual factor factorization as per Equation 16 in Correa et al. (2022).

        Source: Equations 11, 14, and 16 of Correa et al. (2022).
        """
        test_equation_16_in = P(
            [(Y @ -X), (X)]
        )  # Question for Jeremy: should the second term be -X instead of just X?
        # Is the multiplication syntax correct?
        test_equation_16_expected = Sum.safe(
            P([(Y @ (-X, -W, -Z)), (W @ -X)]) * P([(X @ -Z), (Z)]), [Z, W]
        )
        self.assert_expr_equal(
            do_ctf_factor_factorization(test_equation_16_in, figure_2a_graph),
            test_equation_16_expected,
        )


# Next test: convert_counterfactual_variables_to_counterfactual_factor_form
