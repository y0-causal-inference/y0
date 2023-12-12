# -*- coding: utf-8 -*-

"""Tests for counterfactual transportability.

.. [correa22a] https://proceedings.mlr.press/v162/correa22a/correa22a.pdf.
"""

import logging
import unittest

from networkx import NetworkXError

from tests.test_algorithm import cases
from y0.algorithm.counterfactual_transportability import (  # get_counterfactual_factor_query,
    convert_to_counterfactual_factor_form,
    counterfactual_factors_are_transportable,
    do_counterfactual_factor_factorization,
    get_ancestors_of_counterfactual,
    get_counterfactual_factors,
    is_counterfactual_factor_form,
    make_selection_diagram,
    minimize,
    same_district,
    simplify,
)
from y0.algorithm.transport import transport_variable
from y0.dsl import (
    CounterfactualVariable,
    Intervention,
    P,
    R,
    Sum,
    Variable,
    W,
    X,
    Y,
    Z,
    Zero,
)
from y0.graph import NxMixedGraph

# from y0.tests.test_algorithm.cases import GraphTestCase

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

# From [correa22a]_, Figure 3a.
figure_2_graph_domain_1 = NxMixedGraph.from_edges(
    directed=[
        (Z, X),
        (Z, Y),
        (X, Y),
        (X, W),
        (W, Y),
        (transport_variable(Z), Z),
    ],
    undirected=[(Z, X), (W, Y)],
)

# From [correa22a]_, Figure 3b.
figure_2_graph_domain_2 = NxMixedGraph.from_edges(
    directed=[
        (Z, X),
        (Z, Y),
        (X, Y),
        (X, W),
        (W, Y),
        (transport_variable(W), W),
    ],
    undirected=[(Z, X), (W, Y)],
)

logger = logging.getLogger(__name__)


class TestGetAncestorsOfCounterfactual(unittest.TestCase):
    """Test getting the ancestors of a counterfactual.

    This test follows [correa22a]_, Definition 2.1 and Example 2.1.
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

        The problem reduces to just getting ancestors. Source: out of RJC's head.
        """
        test4_in = Variable(name="Z", star=None)
        test4_out = {Variable(name="Z")}
        result = get_ancestors_of_counterfactual(event=test4_in, graph=figure_2a_graph)
        self.assertTrue(variable in test4_out for variable in result)
        self.assertTrue(test4_in in result)  # Every outcome is its own ancestor

    def test_5(self):
        """Test passing in a variable intervening on itself.

        The problem reduces to just getting ancestors. Source: out of RJC's head.
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
        is guaranteed to have probability 0. Source: RJC's mind.
        """
        event = [(Y @ -X, -Y), (Y @ -X, +Y)]
        self.assertEquals(simplify(event), Zero())

    def test_inconsistent_2(self):
        """Second test for simplifying an inconsistent event. Source: RJC's mind."""
        event = [(Y @ -Y, +Y)]
        self.assertEquals(simplify(event), Zero())

    def test_redundant_1(self):
        """First test for simplifying an event with redundant subscripts. Source: RJC's mind."""
        event = [(Y @ -X, -Y), (Y @ -X, -Y)]
        result = simplify(event)
        self.assertEqual(result, [(Y @ -X, -Y)])

    def test_redundant_2(self):
        """Second test for simplifying an event with redundant subscripts. Source: RJC's mind."""
        event = [(Y @ -X, -Y), (Y @ -X, -Y), (X @ -W, -X)]
        self.assertEqual(simplify(event), [(Y @ -X, -Y), (X @ -W, -X)])

    def test_redundant_3(self):
        """Third test for simplifying an event with redundant subscripts.

        (Y @ (-Y,-X), -Y) reduces to (Y @ -Y, -Y) via line 1 of the SIMPLIFY algorithm.
        Source: JZ's mind.
        """
        event = [
            (Y @ (-Y, -X), -Y),
            (Y @ -Y, -Y),
            (X @ -W, -X),
        ]
        self.assertEqual(simplify(event), [(Y @ -Y, -Y), (X @ -W, -X)])

    def test_redundant_4(self):
        """Fourth test for simplifying an event with redundant subscripts. Source: RJC's mind."""
        event = [
            (Y @ -Y, -Y),
            (Y @ -Y, -Y),
        ]
        self.assertIsNone(simplify(event))


class TestIsCounterfactualFactorForm(unittest.TestCase):
    """Test whether a set of counterfactual variables are all in counterfactual factor form."""

    # TODO: Incorporate a test involving counterfactual unnesting.

    def test_is_counterfactual_factor_form(self):
        """From Example 3.3 of [correa22a]_."""
        event1 = {(-Y @ (-Z, -W, -X)), (-W @ -X)}  # (Y @ -Z @ -W @ -X)
        self.assertTrue(is_counterfactual_factor_form(event=event1, graph=figure_2a_graph))

        event2 = {(-W @ -X), (-Z)}
        self.assertTrue(is_counterfactual_factor_form(event=event2, graph=figure_2a_graph))

        event3 = {(-Y @ (-Z, -W)), (-W @ -X)}
        self.assertFalse(is_counterfactual_factor_form(event=event3, graph=figure_2a_graph))

        # Y has parents, so they should be intervened on but are not
        event4 = {(-Y)}
        self.assertFalse(is_counterfactual_factor_form(event=event4, graph=figure_2a_graph))

        # Z has no parents, so this variable is also a ctf-factor
        event5 = {(-Z)}
        self.assertTrue(is_counterfactual_factor_form(event=event5, graph=figure_2a_graph))

        # Z is not a parent of W, so the second counterfactual variable is not a ctf-factor,
        # because it is not a valid counterfactual variable
        event6 = {(-Y @ (-Z, -W)), (-W @ (-X, Z))}
        self.assertFalse(is_counterfactual_factor_form(event=event6, graph=figure_2a_graph))


class TestMakeSelectionDiagram(unittest.TestCase):
    """Test the results of creating a selection diagram that is an amalgamation of domain selection diagrams."""

    def test_make_selection_diagram(self):
        """Create Figure 2(b) of [correa22a]_ from Figures 3(a) and 3(b)."""
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

    Source: last paragraph in Section 4 of [correa22a]_, before Section 4.1.
    Mathematical expression: ||\mathbf Y_*|| = {||Y_{\mathbf x}|| | Y_{\mathbf x}} \elementof \mathbf Y_*}, and
    ||Y_{\mathbf x}|| = Y_{\mathbf t}, where \mathbf T = \mathbf X \intersect An(Y)_{G_{\overline{\mathbf X}}}}.
    (The math syntax is not necessarily cannonical LaTeX.)
    """

    def test_minimize_1(self):
        """Test the minimize function sending in a single counterfactual variable. Source: out of RJC's head."""
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
        """Test the minimize function for multiple counterfactual variables. Source: out of RJC's head."""
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

        Source: out of RJC's head.
        """
        same_district_test1_in = {(Y @ (-X, -W, -Z)), (W @ (-X))}
        self.assertTrue(same_district(same_district_test1_in, figure_2a_graph))

    def test_same_district_2(self):
        """Test #2 for whether a set of counterfactual variables are in the same district (c-component).

        Source: out of RJC's head.
        """
        same_district_test2_in = {(Z), (X @ -Z)}
        self.assertTrue(same_district(same_district_test2_in, figure_2a_graph))

    def test_same_district_3(self):
        """Test #3 for whether a set of counterfactual variables are in the same district (c-component).

        Source: out of RJC's head.
        """
        same_district_test3_in = {(Y @ -Y), (Z), (X @ -Z)}
        self.assertFalse(same_district(same_district_test3_in, figure_2a_graph))

    def test_same_district_4(self):
        """Test #4 for whether a set of counterfactual variables are in the same district (c-component).

        Source: out of RJC's head.
        """
        same_district_test4_in = {(Y @ -Y)}
        self.assertTrue(same_district(same_district_test4_in, figure_2a_graph))

    def test_same_district_5(self):
        """Test #5 for whether a set of counterfactual variables are in the same district (c-component).

        Source: out of RJC's head. This test is meant to raise a key error.
        """
        same_district_test5_in = {(R @ -Y)}
        self.assertRaises(KeyError, same_district, same_district_test5_in, figure_2a_graph)

    def test_same_district_6(self):
        """Test #6 for whether a set of counterfactual variables are in the same district (c-component).

        Source: out of RJC's head. Edge case where we pass in no variables.
        """
        same_district_test6_in = set()
        self.assertTrue(KeyError, same_district(same_district_test6_in, figure_2a_graph))


class TestGetCounterfactualFactors(unittest.TestCase):
    """Test the GetCounterfactualFactors function in counterfactual_transportability.py.

    This is one step in the ctf-factor factorization process. Here we want
    to check that we can separate a joint probability distribution of ctf-factors
    into a set of joint probability distributions that we'll later multiply
    together as per Equation 15 in [correa22a]_.
    """

    def assert_collection_of_set_equal(
        self, left: Collection[set[Variable]], right: Collection[set[Variable]]
    ) -> None:
        """Check that two collections contain sets with the same elements."""
        c1 = Counter(frozenset(element) for element in left)
        c2 = Counter(frozenset(el) for el in right)
        self.assertEqual(c1, c2)

    def test_get_counterfactual_factors_1(self):
        """Test factoring a set of counterfactual variables by district (c-component).

        Source: Example 4.2 of [correa22a]_. Note that
                we're not testing the full example, just computing factors
                for the query.
        """
        get_counterfactual_factors_test_1_in = {
            (Y @ (-X, -W, -Z)),
            (W @ -X),
            (X @ -Z),
            (Z @ -Z),
            (-Z),
        }
        get_counterfactual_factors_test_1_expected = [
            {(Y @ (-X, -W, -Z)), (W @ -X)},
            {(X @ -Z), (-Z), (Z @ -Z)},
        ]
        self.assert_collection_of_set_equal(
            get_counterfactual_factors_test_1_expected,
            get_counterfactual_factors(
                event=get_counterfactual_factors_test_1_in, graph=figure_2a_graph
            ),
        )

    def test_get_counterfactual_factors_2(self):
        """Test factoring a set of counterfactual variables by district (c-component).

        Source: RJC. We send in a variable not in the graph. Throws a NetworkXError
           when we try to get that node's parents.
        """
        get_counterfactual_factors_test_2_in = {
            (Y @ (-X, -W, -Z)),
            (W @ -X),
            (X @ -Z),
            (Z @ -Z),
            (-Z),
            (R @ -Y),
        }
        self.assertRaises(
            NetworkXError,
            get_counterfactual_factors,
            event=get_counterfactual_factors_test_2_in,
            graph=figure_2a_graph,
        )

    def test_get_counterfactual_factors_3(self):
        """Test factoring a set of counterfactual variables by district (c-component).

        Source: RJC. We send in a variable not in counterfactual factor form.
        """
        get_counterfactual_factors_test_3_in = {
            (Y @ (-X, -W)),
            (W @ -X),
            (X @ -Z),
            (Z @ -Z),
            (-Z),
        }
        self.assertRaises(
            KeyError,
            get_counterfactual_factors,
            event=get_counterfactual_factors_test_3_in,
            graph=figure_2a_graph,
        )


class TestDoCounterfactualFactorFactorization(cases.GraphTestCase):
    """Test factorizing the counterfactual factors corresponding to a query, as per Example 4.2 of [correa22a]_.

    This puts together getting the ancestral set of a query, getting the ctf-factors for each element of the set,
    and factorizing the resulting joint distribution according to the C-components of the graph.
    """

    # TODO: Add more tests, looking at edge cases: empty set, One(),
    # maybe check that a graph with distinct ancestral components just treats
    # everything as having one ancestral set here, etc.
    def test_do_counterfactual_factor_factorization_1(self):
        """Test counterfactual factor factorization as per Equation 16 in [correa22a]_.

        Source: Equations 11, 14, and 16 of [correa22a]_.
        """
        # First test is already in counterfactual factor form
        equation_16_test_1_in = {(Y @ (-X, -W, -Z)), (W @ -X), (X @ -Z), (Z)}
        # Question for Jeremy: should the second term be -X instead of just X?
        # Is the multiplication syntax correct?
        # equation_16_test_1_expected = Sum.safe(
        #    P([(Y @ (-X, -W, -Z)), (W @ -X)]) * P([(X @ -Z), (Z)]), [Z, W]
        # )
        equation_16_test_1_expected = P([(Y @ (-X, -W, -Z)), (W @ -X)]) * P([(X @ -Z), (Z)])
        self.assert_expr_equal(
            do_counterfactual_factor_factorization(
                event=equation_16_test_1_in, graph=figure_2a_graph
            ),
            equation_16_test_1_expected,
        )

    def test_do_counterfactual_factor_factorization_2(self):
        """Test counterfactual factor factorization as per Equation 16 in [correa22a]_.

        Source: Equations 11, 14, and 16 of [correa22a]_.
        """
        # Second test is not in counterfactual factor form
        equation_16_test_2_in = {(Y @ (-X)), (W @ -X), (X @ -Z), (Z)}
        equation_16_test_2_expected = P((Y @ (-X, -W, -Z)), (W @ -X)) * P((X @ -Z), (Z))
        self.assert_expr_equal(
            do_counterfactual_factor_factorization(
                event=equation_16_test_2_in, graph=figure_2a_graph
            ),
            equation_16_test_2_expected,
        )

    def test_do_counterfactual_factor_factorization_3(self):
        """Test counterfactual factor factorization as per Equation 16 in [correa22a]_.

        Source: Equations 11, 14, and 16 of [correa22a]_.
        """
        # This is the actual equation 16 content in [correa22a]_
        equation_16_test_3_in = {(Y @ -X), (X)}
        equation_16_test_3_expected = Sum.safe(
            P((Y @ (-X, -W, -Z)), (W @ -X)) * P((X @ -Z), (Z)), [Z, W]
        )
        self.assert_expr_equal(
            do_counterfactual_factor_factorization(
                event=equation_16_test_3_in, graph=figure_2a_graph
            ),
            equation_16_test_3_expected,
        )


class TestConvertToCounterfactualFactorForm(unittest.TestCase):
    """Test converting a set of variables to counterfactual factor form.

    Source: Example 4.2 of [correa22a]_.
    """

    def test_convert_to_counterfactual_factor_form_1(self):
        """Test conversion of a set of variables to counterfactual factor form.

        Source: Equation 12 of [correa22a]_.
        """
        test_1_in = {(Y @ -X), (W)}
        test_1_expected = {(Y @ (-X, -W, -Z)), (W @ -X)}
        self.assertSetEqual(
            convert_to_counterfactual_factor_form(event=test_1_in, graph=figure_2a_graph),
            test_1_expected,
        )

    def test_convert_to_counterfactual_factor_form_2(self):
        """Test conversion of a set of variables to counterfactual factor form.

        Source: Equation 12 of [correa22a]_.

        Here we pass in a simple variable with no parents, and should get it back.
        """
        test_2_in = {(Z)}
        test_2_expected = {(Z)}
        self.assertSetEqual(
            convert_to_counterfactual_factor_form(event=test_2_in, graph=figure_2a_graph),
            test_2_expected,
        )


class TestCounterfactualFactorTransportability(unittest.TestCase):
    """Test whether a counterfactual factor can be transported from a domain to the target domain.

    Source: Lemma 3.1 and Example 3.4 of [correa22a]_.
    """

    def test_counterfactual_factor_transportability_1(self):
        """Test equation 9 of [correa22a]_."""
        test_1_in = {(-Y @ -X @ -Z @ -W), (-W @ -X)}
        self.assertTrue(
            counterfactual_factors_are_transportable(
                factors=test_1_in, domain_graph=figure_2_graph_domain_1
            )
        )
        self.assertFalse(
            counterfactual_factors_are_transportable(
                factors=test_1_in, domain_graph=figure_2_graph_domain_2
            )
        )

    def test_counterfactual_factor_transportability_2(self):
        """Test equation 10 of [correa22a]_."""
        test_2_in = {(-Y @ -X @ -Z @ -W)}
        self.assertTrue(
            counterfactual_factors_are_transportable(
                factors=test_2_in, domain_graph=figure_2_graph_domain_1
            )
        )
        self.assertTrue(
            counterfactual_factors_are_transportable(
                factors=test_2_in, domain_graph=figure_2_graph_domain_2
            )
        )

    def test_counterfactual_factor_transportability_3(self):
        """Test Example 3.4 of [correa22a]_."""
        test_3_in = {(-W @ -X), (-Z)}
        self.assertFalse(
            counterfactual_factors_are_transportable(
                factors=test_3_in, domain_graph=figure_2_graph_domain_1
            )
        )
        self.assertFalse(
            counterfactual_factors_are_transportable(
                factors=test_3_in, domain_graph=figure_2_graph_domain_2
            )
        )

    def test_counterfactual_factor_transportability_4(self):
        """Another test related to Example 3.4 of [correa22a]_."""
        test_4_in = {(-X @ -Z), (-Z)}
        self.assertFalse(
            counterfactual_factors_are_transportable(
                factors=test_4_in, domain_graph=figure_2_graph_domain_1
            )
        )
        self.assertTrue(
            counterfactual_factors_are_transportable(
                factors=test_4_in, domain_graph=figure_2_graph_domain_2
            )
        )

    def test_counterfactual_factor_transportability_5(self):
        """Another test related to Example 3.4 of [correa22a]_.

        Checking that a child of a transported node in the same c-component is still tranportable.
        """
        test_5_in = {(-X @ -Z)}
        self.assertTrue(
            counterfactual_factors_are_transportable(
                factors=test_5_in, domain_graph=figure_2_graph_domain_1
            )
        )
        self.assertTrue(
            counterfactual_factors_are_transportable(
                factors=test_5_in, domain_graph=figure_2_graph_domain_2
            )
        )
