# -*- coding: utf-8 -*-

"""Tests for counterfactual transportability.

.. [huang08a] https://link.springer.com/article/10.1007/s10472-008-9101-x.
.. [correa20a] https://proceedings.neurips.cc/paper/2020/file/7b497aa1b2a83ec63d1777a88676b0c2-Paper.pdf.
.. [correa22a] https://proceedings.mlr.press/v162/correa22a/correa22a.pdf.
.. [santikka20a] https://github.com/santikka/causaleffect/blob/master/R/compute.c.factor.R.
.. [santikka20b] https://github.com/santikka/causaleffect/blob/master/R/identify.R.
.. [tian03a] https://ftp.cs.ucla.edu/pub/stat_ser/R290-L.pdf.
"""

import logging
import unittest
from collections import defaultdict

from networkx import NetworkXError

from tests.test_algorithm import cases
from y0.algorithm.counterfactual_transportability import (
    _any_variables_with_inconsistent_values,
    _reduce_reflexive_counterfactual_variables_to_interventions,
    _remove_repeated_variables_and_values,
    _split_event_by_reflexivity,
    convert_to_counterfactual_factor_form,
    counterfactual_factors_are_transportable,
    do_counterfactual_factor_factorization,
    get_ancestors_of_counterfactual,
    get_counterfactual_factors,
    is_counterfactual_factor_form,
    make_selection_diagram,
    minimize,
    minimize_event,
    same_district,
    simplify,
)
from y0.algorithm.tian_id import (
    _compute_c_factor,
    _tian_equation_69,
    _tian_equation_72,
    _tian_lemma_1_i,
    _tian_lemma_4_ii,
    tian_pearl_identify,
)
from y0.algorithm.transport import transport_variable
from y0.dsl import (  # TARGET_DOMAIN,; Pi1,
    PP,
    W1,
    W2,
    W3,
    W4,
    W5,
    X1,
    X2,
    CounterfactualVariable,
    Fraction,
    Intervention,
    One,
    P,
    Population,
    Product,
    R,
    Sum,
    Variable,
    W,
    X,
    Y,
    Z,
)
from y0.graph import NxMixedGraph

# from y0.tests.test_algorithm.cases import GraphTestCase

# From [correa22a]_, Figure 1.
figure_1_graph = NxMixedGraph.from_edges(
    directed=[
        (X, Z),
        (Z, Y),
        (X, Y),
        (transport_variable(Y), Y),
    ],
    undirected=[(Z, X)],
)

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

# From [correa20a]_, Figure 1a.
soft_interventions_figure_1a_graph = NxMixedGraph.from_edges(
    directed=[(X1, Z), (X1, X2), (Z, X2), (X1, Y), (X2, Y)],
    undirected=[(X1, Z), (X2, Y)],
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

# From [correa20a]_, Figure 1c.
soft_interventions_figure_1c_graph = NxMixedGraph.from_edges(
    directed=[
        (X1, Z),
        (X1, X2),
        (X1, Y),
        (Z, X2),
        (X2, Y),
        (transport_variable(Z), Z),
    ],
    undirected=[(X1, Z)],
)

# From [correa20a]_, Figure 1d.
soft_interventions_figure_1d_graph = NxMixedGraph.from_edges(
    directed=[
        (X1, Z),
        (X1, X2),
        (X1, Y),
        (Z, X2),
        (X2, Y),
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

# From [correa20a]_, Figure 2b.
soft_interventions_figure_2b_graph = NxMixedGraph.from_edges(
    directed=[
        (R, Z),
        (R, X),
        (X, Z),
        (Z, Y),
    ],
    undirected=[
        (R, Y),
        (W, R),
        (W, Z),
    ],
)

# From [correa20a]_, Figure 2c.
soft_interventions_figure_2c_graph = NxMixedGraph.from_edges(
    directed=[
        (R, Z),
        (X, Z),
        (W, X),
        (Z, Y),
        (transport_variable(R), R),
        (transport_variable(W), W),
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

# From [correa20a]_, Figure 2e.
soft_interventions_figure_2e_graph = NxMixedGraph.from_edges(
    directed=[
        (R, Z),
        (X, Z),
        (W, X),
        (Z, Y),
        (transport_variable(R), R),
    ],
    undirected=[
        (R, Y),
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
            name="Y", star=None, interventions=frozenset({Intervention(name="X", star=False)})
        )  # Y @ -X
        # test1_in = (Y @ -X)
        test1_out = {
            CounterfactualVariable(
                name="Y", star=None, interventions=frozenset({Intervention(name="X", star=False)})
            ),
            CounterfactualVariable(
                name="W", star=None, interventions=frozenset({Intervention(name="X", star=False)})
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
            interventions=frozenset(
                {Intervention(name="Y", star=False), Intervention(name="Z", star=False)}
            ),
        )  # W @ -Y @ -Z
        test2_out = {
            CounterfactualVariable(
                name="W", star=None, interventions=frozenset({Intervention(name="Z", star=False)})
            ),
            CounterfactualVariable(
                name="X", star=None, interventions=frozenset({Intervention(name="Z", star=False)})
            ),
        }
        result = get_ancestors_of_counterfactual(event=test2_in, graph=figure_2a_graph)
        self.assertTrue(variable in test2_out for variable in result)
        self.assertTrue((W @ -Z) in result)  # The outcome, (W @ -Y @ -Z), simplifies to this
        self.assertFalse(test2_in in result)

    def test_example_2_1_3(self):
        """Test the third result of Example 2.1."""
        test3_in = CounterfactualVariable(
            name="Y", star=None, interventions=frozenset({Intervention(name="W", star=False)})
        )  # Y @ -W
        test3_out = {
            CounterfactualVariable(
                name="Y", star=None, interventions=frozenset({Intervention(name="W", star=False)})
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
            name="Y", star=None, interventions=frozenset({Intervention(name="Y", star=False)})
        )
        test5_out = {
            CounterfactualVariable(
                name="Y", star=None, interventions=frozenset({Intervention(name="Y", star=False)})
            )
        }
        result = get_ancestors_of_counterfactual(event=test5_in, graph=figure_2a_graph)
        self.assertTrue(variable in test5_out for variable in result)
        self.assertTrue(test5_in in result)  # Every outcome is its own ancestor


class TestSimplify(cases.GraphTestCase):
    """Test the simplify algorithm from counterfactual transportability.

    We also test the subroutines that only simplify calls:
    1. _bifurcate_event_by_reflexivity
    2. _reduce_reflexive_counterfactual_variables_to_interventions
    3. _remove_repeated_values
    """

    # TODO: 1. Incorporate a test involving counterfactual unnesting.
    #       2. Take the line 2 and line 3 tests and run them through the
    #           simplify() function in addition to the separate functions
    #           for lines 2 and 3.

    def test_inconsistent_1(self):
        """Test simplifying an inconsistent event.

        Correa et al. specify the output should be 0 if the counterfactual event
        is guaranteed to have probability 0. Source: RJC's mind.
        """
        event = [(Y @ -X, -Y), (Y @ -X, +Y)]
        result = simplify(event=event, graph=figure_2a_graph)
        logger.warning("Result for test_inconsistent_1 is " + str(result))
        self.assertIsNone(simplify(event=event, graph=figure_2a_graph))

    def test_inconsistent_2(self):
        """Second test for simplifying an inconsistent event. Source: RJC's mind."""
        event = [(Y @ -Y, +Y)]
        self.assertIsNone(simplify(event=event, graph=figure_2a_graph))

    def test_split_event_by_reflexivity(self):
        """Test splitting event variables that intervene on themselves from those that do not.

        Source: RJC's mind.
        """
        # TODO: test queries like (Y @ -Y @ -X, -Y), a low priority because we
        # call this function after calling minimize_event().
        reflexive_event_1, nonreflexive_event_1 = _split_event_by_reflexivity(
            event=[(Y @ -X, -Y), (Y @ -X, +Y), (Y @ -Y, -Y)]
        )
        self.assertCountEqual(reflexive_event_1, [(Y @ -Y, -Y)])
        self.assertCountEqual(nonreflexive_event_1, [(Y @ -X, -Y), (Y @ -X, +Y)])
        reflexive_event_2, nonreflexive_event_2 = _split_event_by_reflexivity(
            event=[(Y @ -X, -Y), (Y @ -X, +Y)]
        )
        self.assertCountEqual(reflexive_event_2, [])
        self.assertCountEqual(nonreflexive_event_2, [(Y @ -X, -Y), (Y @ -X, +Y)])
        reflexive_event_3, nonreflexive_event_3 = _split_event_by_reflexivity(event=[(Y @ -Y, -Y)])
        self.assertCountEqual(reflexive_event_3, [(Y @ -Y, -Y)])
        self.assertCountEqual(nonreflexive_event_3, [])

    def test_reduce_reflexive_counterfactual_variables_to_interventions(self):
        """Test reducing counterfactual variables that intervene on themselves to simpler Intervention objects.

        Source: RJC's mind.
        """
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ -Y].add(+Y)
        reflexive_variable_to_value_mappings[Y].add(-Y)
        logger.warning(
            "test_reduce_reflexive_counterfactual_variables_to_interventions: input dict = "
            + str(reflexive_variable_to_value_mappings)
        )
        result_dict = _reduce_reflexive_counterfactual_variables_to_interventions(
            reflexive_variable_to_value_mappings
        )
        assert Y in result_dict
        self.assertSetEqual(result_dict[Y], {+Y, -Y})

    def test_remove_repeated_variables_and_values(self):
        """Test removing repeated occurrences of variables and associated values in an event.

        Source: RJC's mind.
        """
        event_1 = [(Y @ -Y, -Y), (Y @ -Y, -Y)]
        event_2 = [(X, -X), (X, -X)]
        result_1 = _remove_repeated_variables_and_values(event=event_1)
        result_2 = _remove_repeated_variables_and_values(event=event_2)
        # @cthoyt: Just curious: is there a one-liner to assert that two dictionaries are equal?
        self.assertCountEqual([Y @ -Y], result_1.keys())
        self.assertEqual(len(result_1[Y @ -Y]), 1)
        self.assertSetEqual(result_1[Y @ -Y], {-Y})
        self.assertCountEqual([X], result_2.keys())
        self.assertEqual(len(result_2[X]), 1)
        self.assertSetEqual(result_2[X], {-X})

    def test_line_2_1(self):
        """Directly test the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls."""
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ -Y].add(-Y)

        nonreflexive_variable_to_value_mappings = defaultdict(set)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(-Y)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(+Y)

        logger.warning(
            "In test_line_2_1: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.warning(
            "In test_line_2_1: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        self.assertTrue(
            _any_variables_with_inconsistent_values(
                nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
                reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
            )
        )

    def test_line_2_2(self):
        """Second test for the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls."""
        reflexive_variable_to_value_mappings = defaultdict(set)

        nonreflexive_variable_to_value_mappings = defaultdict(set)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(-Y)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(-Y)
        logger.warning(
            "In test_line_2_2: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.warning(
            "In test_line_2_2: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        self.assertFalse(
            _any_variables_with_inconsistent_values(
                nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
                reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
            )
        )

    def test_line_2_3(self):
        """Third test for the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls."""
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ -Y].add(+Y)

        nonreflexive_variable_to_value_mappings = defaultdict(set)
        logger.warning(
            "In test_line_2_3: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.warning(
            "In test_line_2_3: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        self.assertTrue(
            _any_variables_with_inconsistent_values(
                nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
                reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
            )
        )

    def test_line_2_4(self):
        """Fourth test for the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls."""
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ -Y].add(-Y)

        nonreflexive_variable_to_value_mappings = defaultdict(set)

        logger.warning(
            "In test_line_2_4: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.warning(
            "In test_line_2_4: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        self.assertFalse(
            _any_variables_with_inconsistent_values(
                nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
                reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
            )
        )

    def test_line_2_5(self):
        """Fifth test for the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls."""
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ +Y].add(-Y)

        nonreflexive_variable_to_value_mappings = defaultdict(set)
        logger.warning(
            "In test_line_2_5: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.warning(
            "In test_line_2_5: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        self.assertTrue(
            _any_variables_with_inconsistent_values(
                nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
                reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
            )
        )

    def test_line_2_6(self):
        """Sixth test for the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls."""
        reflexive_variable_to_value_mappings = defaultdict(set)

        nonreflexive_variable_to_value_mappings = defaultdict(set)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(-Y)
        nonreflexive_variable_to_value_mappings[Y @ -Z].add(+Y)
        logger.warning(
            "In test_line_2_6: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.warning(
            "In test_line_2_6: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        self.assertFalse(
            _any_variables_with_inconsistent_values(
                nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
                reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
            )
        )

    def test_line_2_7(self):
        """Seventh test for the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls."""
        event = [(Y @ -Y, -Y), (Y, +Y)]
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ -Y].add(-Y)
        reflexive_variable_to_value_mappings[Y].add(+Y)

        nonreflexive_variable_to_value_mappings = defaultdict(set)

        logger.warning(
            "In test_line_2_7: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.warning(
            "In test_line_2_7: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        # Should be false because by calling _any_variables_with_inconsistent_values
        # directly, we never reduce Y@-Y to an intervention.
        self.assertFalse(
            _any_variables_with_inconsistent_values(
                nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
                reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
            )
        )
        self.assertIsNone(simplify(event=event, graph=figure_2a_graph))

    def test_redundant_1(self):
        """First test for simplifying an event with redundant subscripts. Source: RJC's mind."""
        event = [(Y @ -X, -Y), (Y @ -X, -Y)]
        result = simplify(event=event, graph=figure_2a_graph)
        self.assertEqual(result, [(Y @ -X, -Y)])

    def test_redundant_2(self):
        """Second test for simplifying an event with redundant subscripts. Source: RJC's mind."""
        event = [(Y @ -X, -Y), (Y @ -X, -Y), (X @ -Z, -X)]
        # Kind of misleading assertion name, but checks that two lists are the same regardless of order
        test_result = simplify(event=event, graph=figure_2a_graph)
        self.assertCountEqual(test_result, [(Y @ -X, -Y), (X @ -Z, -X)])
        self.assertCountEqual(test_result, [(X @ -Z, -X), (Y @ -X, -Y)])

    def test_redundant_3(self):
        """Third test for simplifying an event with redundant subscripts.

        (Y @ (-Y,-X), -Y) reduces to (Y @ -Y, -Y) via line 1 of the SIMPLIFY algorithm.
        And then we want to further simplify (Y @ -Y, -Y) to
        Source: JZ's mind.
        """
        event = [
            (Y @ (-Y, -X), -Y),
            (Y @ -Y, -Y),
            (X @ -Z, -X),
        ]
        self.assertCountEqual(simplify(event=event, graph=figure_2a_graph), [(Y, -Y), (X @ -Z, -X)])

    def test_redundant_4(self):
        """Test that Y@-Y and -Y are treated as redundant and properly minimized to -Y. Source: out of RJC's mind."""
        event1 = [(Y @ -Y, -Y), (Y, -Y)]
        event2 = [(Y, -Y), (Y @ -Y, -Y), (Y, -Y)]
        self.assertEqual(simplify(event=event1, graph=figure_2a_graph), [(Y, -Y)])
        self.assertEqual(simplify(event=event2, graph=figure_2a_graph), [(Y, -Y)])

    def test_misspecified_input(self):
        """Make sure users don't pass improper input into SIMPLIFY.

        That includes variables with star values that are not counterfactual variables,
        as well as events with tuples that are the wrong length or elements that
        are not pairs of variables and interventions.

        Source: JZ and RJC
        """
        event_1 = [
            (Y @ -Y, -Y),
            (
                Variable(
                    name="Y",
                    star=False,
                ),
                -Y,
            ),
        ]
        event_2 = [
            (Y @ -Y, -Y),
            (
                Variable(
                    name="Y",
                    star=True,
                ),
                -Y,
            ),
        ]
        event_3 = [
            (Y @ -Y, -Y, -Y),
            (
                Variable(
                    name="Y",
                    star=None,
                ),
                -Y,
            ),
        ]
        self.assertRaises(TypeError, simplify, event=event_1, graph=figure_2a_graph)
        self.assertRaises(TypeError, simplify, event=event_2, graph=figure_2a_graph)
        self.assertRaises(TypeError, simplify, event=event_3, graph=figure_2a_graph)

    def test_simplified_1(self):
        """Fourth test for simplifying an event with redundant subscripts. Source: RJC's mind."""
        event = [
            (Y @ -X, -Y),
            (Y @ -Z, -Y),
        ]
        self.assertCountEqual(
            simplify(event=event, graph=figure_2a_graph), [(Y @ -Z, -Y), (Y @ -X, -Y)]
        )

    def test_simplify_y(self):
        """Comprehensive test involving combinations of outcome variables with redundant and conflicting subscripts.

        Source: RJC's mind.
        """
        event_1 = [(Y @ -Y, -Y)]
        event_2 = [(Y @ -Y, +Y)]
        event_3 = [(Y @ +Y, -Y)]
        event_4 = [(Y @ +Y, +Y)]
        event_5 = [(Y @ -Y, -Y), (Y, -Y)]
        event_6 = [(Y @ -Y, -Y), (Y, +Y)]
        event_7 = [(Y @ -Y, +Y), (Y, -Y)]
        event_8 = [(Y @ -Y, +Y), (Y, +Y)]
        event_9 = [(Y @ +Y, -Y), (Y, -Y)]
        event_10 = [(Y @ +Y, -Y), (Y, +Y)]
        event_11 = [(Y @ +Y, +Y), (Y, -Y)]
        event_12 = [(Y @ +Y, +Y), (Y, +Y)]
        event_13 = [(Y, -Y)]
        event_14 = [(Y, +Y)]
        # event_15 = [(Y, -Y),(Y, +Y)]
        # The reason event_1 can safely simplify to an intervention is that An(Y_y) == An(Y) (Algorithm 2, Line 2).
        self.assertCountEqual(simplify(event=event_1, graph=figure_2a_graph), [(Y, -Y)])
        self.assertIsNone(simplify(event=event_2, graph=figure_2a_graph))
        self.assertIsNone(simplify(event=event_3, graph=figure_2a_graph))
        self.assertCountEqual(simplify(event=event_4, graph=figure_2a_graph), [(Y, +Y)])
        self.assertCountEqual(simplify(event=event_5, graph=figure_2a_graph), [(Y, -Y)])
        self.assertIsNone(simplify(event=event_6, graph=figure_2a_graph))
        self.assertIsNone(simplify(event=event_7, graph=figure_2a_graph))
        self.assertIsNone(simplify(event=event_8, graph=figure_2a_graph))
        self.assertIsNone(simplify(event=event_9, graph=figure_2a_graph))
        self.assertIsNone(simplify(event=event_10, graph=figure_2a_graph))
        self.assertIsNone(simplify(event=event_11, graph=figure_2a_graph))
        self.assertCountEqual(simplify(event=event_12, graph=figure_2a_graph), [(Y, +Y)])
        self.assertCountEqual(simplify(event=event_13, graph=figure_2a_graph), [(Y, -Y)])
        self.assertCountEqual(simplify(event=event_14, graph=figure_2a_graph), [(Y, +Y)])


class TestIsCounterfactualFactorForm(unittest.TestCase):
    """Test whether a set of counterfactual variables are all in counterfactual factor form."""

    # TODO: Incorporate a test involving counterfactual unnesting.

    def test_is_counterfactual_factor_form(self):
        """From Example 3.3 of [correa22a]_."""
        event1 = {(Y @ (-Z, -W, -X)), (W @ -X)}  # (Y @ -Z @ -W @ -X)
        self.assertTrue(is_counterfactual_factor_form(event=event1, graph=figure_2a_graph))

        event2 = {(W @ -X), (-Z)}
        self.assertTrue(is_counterfactual_factor_form(event=event2, graph=figure_2a_graph))

        event3 = {(Y @ (-Z, -W)), (W @ -X)}
        self.assertFalse(is_counterfactual_factor_form(event=event3, graph=figure_2a_graph))

        # Y has parents, so they should be intervened on but are not
        event4 = {(Y)}
        self.assertFalse(is_counterfactual_factor_form(event=event4, graph=figure_2a_graph))

        # Z has no parents, so this variable is also a ctf-factor
        event5 = {(Z)}
        self.assertTrue(is_counterfactual_factor_form(event=event5, graph=figure_2a_graph))

        # Z is not a parent of W, so the second counterfactual variable is not a ctf-factor,
        # because it is not a valid counterfactual variable
        event6 = {(Y @ (-Z, -W)), (W @ (-X, Z))}
        self.assertFalse(is_counterfactual_factor_form(event=event6, graph=figure_2a_graph))

        # Check that P(Y_y) is not in counterfactual factor form
        event7 = {(Y @ (-Z, -W, -X, -Y)), (W @ -X)}  # (Y @ -Z @ -W @ -X)
        self.assertFalse(is_counterfactual_factor_form(event=event7, graph=figure_2a_graph))


class TestMakeSelectionDiagram(unittest.TestCase):
    """Test the results of creating a selection diagram that is an amalgamation of domain selection diagrams."""

    def test_make_selection_diagram(self):
        """Create Figure 2(b) of [correa22a]_ from Figures 3(a) and 3(b)."""
        selection_nodes = {1: {Z}, 2: {W}}
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
                ),
                # How do we indicate with a superscript that this is from domain 1?
                # cthoyt:The domain-variable association explicitly does not live in the graph.
                (
                    transport_variable(W),
                    W,
                ),  # How do we indicate with a superscript that this is from domain 2?
            ],
            undirected=[(Z, X), (W, Y)],
        )
        self.assertEquals(selection_diagram, expected_selection_diagram)


class TestMinimizeEvent(cases.GraphTestCase):
    r"""Test minimizing a set of counterfactual variables.

    Source: last paragraph in Section 4 of [correa22a]_, before Section 4.1.
    Mathematical expression: ||\mathbf Y_*|| = {||Y_{\mathbf x}|| | Y_{\mathbf x}} \elementof \mathbf Y_*}, and
    ||Y_{\mathbf x}|| = Y_{\mathbf t}, where \mathbf T = \mathbf X \intersect An(Y)_{G_{\overline{\mathbf X}}}}.
    (The math syntax is not necessarily cannonical LaTeX.)
    """

    minimize_graph_1 = NxMixedGraph.from_edges(
        directed=[
            (X, W),
            (W, Y),
        ],
        undirected=[(X, Y)],
    )

    minimize_graph_2 = NxMixedGraph.from_edges(
        directed=[
            (Z, X),
            (X, W),
            (W, Y),
        ],
        undirected=[(X, Y)],
    )

    def test_minimize_event_1(self):
        """Test the minimize_event function sending in a single counterfactual variable.

        Source: out of RJC's head.
        """
        minimize_event_test1_in = [(Y @ -W @ -X, -Y)]
        minimize_event_test1_out = [(Y @ -W, -Y)]
        self.assertCountEqual(
            minimize_event_test1_out,
            minimize_event(event=minimize_event_test1_in, graph=self.minimize_graph_1),
        )

    def test_minimize_event_2(self):
        """Test the minimize_event function for multiple counterfactual variables.

        Source: out of RJC's head.
        """
        minimize_event_test2_in = [(Y @ (-W, -X, -Z), -Y), (W @ (-X, -Z), -W)]
        minimize_event_test2_out = [(Y @ -W, -Y), (W @ -X, -W)]
        self.assertCountEqual(
            minimize_event_test2_out,
            frozenset(minimize_event(event=minimize_event_test2_in, graph=self.minimize_graph_2)),
        )

    def test_minimize_event_3(self):
        """Test the minimize_event function for multiple counterfactual variables and a different graph.

        Source: out of RJC's head.
        """
        minimize_event_test3_in = [(Y @ (-W, -X), -Y)]
        minimize_event_test3_out = [(Y @ (-X, -W), -Y)]
        # self.assertSetEqual(minimize_test3_out, minimize(variables = minimize_test3_in, graph = figure_2a_graph))
        # The intentional reverse order of the interventions means we have to use P() to sort it out (no pun intended).
        self.assertCountEqual(
            minimize_event_test3_out,
            minimize_event(event=minimize_event_test3_in, graph=figure_2a_graph),
        )

    def test_minimize_event_4(self):
        """Test the minimize_event function sending in a single variable with no interventions.

        Source: out of RJC's head.
        """
        minimize_event_test4_in = [(Y, -Y)]
        minimize_event_test4_out = [(Y, -Y)]
        self.assertCountEqual(
            minimize_event_test4_out,
            minimize_event(event=minimize_event_test4_in, graph=self.minimize_graph_1),
        )

    def test_minimize_event_5(self):
        """Test the minimize_event function sending in a single variable with no interventions.

        Source: out of RJC's head.
        """
        minimize_event_test5_in = [(Y @ -X, -Y)]
        minimize_event_test5_out = [(Y @ -X, -Y)]
        self.assertCountEqual(
            minimize_event_test5_out,
            minimize_event(event=minimize_event_test5_in, graph=self.minimize_graph_1),
        )

    def test_minimize_event_6(self):
        """Test the minimize_event function sending in a single variable with no interventions.

        Source: out of RJC's head.
        """
        minimize_event_test6_in = [(Y @ -X @ -Y, -Y)]
        minimize_event_test6_out = [(Y @ -Y, -Y)]
        self.assertCountEqual(
            minimize_event_test6_out,
            minimize_event(event=minimize_event_test6_in, graph=self.minimize_graph_1),
        )

    def test_minimize_event_7(self):
        """Test the application of the minimize_event function from [correa22a], Example 4.5.

        Source: out of RJC's head.
        """
        minimize_event_test7_in = [(Y @ -X, -Y)]
        minimize_event_test7_out = [(Y @ -X, -Y)]
        self.assertCountEqual(
            minimize_event_test7_out,
            minimize_event(event=minimize_event_test7_in, graph=self.minimize_graph_1),
        )


class TestMinimize(cases.GraphTestCase):
    r"""Test minimizing a set of counterfactual variables.

    Source: last paragraph in Section 4 of [correa22a]_, before Section 4.1.
    Mathematical expression: ||\mathbf Y_*|| = {||Y_{\mathbf x}|| | Y_{\mathbf x}} \elementof \mathbf Y_*}, and
    ||Y_{\mathbf x}|| = Y_{\mathbf t}, where \mathbf T = \mathbf X \intersect An(Y)_{G_{\overline{\mathbf X}}}}.
    (The math syntax is not necessarily cannonical LaTeX.)
    """

    minimize_graph_1 = NxMixedGraph.from_edges(
        directed=[
            (X, W),
            (W, Y),
        ],
        undirected=[(X, Y)],
    )

    minimize_graph_2 = NxMixedGraph.from_edges(
        directed=[
            (Z, X),
            (X, W),
            (W, Y),
        ],
        undirected=[(X, Y)],
    )

    def test_minimize_1(self):
        """Test the minimize function sending in a single counterfactual variable.

        Source: out of RJC's head.
        """
        minimize_test1_in = {(Y @ -W @ -X)}
        minimize_test1_out = {(Y @ -W)}
        self.assertSetEqual(
            frozenset(minimize_test1_out),
            frozenset(minimize(variables=minimize_test1_in, graph=self.minimize_graph_1)),
        )

    def test_minimize_2(self):
        """Test the minimize function for multiple counterfactual variables.

        Source: out of RJC's head.
        """
        minimize_test2_in = {(Y @ (-W, -X, -Z)), (W @ (-X, -Z))}
        minimize_test2_out = {(Y @ -W), (W @ -X)}
        self.assertSetEqual(
            frozenset(minimize_test2_out),
            frozenset(minimize(variables=minimize_test2_in, graph=self.minimize_graph_2)),
        )

    def test_minimize_3(self):
        """Test the minimize function for multiple counterfactual variables and a different graph.

        Source: out of RJC's head.
        """
        minimize_test3_in = {(Y @ (-W, -X))}
        minimize_test3_out = {(Y @ (-X, -W))}
        # self.assertSetEqual(minimize_test3_out, minimize(variables = minimize_test3_in, graph = figure_2a_graph))
        # The intentional reverse order of the interventions means we have to use P() to sort it out (no pun intended).
        self.assertSetEqual(
            frozenset(minimize_test3_out),
            frozenset(minimize(variables=minimize_test3_in, graph=figure_2a_graph)),
        )

    def test_minimize_4(self):
        """Test the minimize function sending in a single variable with no interventions.

        Source: out of RJC's head.
        """
        minimize_test4_in = {(Y)}
        minimize_test4_out = {(Y)}
        self.assertSetEqual(
            frozenset(minimize_test4_out),
            frozenset(minimize(variables=minimize_test4_in, graph=self.minimize_graph_1)),
        )

    def test_minimize_5(self):
        """Test the minimize function sending in a single variable with no interventions.

        Source: out of RJC's head.
        """
        minimize_test5_in = {(Y @ -X)}
        minimize_test5_out = {(Y @ -X)}
        self.assertSetEqual(
            frozenset(minimize_test5_out),
            frozenset(minimize(variables=minimize_test5_in, graph=self.minimize_graph_1)),
        )

    def test_minimize_6(self):
        """Test the minimize function sending in a single variable with no interventions.

        Source: out of RJC's head.
        """
        minimize_test6_in = {(Y @ -X @ -Y)}
        minimize_test6_out = {(Y @ -Y)}
        self.assertSetEqual(
            frozenset(minimize_test6_out),
            frozenset(minimize(variables=minimize_test6_in, graph=self.minimize_graph_1)),
        )

    def test_minimize_7(self):
        """Test the application of the minimize function from [correa22a], Example 4.5.

        Source: out of RJC's head.
        """
        minimize_test7_in = {(Y @ -X)}
        minimize_test7_out = {(Y @ -X)}
        self.assertSetEqual(
            frozenset(minimize_test7_out),
            frozenset(minimize(variables=minimize_test7_in, graph=self.minimize_graph_1)),
        )


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


class TestGetCounterfactualFactors(cases.GraphTestCase):
    """Test the GetCounterfactualFactors function in counterfactual_transportability.py.

    This is one step in the ctf-factor factorization process. Here we want
    to check that we can separate a joint probability distribution of ctf-factors
    into a set of joint probability distributions that we'll later multiply
    together as per Equation 15 in [correa22a]_.
    """

    # def assert_collection_of_set_equal(
    #    self, left: Collection[set[Variable]], right: Collection[set[Variable]]
    # ) -> None:
    #    """Check that two collections contain sets with the same elements."""
    #    c1 = Counter(frozenset(element) for element in left)
    #    c2 = Counter(frozenset(el) for el in right)
    #    self.assertEqual(c1, c2)

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
            (-Z),
        }
        get_counterfactual_factors_test_1_expected = [
            {(Y @ (-X, -W, -Z)), (W @ -X)},
            {(X @ -Z), (-Z)},
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
    # everything as having one ancestral set here, and check that every variable
    # that's in the returned event is also somewhere in the returned expression.
    def test_do_counterfactual_factor_factorization_1(self):
        """Test counterfactual factor factorization as per Equation 16 in [correa22a]_.

        Source: Equations 11, 14, and 16 of [correa22a]_.
        """
        # First test is already in counterfactual factor form
        equation_16_test_1_in = [(Y @ (-X, -W, -Z), -Y), (W @ -X, -W), (X @ -Z, -X), (Z, -Z)]
        equation_16_test_1_expected = (
            P([(Y @ (-X, -W, -Z)), (W @ -X)]) * P([(X @ -Z), (Z)]),
            [(Y @ (-X, -W, -Z), -Y), (X @ -Z, -X), (W @ -X, -W), (Z, -Z)],
        )
        equation_16_test_1_out = do_counterfactual_factor_factorization(
            variables=equation_16_test_1_in, graph=figure_2a_graph
        )
        self.assert_expr_equal(
            equation_16_test_1_out[0],
            equation_16_test_1_expected[0],
        )
        self.assertCountEqual(equation_16_test_1_out[1], equation_16_test_1_expected[1])

    def test_do_counterfactual_factor_factorization_2(self):
        """Test counterfactual factor factorization as per Equation 16 in [correa22a]_.

        Source: Equations 11, 14, and 16 of [correa22a]_.
        """
        # Second test is not in counterfactual factor form
        equation_16_test_2_in = [(Y @ -X, -Y), (W @ -X, -W), (X @ -Z, -X), (Z, -Z)]
        equation_16_test_2_expected = (
            P((Y @ (-X, -W, -Z)), (W @ -X)) * P((X @ -Z), (Z)),
            [(Y @ (-X, -W, -Z), -Y), (X @ -Z, -X), (W @ -X, -W), (Z, -Z)],
        )
        equation_16_test_2_out = do_counterfactual_factor_factorization(
            variables=equation_16_test_2_in, graph=figure_2a_graph
        )
        self.assert_expr_equal(
            equation_16_test_2_out[0],
            equation_16_test_2_expected[0],
        )
        self.assertCountEqual(equation_16_test_2_out[1], equation_16_test_2_expected[1])

    def test_do_counterfactual_factor_factorization_3(self):
        """Test counterfactual factor factorization as per Equation 16 in [correa22a]_.

        Source: Equations 11, 14, and 16 of [correa22a]_.
        """
        # This is the actual equation 16 content in [correa22a]_
        equation_16_test_3_in = [(Y @ -X, -Y), (X, -X)]
        equation_16_test_3_expected = (
            Sum.safe(P((Y @ (-X, -W, -Z)), (W @ -X)) * P((X @ -Z), (Z)), [Z, W]),
            [(Y @ (-X, -W, -Z), -Y), (X @ -Z, -X)],
        )
        equation_16_test_3_out = do_counterfactual_factor_factorization(
            variables=equation_16_test_3_in, graph=figure_2a_graph
        )
        self.assert_expr_equal(
            equation_16_test_3_out[0],
            equation_16_test_3_expected[0],
        )
        self.assertCountEqual(equation_16_test_3_out[1], equation_16_test_3_expected[1])


class TestConvertToCounterfactualFactorForm(unittest.TestCase):
    """Test converting a set of variables to counterfactual factor form.

    Source: Example 4.2 of [correa22a]_.
    """

    def test_convert_to_counterfactual_factor_form_1(self):
        """Test conversion of a set of variables to counterfactual factor form.

        Source: Equation 12 of [correa22a]_.
        """
        test_1_in = [(Y @ -X, -Y), (W, -W)]
        test_1_expected = [(Y @ (-X, -W, -Z), -Y), (W @ -X, -W)]
        self.assertCountEqual(
            convert_to_counterfactual_factor_form(event=test_1_in, graph=figure_2a_graph),
            test_1_expected,
        )

    def test_convert_to_counterfactual_factor_form_2(self):
        """Test conversion of a set of variables to counterfactual factor form.

        Source: Equation 12 of [correa22a]_.

        Here we pass in a simple variable with no parents, and should get it back.
        """
        test_2_in = [(Z, -Z)]
        test_2_expected = [(Z, -Z)]
        self.assertCountEqual(
            convert_to_counterfactual_factor_form(event=test_2_in, graph=figure_2a_graph),
            test_2_expected,
        )

    def test_convert_to_counterfactual_factor_form_3(self):
        """Test conversion of an outcome intervened on itself to counterfactual factor form.

        Source: out of RJC's head.
        """
        test_3_1_in = [(Y, -Y)]
        test_3_2_in = [(Y @ -Y, -Y)]
        test_3_3_in = [(Y @ -Y, -Y), (Y, -Y)]
        test_3_expected = [(Y @ (-X, -W, -Z), -Y)]
        self.assertCountEqual(
            convert_to_counterfactual_factor_form(event=test_3_1_in, graph=figure_2a_graph),
            test_3_expected,
        )
        self.assertCountEqual(
            convert_to_counterfactual_factor_form(
                event=simplify(event=test_3_2_in, graph=figure_2a_graph), graph=figure_2a_graph
            ),
            test_3_expected,
        )
        self.assertCountEqual(
            convert_to_counterfactual_factor_form(
                event=simplify(event=test_3_3_in, graph=figure_2a_graph), graph=figure_2a_graph
            ),
            test_3_expected,
        )

    def test_convert_to_counterfactual_factor_form_4(self):
        """Convert a variable with no value to counterfactual factor form.

        Source: Equation 12 of [correa22a]_.

        Here we pass in a simple variable with no parents, and should get it back.
        """
        test_4_in = [(Z, None)]
        test_4_expected = [(Z, None)]
        self.assertCountEqual(
            convert_to_counterfactual_factor_form(event=test_4_in, graph=figure_2a_graph),
            test_4_expected,
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
            tian_pearl_identify,
            input_variables=frozenset({Y, R}),
            input_district=frozenset({R, X, W, Z}),
            district_probability=PP[Population("pi*")](R, W, X, Y, Z),
            graph=soft_interventions_figure_3_graph,
            topo=list(soft_interventions_figure_3_graph.topological_sort()),
        )
        # Raises a KeyError because a variable in T is not in the topo.
        self.assertRaises(
            KeyError,
            tian_pearl_identify,
            input_variables=frozenset({X, Z}),
            input_district=frozenset({R, X, W, Z}),
            district_probability=PP[Population("pi*")](R, W, X, Y, Z),
            graph=soft_interventions_figure_3_graph,
            topo=[W, X, Z, Y],
        )
        # Raises a TypeError because G_{X,Y,Z} has two districts and there should be at most one.
        # {X,Y} happen to be in two different districts of G_{X,Y,Z}.
        self.assertRaises(
            TypeError,
            tian_pearl_identify,
            input_variables=frozenset({X, Y}),
            input_district=frozenset({X, Y, Z}),
            district_probability=PP[Population("pi*")](R, W, X, Y, Z),
            graph=soft_interventions_figure_3_graph,
            topo=list(soft_interventions_figure_3_graph.topological_sort()),
        )
        # Raises a TypeError because G_{X,Y,Z} has two districts and there should be at most one.
        # {W,Y} happen to be in the same district of G_{X,Y,Z}.
        self.assertRaises(
            TypeError,
            tian_pearl_identify,
            input_variables=frozenset({W, Y}),
            input_district=frozenset({W, Y, Z}),
            district_probability=PP[Population("pi*")](R, W, X, Y, Z),
            graph=soft_interventions_figure_3_graph,
            topo=list(soft_interventions_figure_3_graph.topological_sort()),
        )

    def test_identify_1(self):
        """Test Line 2 of Algorithm 5 of [correa22a]_.

        This tests the case where A == C.
        """
        # π_star = Pi_star = Variable(f"π*")
        test_1_identify_input_variables = {Z}  # A
        test_1_identify_input_district = {Z}  # B
        # TODO: In identify(), during pre-processing make sure that A is contained in B, and
        # that B is a district of G. Raise a TypeError if not.

        # @cthoyt @JZ The next two commented-out lines produce a mypy error:
        # pi1 = Population("pi1")
        # test_1_district_probability = PP[Population("pi1")](Z | X1)
        # error: Type application targets a non-generic function or class  [misc]
        # test_transport.py uses a similar syntax and does not trigger the error,
        #   so I'm probably missing something simple.
        test_1_district_probability = PP[Population("pi1")](Z | X1)  # Q
        result = tian_pearl_identify(
            input_variables=frozenset(test_1_identify_input_variables),
            input_district=frozenset(test_1_identify_input_district),
            district_probability=test_1_district_probability,
            graph=soft_interventions_figure_1b_graph,
            topo=list(soft_interventions_figure_1b_graph.topological_sort()),
        )
        logger.warning("Result of identify() call for test_identify_1 is " + str(result))
        self.assert_expr_equal(result, PP[Population("pi1")](Z | X1))

    def test_identify_2(self):
        """Test Line 3 of Algorithm 5 of [correa22a]_.

        This tests the case where A == T.
        Sources: a modification of the example following Theorem 2 in [correa20a]_
        and the paragraph at the end of section 4 in [correa20a]_.
        """
        result1 = tian_pearl_identify(
            input_variables=frozenset({R, Y}),
            input_district=frozenset({W, R, X, Z, Y}),
            district_probability=PP[Population("pi*")](
                W, R, X, Z, Y
            ),  # This is a c-factor if the input variables comprise a c-component
            graph=soft_interventions_figure_2a_graph,
            topo=list(soft_interventions_figure_2a_graph.topological_sort()),
        )
        logger.warning("Result of identify() call for test_identify_2 part 1 is " + str(result1))
        self.assertIsNone(result1)
        result2 = tian_pearl_identify(
            input_variables=frozenset({Z, R}),
            input_district=frozenset({R, X, W, Z}),
            district_probability=PP[Population("pi*")](R, W, X, Z),
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
        test_3_district_probability = PP[Population("pi1")]((Y, W)).conditional([R, X, Z]) * PP[
            Population("pi1")
        ](R, X)
        result1 = tian_pearl_identify(
            input_variables=frozenset(test_3_identify_input_variables),
            input_district=frozenset(test_3_identify_input_district),
            district_probability=test_3_district_probability,
            graph=soft_interventions_figure_2d_graph,
            topo=list(soft_interventions_figure_2d_graph.topological_sort()),
        )
        logger.warning("Result of identify() call for test_identify_3 is " + str(result1))
        self.assertIsNone(result1)
        result2 = tian_pearl_identify(
            input_variables=frozenset({Z, R}),
            input_district=frozenset({R, X, W, Y, Z}),
            district_probability=PP[Population("pi*")](R, W, X, Y, Z),
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
        implementation of identify in their R package, Causal Effect ([santikka20b]_), and is more
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
        result_4 = tian_pearl_identify(
            input_variables=frozenset({Y}),
            input_district=frozenset({X, Y, W1, W2, W3, W4, W5}),
            district_probability=P(W1, W2, W3, W4, W5, X, Y),
            graph=tian_pearl_figure_9a_graph,
            topo=list(tian_pearl_figure_9a_graph.topological_sort()),
        )
        self.assert_expr_equal(result_4, expected_result)


class TestComputeCFactor(cases.GraphTestCase):
    """Test the "compute_c_factor" subroutine of Tian and Pearl's identify algorithm as implemented by [santikka20a].

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
        result_1 = _compute_c_factor(
            district=[Y, W1, W3, W2, X],
            subgraph_variables=[X, W4, W2, W3, W1, Y],
            subgraph_probability=P(W1, W2, W3, W4, X, Y),
            graph_topo=list(tian_pearl_figure_9a_graph.topological_sort()),
        )
        self.assert_expr_equal(result_1, self.expected_result_1)

    def test_compute_c_factor_2(self):
        """Second test of the compute C factor subroutine, based on the example on page 29 of [tian03a]."""
        result_2 = _compute_c_factor(
            district=[W1, X, Y],
            subgraph_variables=[W1, W2, X, Y],
            subgraph_probability=Sum.safe(self.expected_result_1, [W3]),
            graph_topo=list(tian_pearl_figure_9a_graph.topological_sort()),
        )
        self.assert_expr_equal(result_2, self.expected_result_2)

    def test_compute_c_factor_3(self):
        """Third test of the compute C factor subroutine, based on the example on page 29 of [tian03a]."""
        result_3 = _compute_c_factor(
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
        result_4 = _compute_c_factor(
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
        """Fourth test of the Compute C Factor function.

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
        result_5 = _compute_c_factor(
            district=district,
            subgraph_variables=subgraph_variables,
            subgraph_probability=subgraph_probability,
            graph_topo=topo,
        )
        self.assert_expr_equal(result_5, expected_result_5)


class TestTianLemma1i(cases.GraphTestCase):
    """Test the use of Lemma 1, part (i), of [tian03a]_ to compute a C factor."""

    def test_tian_lemma_1_i_part_1(self):
        """First test of Lemma 1, part (i) (Equation 37 in [tian03a]_.

        Source: The example on p. 30 of [Tian03a]_, run initially through [santikka20a]_.
        """
        topo = [W1, W3, W2, W4, X, Y]
        part_1_graph = tian_pearl_figure_9a_graph.subgraph([Y, X, W1, W2, W3, W4])
        result_1 = _tian_lemma_1_i(
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
            _tian_lemma_1_i,
            district=[],
            topo=topo,
            graph_probability=part_1_graph.joint_probability(),
        )
        # District variable not in topo set
        self.assertRaises(
            KeyError,
            _tian_lemma_1_i,
            district=[Y, W1, W3, W2, X, Z],
            topo=topo,
            graph_probability=part_1_graph.joint_probability(),
        )

    def test_tian_lemma_1_i_part_2(self):
        """Second test of Lemma 1, part (i) (Equation 37 in [tian03a]_.

        This one handles a graph_probability conditioning on variables.
        Source: The example on p. 30 of [Tian03a]_, run initially through [santikka20a]_.
        """
        # working with tian_pearl_figure_9a_graph.subgraph([Y, X, W1, W2, W3, W4])
        topo = [W1, W3, W2, W4, X, Y]
        result_1 = _tian_lemma_1_i(
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


class TestTianLemma4ii(cases.GraphTestCase):
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

    def test_tian_lemma_4_ii_part_1(self):
        """First test of Lemma 4, part (ii) (Equations 71 and 72 in [tian03a]_.

        Source: The example on p. 30 of [Tian03a]_, run initially through [santikka20a]_.
        """
        result = _tian_lemma_4_ii(
            district={W1, X, Y},
            graph_probability=Sum.safe(self.result_piece, [W3]),
            topo=list(tian_pearl_figure_9a_graph.subgraph({W1, W2, X, Y}).topological_sort()),
        )
        logger.warning(
            "In first test of Lemma 4(ii): expecting this result: " + str(self.expected_result_1)
        )
        self.assert_expr_equal(result, self.expected_result_1)

    def test_tian_lemma_4_ii_part_2(self):
        """First test of Lemma 4, part (ii) (Equations 71 and 72 in [tian03a]_.

        Source: The example on p. 30 of [Tian03a]_, run initially through [santikka20a]_.
        """
        logger.warning(
            "In second test of Lemma 4(ii): expecting this result: " + str(self.expected_result_2)
        )
        logger.warning("Expected_result_1 = " + str(self.expected_result_1))
        result = _tian_lemma_4_ii(
            district={Y},
            graph_probability=Sum.safe(self.expected_result_1, [W1]),
            topo=list(tian_pearl_figure_9a_graph.subgraph({X, Y}).topological_sort()),
        )
        self.assert_expr_equal(result, self.expected_result_2)


class TestTianEquation72(cases.GraphTestCase):
    """Test the use of Equation 72 in Lemma 1, part (ii), of [tian03a]_."""

    def test_tian_equation_72_part_1(self):
        """First test of Equation 72 in [tian03a]_.

        Source: RJC's mind.
        """
        topo = [variable for variable in figure_2a_graph.subgraph({Z, X, Y, W}).topological_sort()]
        result = _tian_equation_72(
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
            _tian_equation_72,
            vertex={R},
            graph_probability=P(Y | W, X, Z) * P(W | X, Z) * P(X | Z) * P(Z),
            topo=topo,
        )

    def test_tian_equation_72_part_2(self):
        r"""Second test of Equation 72 in [tian03a]_, checking $Q[H^{(0)}]=Q[\emptyset]$.

        Source: RJC's mind.
        """
        topo = [variable for variable in figure_2a_graph.subgraph({Z, X, Y, W}).topological_sort()]
        result = _tian_equation_72(
            vertex=None,
            graph_probability=P(Y | W, X, Z) * P(W | X, Z) * P(X | Z) * P(Z),
            topo=topo,
        )
        self.assert_expr_equal(result, One())


class TestTianLemma3(cases.GraphTestCase):
    """Test the use of Lemma 3 (i.e., Equation 69) of [tian03a]_ to compute a C factor."""

    def test_tian_lemma_3_part_1(self):
        """First test of Lemma 3 in [tian03a]_ (Equation 69).

        Source: The example on p. 30 of [Tian03a]_, run initially through [santikka20a]_.
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
        result_1 = _tian_equation_69(
            ancestral_set=ancestral_set,
            subgraph_variables=subgraph_variables,
            subgraph_probability=subgraph_probability,
            graph_topo=topo,
        )
        expected_result_1 = Sum.safe(subgraph_probability, [W3])
        self.assert_expr_equal(expected_result_1, result_1)
