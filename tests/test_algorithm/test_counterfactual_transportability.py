"""Tests for counterfactual transportability.

.. [huang08a] https://link.springer.com/article/10.1007/s10472-008-9101-x.
.. [correa20a] https://proceedings.neurips.cc/paper/2020/file/7b497aa1b2a83ec63d1777a88676b0c2-Paper.pdf.
.. [correa22a] https://proceedings.mlr.press/v162/correa22a/correa22a.pdf.
.. [tikka20a] https://github.com/santikka/causaleffect/blob/master/R/compute.c.factor.R.
.. [tikka20b] https://github.com/santikka/causaleffect/blob/master/R/identify.R.
.. [tian03a] https://ftp.cs.ucla.edu/pub/stat_ser/R290-L.pdf.
"""

import logging
import unittest
from collections import defaultdict

from networkx import NetworkXError

from tests.test_algorithm import cases
from y0.algorithm.counterfactual_transport.ancestor_utils import (
    _compute_ancestral_components_from_ancestral_sets,
    _get_ancestral_set_after_intervening_on_conditioned_variables,
    _get_conditioned_variables_in_ancestral_set,
    _merge_frozen_sets_linked_by_bidirectional_edges,
    _merge_frozen_sets_with_common_vertices,
    _minimize_set,
    get_ancestors_of_counterfactual,
    get_ancestral_components,
    get_base_variables,
)
from y0.algorithm.counterfactual_transport.api import (
    _any_inconsistent_intervention_values,
    _any_variable_values_inconsistent_with_interventions,
    _any_variables_with_inconsistent_values,
    _counterfactual_factor_is_inconsistent,
    _initialize_conditional_transportability_data_structures,
    _no_intervention_variables_in_domain,
    _no_transportability_nodes_in_domain,
    _reduce_reflexive_counterfactual_variables_to_interventions,
    _remove_repeated_variables_and_values,
    _remove_transportability_vertices,
    _split_event_by_reflexivity,
    _transport_conditional_counterfactual_query_line_2,
    _transport_conditional_counterfactual_query_line_4,
    _transport_unconditional_counterfactual_query_line_2,
    _valid_topo_list,
    _validate_transport_conditional_counterfactual_query_input,
    _validate_transport_conditional_counterfactual_query_line_4_output,
    _validate_transport_unconditional_counterfactual_query_input,
    convert_to_counterfactual_factor_form,
    counterfactual_factors_are_transportable,
    do_counterfactual_factor_factorization,
    get_counterfactual_factors,
    get_counterfactual_factors_retaining_variable_values,
    is_counterfactual_factor_form,
    minimize_event,
    same_district,
    simplify,
    transport_conditional_counterfactual_query,
    transport_district_intervening_on_parents,
    transport_unconditional_counterfactual_query,
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
    CounterfactualVariable,
    Fraction,
    Intervention,
    One,
    P,
    Pi1,
    Pi2,
    Product,
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

logger = logging.getLogger(__name__)

# [correa22a]_, Figure 1, without the transportability node.
# (This graph represents the target domain, so there is no
# transportability node. Figure 1 may include a transportability
# node because at that point in the paper, the notion of target
# and source domains had not been introduced.)
figure_1_graph_no_transportability_nodes = NxMixedGraph.from_edges(
    directed=[
        (X, Z),
        (Z, Y),
        (X, Y),
    ],
    undirected=[(Z, X)],
)
figure_1_graph_no_transportability_nodes_topo = list(
    figure_1_graph_no_transportability_nodes.topological_sort()
)

# The graph for Domain 1 as described by the text of Example 1.1 and
# Figure 1 of [correa22a]_. The graph isn't in any figures in the
# paper, but a reader can infer it.
figure_1_graph_domain_1_with_interventions = NxMixedGraph.from_edges(
    directed=[(X, Z), (X, Y), (Z, Y), (transport_variable(Y), Y)],
    undirected=[],
)
figure_1_graph_domain_1_with_interventions_topo = list(
    figure_1_graph_domain_1_with_interventions.topological_sort()
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

# From [correa22a]_, Figures 3a and 4.
figure_2_graph_domain_1_with_interventions = NxMixedGraph.from_edges(
    directed=[(X, Y), (X, W), (W, Y), (Z, Y), (transport_variable(Z), Z)],
    undirected=[(W, Y)],
)
figure_2_graph_domain_1_with_interventions_topo = list(
    figure_2_graph_domain_1_with_interventions.topological_sort()
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
figure_2_graph_domain_2_topo = list(figure_2_graph_domain_2.topological_sort())

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

    def test_6(self):
        """Test the pre-processing check for get_ancestors_of_counterfactual().

        Send in anything that's not a variable.
        Source: out of RJC's head.
        """
        test6_in = [
            CounterfactualVariable(
                name="Y", star=None, interventions=frozenset({Intervention(name="Y", star=False)})
            ),
            CounterfactualVariable(
                name="X", star=None, interventions=frozenset({Intervention(name="Y", star=False)})
            ),
        ]
        self.assertRaises(
            TypeError,
            get_ancestors_of_counterfactual,
            event=test6_in,
            graph=figure_2a_graph,
        )
        self.assertRaises(
            TypeError,
            get_ancestors_of_counterfactual,
            event=[],
            graph=figure_2a_graph,
        )

    def test_7(self):
        """Make sure the star values of interventions are properly handled for get_ancestors_of_counterfactual().

        Source: out of RJC's head.
        """
        test7_in = W @ +X
        test7_out = {W @ +X}
        result = get_ancestors_of_counterfactual(event=test7_in, graph=figure_2a_graph)
        logger.debug("In test_7: result = " + str(result))
        self.assertTrue(variable in test7_out for variable in result)


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
    #       x 3. Test this algorithm and the functions it calls for events
    #           with values of None assigned to variables.
    #       x 4. A. Trigger this line in _any_variables_with_inconsistent_values():
    #             if any(len(value_set) > 1 and None in value_set for value_set in \
    #                nonreflexive_variable_to_value_mappings.values()):
    #                raise TypeError("In _any_variables_with_inconsistent values: "+\
    #                       " a variable lacking interventions on itself "+\
    #                       "has an assigned value and also a value of None. "+\
    #                       "That should not occur. Check your inputs.")
    #           B. Also trigger the equivalent line checking the dictionary containing
    #            variable-to-value mappings for variables with reflexive interventions.
    #           C. Also make sure that _any_variables_with_inconsistent_values()
    #            handles None values for reflexive and nonreflexive interventions
    #            in the case where there are no conflicts with other intervention values.
    #       x 5. Test _reduce_reflexive_counterfactual_variables_to_interventions()
    #            for a case in which a variable, counterfactual or otherwise, has a
    #            value of None.
    #       x 6. Test _split_event_by_reflexivity() for cases in which variables have
    #            values of None.
    #       x 7. Test that _remove_repeated_variables_and_values() removes None as a
    #            duplicate value for a variable that also has an actual value.
    def test_inconsistent_1(self):
        """Test simplifying an inconsistent event.

        Correa et al. specify the output should be 0 if the counterfactual event
        is guaranteed to have probability 0. Source: RJC's mind.
        """
        event = [(Y @ -X, -Y), (Y @ -X, +Y)]
        result = simplify(event=event, graph=figure_2a_graph)
        logger.debug("Result for test_inconsistent_1 is " + str(result))
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
        reflexive_event_4, nonreflexive_event_4 = _split_event_by_reflexivity(
            event=[(Y @ -X, -Y), (Y @ -X, None), (Y @ -Y, None)]
        )
        self.assertCountEqual(reflexive_event_4, [(Y @ -Y, None)])
        self.assertCountEqual(nonreflexive_event_4, [(Y @ -X, -Y), (Y @ -X, None)])

    def test_reduce_reflexive_counterfactual_variables_to_interventions_part_1(self):
        """Test reducing counterfactual variables that intervene on themselves to simpler Intervention objects.

        We test both TypeError tests and normal functioning of the function.
        Source: RJC's mind.
        """
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ -Y].add(+Y)
        reflexive_variable_to_value_mappings[Y].add(-Y)
        result_dict = _reduce_reflexive_counterfactual_variables_to_interventions(
            reflexive_variable_to_value_mappings
        )
        self.assertIn(Y, result_dict)
        self.assertSetEqual(result_dict[Y], {+Y, -Y})
        # The next test sends in a counterfactual variable intervening on itself and something else.
        # The minimize() algorithm should always get rid of the intervention that is not the
        # self-intervention, but here we're testing a pre-processing check in this algorithm
        # that looks for input variables with multiple interventions.
        reflexive_variable_to_value_mappings_2 = defaultdict(set)
        reflexive_variable_to_value_mappings_2[Y @ [-Y, -X]].add(+Y)
        reflexive_variable_to_value_mappings_2[Y].add(-Y)
        self.assertRaises(
            ValueError,
            _reduce_reflexive_counterfactual_variables_to_interventions,
            variables=reflexive_variable_to_value_mappings_2,
        )
        # Here we're testing a check that all input interventions are self-interventions.
        reflexive_variable_to_value_mappings_3 = defaultdict(set)
        reflexive_variable_to_value_mappings_3[Y @ -X].add(+Y)
        reflexive_variable_to_value_mappings_3[Y].add(-Y)
        self.assertRaises(
            ValueError,
            _reduce_reflexive_counterfactual_variables_to_interventions,
            variables=reflexive_variable_to_value_mappings_3,
        )
        # Here we test the case where a variable has a value of None.
        reflexive_variable_to_value_mappings_4 = defaultdict(set)
        reflexive_variable_to_value_mappings_4[Y @ -Y].add(+Y)
        reflexive_variable_to_value_mappings_4[Y].add(None)
        result_dict_4 = _reduce_reflexive_counterfactual_variables_to_interventions(
            reflexive_variable_to_value_mappings_4
        )
        self.assertIn(Y, result_dict_4)
        self.assertSetEqual(result_dict_4[Y], {+Y, None})
        # Here we test the case where a counterfactual variable has a value of None.
        reflexive_variable_to_value_mappings_5 = defaultdict(set)
        reflexive_variable_to_value_mappings_5[Y @ -Y].add(None)
        reflexive_variable_to_value_mappings_5[Y].add(-Y)
        result_dict_5 = _reduce_reflexive_counterfactual_variables_to_interventions(
            reflexive_variable_to_value_mappings_5
        )
        self.assertIn(Y, result_dict_5)
        self.assertSetEqual(result_dict_5[Y], {-Y, None})

    def test_remove_repeated_variables_and_values(self):
        """Test removing repeated occurrences of variables and associated values in an event.

        Source: RJC's mind.
        """
        event_1 = [(Y @ -Y, -Y), (Y @ -Y, -Y)]
        event_2 = [(X, -X), (X, -X)]
        event_3 = [(X, -X), (X, None)]
        event_4 = [(Y @ -Y, None), (Y @ -Y, -Y)]
        result_1 = _remove_repeated_variables_and_values(event=event_1)
        result_2 = _remove_repeated_variables_and_values(event=event_2)
        result_3 = _remove_repeated_variables_and_values(event=event_3)
        result_4 = _remove_repeated_variables_and_values(event=event_4)
        # @cthoyt: Just curious: is there a one-liner to assert that two dictionaries are equal?
        self.assertCountEqual([Y @ -Y], result_1.keys())
        self.assertEqual(len(result_1[Y @ -Y]), 1)
        self.assertSetEqual(result_1[Y @ -Y], {-Y})
        self.assertCountEqual([X], result_2.keys())
        self.assertEqual(len(result_2[X]), 1)
        self.assertSetEqual(result_2[X], {-X})
        self.assertCountEqual([X], result_3.keys())
        self.assertEqual(len(result_3[X]), 1)
        self.assertSetEqual(result_3[X], {-X})
        self.assertCountEqual([Y @ -Y], result_4.keys())
        self.assertEqual(len(result_4[Y @ -Y]), 1)
        self.assertSetEqual(result_4[Y @ -Y], {-Y})

    def test_line_2_1(self):
        """Directly test the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls."""
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ -Y].add(-Y)

        nonreflexive_variable_to_value_mappings = defaultdict(set)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(-Y)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(+Y)

        logger.debug(
            "In test_line_2_1: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.debug(
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
        logger.debug(
            "In test_line_2_2: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.debug(
            "In test_line_2_2: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        self.assertFalse(
            _any_variables_with_inconsistent_values(
                nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
                reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
            )
        )

    def test_line_2_10(self):
        """Tenth test for the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls."""
        reflexive_variable_to_value_mappings = defaultdict(set)

        nonreflexive_variable_to_value_mappings = defaultdict(set)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(None)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(None)
        logger.debug(
            "In test_line_2_10: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.debug(
            "In test_line_2_10: reflexive_variable_to_value_mappings = "
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
        logger.debug(
            "In test_line_2_3: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.debug(
            "In test_line_2_3: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        self.assertTrue(
            _any_variables_with_inconsistent_values(
                nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
                reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
            )
        )

    def test_line_2_11(self):
        """Eleventh test for the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls.

        This one's a little subtle: Y@-Y with a value of None evaluates to Y with a value of -Y and also None,
        which should raise an error instead of merely signalling that values are inconsistent and therefore
        the expression has a probability of Zero.
        """
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ -Y].add(None)

        nonreflexive_variable_to_value_mappings = defaultdict(set)
        logger.debug(
            "In test_line_2_11: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.debug(
            "In test_line_2_11: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        self.assertRaises(
            TypeError,
            _any_variables_with_inconsistent_values,
            nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
            reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
        )

    def test_line_2_4(self):
        """Fourth test for the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls."""
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ -Y].add(-Y)

        nonreflexive_variable_to_value_mappings = defaultdict(set)

        logger.debug(
            "In test_line_2_4: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.debug(
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
        logger.debug(
            "In test_line_2_5: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.debug(
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
        logger.debug(
            "In test_line_2_6: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.debug(
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

        logger.debug(
            "In test_line_2_7: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.debug(
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

    def test_line_2_8(self):
        """Eighth test for the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls.

        We're checking that an error gets thrown when a nonreflexive intervention has a value that is None
        and also a value that is something else.
        """
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ -Y].add(-Y)

        nonreflexive_variable_to_value_mappings = defaultdict(set)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(-Y)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(None)

        logger.debug(
            "In test_line_2_8: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.debug(
            "In test_line_2_8: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        self.assertRaises(
            TypeError,
            _any_variables_with_inconsistent_values,
            nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
            reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
        )

    def test_line_2_9(self):
        """Ninth test for the internal function _any_variables_with_inconsistent_values() that SIMPLIFY calls.

        We're checking that an error gets thrown when a reflexive intervention has a value that is None
        and also a value that is something else.
        """
        reflexive_variable_to_value_mappings = defaultdict(set)
        reflexive_variable_to_value_mappings[Y @ -Y].add(-Y)
        reflexive_variable_to_value_mappings[Y @ -Y].add(None)

        nonreflexive_variable_to_value_mappings = defaultdict(set)
        nonreflexive_variable_to_value_mappings[Y @ -X].add(-Y)

        logger.debug(
            "In test_line_2_9: nonreflexive_variable_to_value_mappings = "
            + str(nonreflexive_variable_to_value_mappings)
        )
        logger.debug(
            "In test_line_2_9: reflexive_variable_to_value_mappings = "
            + str(reflexive_variable_to_value_mappings)
        )
        self.assertRaises(
            TypeError,
            _any_variables_with_inconsistent_values,
            nonreflexive_variable_to_value_mappings=nonreflexive_variable_to_value_mappings,
            reflexive_variable_to_value_mappings=reflexive_variable_to_value_mappings,
        )

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
        # First variable in the tuple must be a variable, second must be an intervention.
        # Pass in a list of variables for the first and a list of interventions for the
        # second test below
        event_4 = [([Y @ -Y, Y @ -X], -Y)]
        event_5 = [(Y @ -Y, [-Y, -X])]
        self.assertRaises(TypeError, simplify, event=event_4, graph=figure_2a_graph)
        self.assertRaises(TypeError, simplify, event=event_5, graph=figure_2a_graph)

    def test_simplify_1(self):
        """Test of the full simplify algorithm. Source: RJC's mind."""
        event_1 = [
            (Y @ -X, -Y),
            (Y @ -Z, -Y),
        ]
        self.assertCountEqual(
            simplify(event=event_1, graph=figure_2a_graph), [(Y @ -Z, -Y), (Y @ -X, -Y)]
        )
        # Check that simplify() correctly processes None as a value.
        event_2 = [
            (Y @ -X, None),
            (Y @ -Z, -Y),
        ]
        self.assertCountEqual(
            simplify(event=event_2, graph=figure_2a_graph), [(Y @ -Z, -Y), (Y @ -X, None)]
        )
        # Check that simplify() gets rid of None as a value when it's redundant with another counterfactual variable.
        event_3 = [
            (Y @ -X, -Y),
            (Y @ -X, None),
            (Y @ -Z, -Y),
        ]
        self.assertCountEqual(
            simplify(event=event_3, graph=figure_2a_graph), [(Y @ -Z, -Y), (Y @ -X, -Y)]
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
        event_15 = [(Y @ [-W, -Y], -Y), (Y, -Y)]
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
        self.assertCountEqual(simplify(event=event_15, graph=figure_2a_graph), [(Y, -Y)])


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

    def test_minimize_event_8(self):
        """Test the minimize_event function for variables with a value of None.

        Source: out of RJC's head.
        """
        minimize_event_test8_in = [(Y @ (-W, -X, -Z), None), (W @ (-X, -Z), -W)]
        minimize_event_test8_out = [(Y @ -W, None), (W @ -X, -W)]
        self.assertCountEqual(
            minimize_event_test8_out,
            frozenset(minimize_event(event=minimize_event_test8_in, graph=self.minimize_graph_2)),
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
            frozenset(_minimize_set(variables=minimize_test1_in, graph=self.minimize_graph_1)),
        )

    def test_minimize_2(self):
        """Test the minimize function for multiple counterfactual variables.

        Source: out of RJC's head.
        """
        minimize_test2_in = {(Y @ (-W, -X, -Z)), (W @ (-X, -Z))}
        minimize_test2_out = {(Y @ -W), (W @ -X)}
        self.assertSetEqual(
            frozenset(minimize_test2_out),
            frozenset(_minimize_set(variables=minimize_test2_in, graph=self.minimize_graph_2)),
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
            frozenset(_minimize_set(variables=minimize_test3_in, graph=figure_2a_graph)),
        )

    def test_minimize_4(self):
        """Test the minimize function sending in a single variable with no interventions.

        Source: out of RJC's head.
        """
        minimize_test4_in = {(Y)}
        minimize_test4_out = {(Y)}
        self.assertSetEqual(
            frozenset(minimize_test4_out),
            frozenset(_minimize_set(variables=minimize_test4_in, graph=self.minimize_graph_1)),
        )

    def test_minimize_5(self):
        """Test the minimize function sending in a single variable with no interventions.

        Source: out of RJC's head.
        """
        minimize_test5_in = {(Y @ -X)}
        minimize_test5_out = {(Y @ -X)}
        self.assertSetEqual(
            frozenset(minimize_test5_out),
            frozenset(_minimize_set(variables=minimize_test5_in, graph=self.minimize_graph_1)),
        )

    def test_minimize_6(self):
        """Test the minimize function sending in a single variable with no interventions.

        Source: out of RJC's head.
        """
        minimize_test6_in = {(Y @ -X @ -Y)}
        minimize_test6_out = {(Y @ -Y)}
        self.assertSetEqual(
            frozenset(minimize_test6_out),
            frozenset(_minimize_set(variables=minimize_test6_in, graph=self.minimize_graph_1)),
        )

    def test_minimize_7(self):
        """Test the application of the minimize function from [correa22a], Example 4.5.

        Source: out of RJC's head.
        """
        minimize_test7_in = {(Y @ -X)}
        minimize_test7_out = {(Y @ -X)}
        self.assertSetEqual(
            frozenset(minimize_test7_out),
            frozenset(_minimize_set(variables=minimize_test7_in, graph=self.minimize_graph_1)),
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
            ValueError,
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

    def test_do_counterfactual_factor_factorization_4(self):
        """Test counterfactual factor factorization with an empty query.

        Source: RJC's head (inspired by a code coverage report).
        """
        self.assertRaises(
            TypeError,
            do_counterfactual_factor_factorization,
            variables=[],
            graph=figure_2a_graph,
        )


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

    def test_convert_to_counterfactual_factor_form_5(self):
        """Convert a variable intervening on an ancestor that is not a parent to counterfactual factor form.

        Source: RJC's mind.

        Here we should expect the ancestor X2 to not be an intervention for Z in the output.
        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (X1, W),
                (X1, Y),
                (X2, W),
                (W, Y),
                (W, Z),
            ],
        )

        test_5_in = [(Y @ -X1, -Y), (Z @ -X2, -Z)]
        test_5_expected = [(Y @ [-X1, -W], -Y), (Z @ -W, -Z)]
        self.assertCountEqual(
            convert_to_counterfactual_factor_form(event=test_5_in, graph=graph),
            test_5_expected,
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


class TestRemoveTransportabilityVertices(cases.GraphTestCase):
    """Test the _remove_transportability_vertices utility function."""

    def test_remove_transportability_vertices(self):
        """Simple test that we're properly removing transportability nodes from a set.

        Source: RJC.
        """
        test_graph_1 = NxMixedGraph.from_edges(
            directed=[(X, Y), (X, W), (W, Y), (Z, Y), (transport_variable(Z), Z)],
            undirected=[
                (W, Y),
            ],
        )
        input_vertices = set(test_graph_1.topological_sort())
        expected_result = frozenset({W, X, Y, Z})
        self.assertSetEqual(
            expected_result, _remove_transportability_vertices(vertices=input_vertices)
        )
        test_graph_2 = NxMixedGraph.from_edges(
            directed=[(X, Y), (X, W), (W, Y), (Z, Y)],
            undirected=[
                (W, Y),
            ],
        )
        input_vertices_2 = set(test_graph_2.topological_sort())
        expected_result_2 = frozenset({W, X, Y, Z})
        self.assertSetEqual(
            expected_result_2, _remove_transportability_vertices(vertices=input_vertices_2)
        )


class TestTransportDistrictInterveningOnParents(cases.GraphTestCase):
    """Test Correa and Bareinboim's algorithm to transport a district (Algorithm 4 from [correa22a]_)."""

    def test_transport_district_intervening_on_parents_1(self):
        """First test case involving a transportable counterfactual factor.

        Source: Equation 17 of [correa22a]_.
        """
        district = {Y, W}
        domain_graphs = [
            (
                figure_2_graph_domain_1_with_interventions,
                figure_2_graph_domain_1_with_interventions_topo,
            ),
            (
                figure_2_graph_domain_2,
                figure_2_graph_domain_2_topo,
            ),
        ]
        domain_data = [({X}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))]
        expected_result = PP[Pi1](Y | W, X, Z) * PP[Pi1](W | X)
        result = transport_district_intervening_on_parents(
            district=district, domain_graphs=domain_graphs, domain_data=domain_data
        )
        self.assert_expr_equal(expected_result, result)

    def test_transport_district_intervening_on_parents_2(self):
        """Second test case involving a transportable counterfactual factor.

        Source: Equation 19 of [correa22a]_.
        """
        district = {X, Z}
        domain_graphs = [
            (
                figure_2_graph_domain_1_with_interventions,
                figure_2_graph_domain_1_with_interventions_topo,
            ),
            (
                figure_2_graph_domain_2,
                figure_2_graph_domain_2_topo,
            ),
        ]
        domain_data = [({X}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))]
        expected_result = PP[Pi2](X | Z) * PP[Pi2](Z)
        logger.debug(
            "In test_transport_district_intervening_on_parents_2: expected_result is "
            + expected_result.to_latex()
        )
        result = transport_district_intervening_on_parents(
            district=district, domain_graphs=domain_graphs, domain_data=domain_data
        )
        self.assert_expr_equal(expected_result, expected_result)
        self.assert_expr_equal(expected_result, PP[Pi2](X | Z) * PP[Pi2](Z))
        self.assert_expr_equal(expected_result, result)

    def test_transport_district_intervening_on_parents_3(self):
        """Test case in which a transportability node interferes with transportability for every domain.

        Source: RJC's mind.
        """
        # This is figure_2_graph_domain_1, changing the transportability node from T(Z) to T(Y)
        graph_1 = NxMixedGraph.from_edges(
            directed=[(X, Y), (X, W), (W, Y), (Z, Y), (transport_variable(Y), Y)],
            undirected=[
                (W, Y),
            ],
        )
        graph_1_topo = list(graph_1.topological_sort())
        district = {W, Y}
        domain_graphs = [
            (graph_1, graph_1_topo),
            (
                figure_2_graph_domain_2,
                figure_2_graph_domain_2_topo,
            ),
        ]
        domain_data = [({X}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))]
        self.assertIsNone(
            transport_district_intervening_on_parents(
                district=district, domain_graphs=domain_graphs, domain_data=domain_data
            )
        )

    def test_transport_district_intervening_on_parents_4(self):
        """Test case in which nothing's transportable because there's an interfering intervention for every domain.

        Source: RJC's mind.
        """
        district = {W, Y}
        graph_1 = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (X, Y),
                (Z, Y),
                (W, Y),
            ],
            undirected=[
                (Z, X),
            ],
        )
        graph_1_topo = list(graph_1.topological_sort())
        graph_2 = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (X, W),
                (Z, Y),
                (W, Y),
            ],
            undirected=[
                (W, Y),
            ],
        )
        graph_2_topo = list(graph_2.topological_sort())
        domain_data = [({W}, PP[Pi1](W, X, Y, Z)), ({Y}, PP[Pi2](W, X, Y, Z))]
        self.assertIsNone(
            transport_district_intervening_on_parents(
                district=district,
                domain_graphs=[(graph_1, graph_1_topo), (graph_2, graph_2_topo)],
                domain_data=domain_data,
            )
        )

    def test_transport_district_intervening_on_parents_5(self):
        """Test case in which a transportability node blocks one domain and an intervention blocks the other.

        Source: RJC's mind.
        """
        district = {W, Y}
        graph_1 = NxMixedGraph.from_edges(
            directed=[(Z, X), (X, Y), (Z, Y), (W, Y), (transport_variable(Y), Y)],
            undirected=[
                (Z, X),
            ],
        )
        graph_1_topo = list(graph_1.topological_sort())
        graph_2 = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (X, W),
                (Z, Y),
                (W, Y),
            ],
            undirected=[
                (W, Y),
            ],
        )
        graph_2_topo = list(graph_2.topological_sort())
        domain_data = [(set(), PP[Pi1](W, X, Y, Z)), ({Y}, PP[Pi2](W, X, Y, Z))]
        self.assertIsNone(
            transport_district_intervening_on_parents(
                district=district,
                domain_graphs=[(graph_1, graph_1_topo), (graph_2, graph_2_topo)],
                domain_data=domain_data,
            )
        )

    def test_transport_district_intervening_on_parents_6(self):
        """Sixth test case involving a transportable counterfactual factor.

        Here we make the input district contain variables in more than one district.

        Source: Equation 17 of [correa22a]_.
        """
        district = {Y, W, X}  # X is from another district
        domain_graphs = [
            (
                figure_2_graph_domain_1_with_interventions,
                figure_2_graph_domain_1_with_interventions_topo,
            ),
            (
                figure_2_graph_domain_2,
                figure_2_graph_domain_2_topo,
            ),
        ]
        domain_data = [(set(), PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))]
        self.assertRaises(
            ValueError,
            transport_district_intervening_on_parents,
            district=district,
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )

    def test_transport_district_intervening_on_parents_7(self):
        """Seventh test case involving a transportable counterfactual factor.

        We need a test case in which a call to tian_pearl_identify() returns None.

        Source: RC's mind.
        """
        district = {Y, Z}
        # For reference: an example (unused) target graph such that Y and Z are one of the C-components
        # target_graph = NxMixedGraph.from_edges(
        #    directed=[(X,Y)],
        #    undirected=[
        #        (Y, Z),
        #    ],
        # )
        domain_graphs = [
            (
                NxMixedGraph.from_edges(
                    directed=[(X, Y)],
                    undirected=[
                        (Y, Z),
                        (X, Y),
                    ],
                ),
                [X, Y, Z],
            ),
            (
                NxMixedGraph.from_edges(
                    directed=[(X, Y)],
                    undirected=[
                        (Y, Z),
                        (X, Y),
                    ],
                ),
                [X, Y, Z],
            ),
        ]
        domain_data = [(set(), PP[TARGET_DOMAIN](X, Y, Z)), ({Y}, PP[Pi1](X, Y, Z))]
        self.assertIsNone(
            transport_district_intervening_on_parents(
                district=district, domain_graphs=domain_graphs, domain_data=domain_data
            )
        )

    def test_transport_district_intervening_on_parents_preprocessing(self):
        """Tests of the various data integrity checks at the start of sigma-tr.

        Source: RJC.
        """
        domain_graphs = [
            (
                figure_2_graph_domain_1_with_interventions,
                figure_2_graph_domain_1_with_interventions_topo,
            ),
            (
                figure_2_graph_domain_2,
                figure_2_graph_domain_2_topo,
            ),
        ]
        domain_data = [({X}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))]
        # Wrong data type for district (not a collection of variable objects)
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district=X,
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district=[1, 2, 3],
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )
        # Wrong data type for domain_graphs
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=X,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=None,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[
                figure_2_graph_domain_1_with_interventions,
                figure_2_graph_domain_2,
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[
                (
                    None,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[
                (
                    figure_2_graph_domain_2,
                    {X, Y},
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[
                (
                    figure_2_graph_domain_2,
                    [1, 2, 3],
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[
                (
                    figure_2_graph_domain_2,
                    [X, Y, 1],
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        # Wrong data type for domain_data. TODO: consider cases where a probability is One() or Zero()
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, R},
            domain_graphs=domain_graphs,
            domain_data=None,
        )
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, R},
            domain_graphs=domain_graphs,
            domain_data=({X}, set()),
        )
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, R},
            domain_graphs=domain_graphs,
            domain_data=[(X, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, R},
            domain_graphs=domain_graphs,
            domain_data=[({1}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, R},
            domain_graphs=domain_graphs,
            domain_data=[({X}, None), (set(), PP[Pi2](W, X, Y, Z))],
        )
        # Empty district
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={},
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )
        # Empty domain graphs
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[],
            domain_data=domain_data,
        )
        # Graph with empty nodes
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[
                (
                    NxMixedGraph(),
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        # Empty domain data
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=domain_graphs,
            domain_data=[],
        )
        # Empty topo list
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    [],
                ),
            ],
            domain_data=domain_data,
        )
        # Length of the domain_graphs is not the same as the length of the domain_data
        self.assertRaises(
            TypeError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=domain_data,
        )
        # topo vertices != graph vertices
        self.assertRaises(
            KeyError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[(X, Y), (X, W), (W, Y), (Z, Y), (transport_variable(Z), Z)],
                        undirected=[
                            (W, Y),
                        ],
                    ),
                    [X, Y, Z, R],
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        # expression vertices not contained in graph vertices
        self.assertRaises(
            KeyError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=[(set(), PP[Pi1](W, X, Y)), ({Y}, PP[Pi2](W, X, Y, Z))],
        )
        # Policy variable not in the graph vertices
        self.assertRaises(
            KeyError,
            transport_district_intervening_on_parents,
            district={W, Y},
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=[({R}, PP[Pi1](W, X, Y, Z)), ({Y}, PP[Pi2](W, X, Y, Z))],
        )
        # District variable not in a graph
        self.assertRaises(
            KeyError,
            transport_district_intervening_on_parents,
            district={X, R},
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )

    def test_transport_district_intervening_on_parents_line_1(self):
        """Tests of the checks in Line 1 of sigma-TR (Algorithm 4 of [correa22a]_).

        Source: RJC.
        """
        self.assertTrue(_no_intervention_variables_in_domain(district={Y, W}, interventions={X}))
        self.assertFalse(_no_intervention_variables_in_domain(district={X, Z}, interventions={X}))
        self.assertTrue(
            _no_transportability_nodes_in_domain(
                district={Y, W}, domain_graph=figure_2_graph_domain_1_with_interventions
            )
        )
        self.assertFalse(
            _no_transportability_nodes_in_domain(
                district={X, Z}, domain_graph=figure_2_graph_domain_1_with_interventions
            )
        )


class TestInconsistentCounterfactualFactorVariableAndInterventionValues(cases.GraphTestCase):
    """Test a check of whether a counterfactual factor has any variable value inconsistent with an intervention.

    See Definition 4.1, part (i) from [correa22a]_.
    """

    def test_any_variable_values_inconsistent_with_interventions_1(self):
        """Test #1 for whether a counterfactual factor variable has a value inconsistent with any intervention.

        Source: RJC
        """
        event = {(Y @ [-X, -W, -Z], -Y), (W @ -X, -W), (X @ -Z, -X), (Z, -Z)}
        self.assertFalse(_any_variable_values_inconsistent_with_interventions(event=event))

    def test_any_variable_values_inconsistent_with_interventions_2(self):
        """Test #2 for whether a counterfactual factor variable has a value inconsistent with any intervention.

        Source: RJC
        """
        event = {(Y @ [+X, -W, -Z], -Y), (W @ -X, -W), (X @ -Z, -X), (Z, -Z)}
        self.assertTrue(_any_variable_values_inconsistent_with_interventions(event=event))

    def test_any_variable_values_inconsistent_with_interventions_3(self):
        """Test #3 for whether a counterfactual factor variable has a value inconsistent with any intervention.

        Source: RJC
        """
        event = {(Y @ [+X, -W, -Z], -Y), (W @ +X, -W), (X @ -Z, +X), (Z, -Z)}
        self.assertFalse(_any_variable_values_inconsistent_with_interventions(event=event))

    def test_any_variable_values_inconsistent_with_interventions_4(self):
        """Test #4 for whether a counterfactual factor variable has a value inconsistent with any intervention.

        Source: RJC
        """
        event = {(Y @ [-X, -W, -Z], -Y), (W @ -X, -W), (X @ -Z, +X), (Z, -Z)}
        self.assertTrue(_any_variable_values_inconsistent_with_interventions(event=event))

    def test_any_variable_values_inconsistent_with_interventions_5(self):
        """Test #5 for whether a counterfactual factor variable has a value inconsistent with any intervention.

        Source: RJC
        """
        event = {(Y @ [+X, -W, -Z], -Y), (W @ +X, -W), (X @ -Z, -X), (Z, -Z)}
        self.assertTrue(_any_variable_values_inconsistent_with_interventions(event=event))

    def test_any_variable_values_inconsistent_with_interventions_6(self):
        """Test #6 for whether a counterfactual factor variable has a value inconsistent with any intervention.

        This is a case that is inconsistent according to Definition 4.1(ii)
        but not Definition 4.1(i).
        Source: RJC
        """
        # graph = NxMixedGraph.from_edges(
        #    directed=[
        #        (X, Y),
        #        (X, Z),
        #    ],
        #    undirected=[(Y, Z)],
        # )
        event = {(Y @ -X, -Y), (Z @ +X, -Z)}
        self.assertFalse(_any_variable_values_inconsistent_with_interventions(event=event))


class TestInconsistentCounterfactualFactorVariableInterventionValues(cases.GraphTestCase):
    """Test function checking whether a counterfactual factor has any inconsistent intervention values.

    See Definition 4.1, part (ii) from [correa22a]_.
    """

    def test_any_inconsistent_intervention_values_1(self):
        """Test #1 for whether a counterfactual factor has any inconsistent intervention values.

        Source: RJC
        """
        event = {(Y @ [-X, -W, -Z], -Y), (W @ -X, -W), (X @ -Z, +X), (Z, -Z)}
        self.assertFalse(_any_inconsistent_intervention_values(event=event))

    def test_any_inconsistent_intervention_values_2(self):
        """Test #2 for whether a counterfactual factor has any inconsistent intervention values.

        Source: RJC
        """
        event = {(Y @ [+X, -W, -Z], -Y), (W @ -X, -W), (X @ -Z, -X), (Z, -Z)}
        self.assertTrue(_any_inconsistent_intervention_values(event=event))

    def test_any_inconsistent_intervention_values_3(self):
        """Test #3 for whether a counterfactual factor has any inconsistent intervention values.

        Source: RJC
        """
        event = [(Y @ [+X, -W, -Z], -Y), (W @ +X, -W), (X @ -Z, +X), (Z, -Z)]
        self.assertFalse(_any_inconsistent_intervention_values(event=event))

    def test_any_inconsistent_intervention_values_4(self):
        """Test #4 for whether a counterfactual factor has any inconsistent intervention values.

        Source: RJC
        """
        event = [(Y @ [+X, +W, +Z], -Y), (W @ +X, -W), (X @ +Z, -X), (Z, -Z)]
        self.assertFalse(_any_inconsistent_intervention_values(event=event))

    def test_any_inconsistent_intervention_values_5(self):
        """Test #5 for whether a counterfactual factor has any inconsistent intervention values.

        This is a case that is inconsistent according to Definition 4.1(ii)
        but not Definition 4.1(i).
        Source: RJC
        """
        # graph = NxMixedGraph.from_edges(
        #    directed=[
        #        (X, Y),
        #        (X, Z),
        #    ],
        #    undirected=[(Y, Z)],
        # )
        event = [(Y @ -X, -Y), (Z @ +X, -Z)]
        self.assertTrue(_counterfactual_factor_is_inconsistent(event=event))


class TestCounterfactualFactorIsInconsistent(cases.GraphTestCase):
    """Test a check of whether a counterfactual factor is inconsistent (Definition 4.1 of [correa22a]_)."""

    def test_counterfactual_factor_is_inconsistent_1(self):
        """Test #1 for whether a counterfactual factor is inconsistent.

        Source: RJC
        """
        event = [(Y @ [-X, -W, -Z], -Y), (W @ -X, -W), (X @ -Z, +X), (Z, -Z)]
        self.assertTrue(_counterfactual_factor_is_inconsistent(event=event))

    def test_counterfactual_factor_is_inconsistent_2(self):
        """Test #2 for whether a counterfactual factor is inconsistent.

        Source: RJC
        """
        event = [(Y @ [+X, -W, -Z], -Y), (W @ -X, -W), (X @ -Z, -X), (Z, -Z)]
        self.assertTrue(_counterfactual_factor_is_inconsistent(event=event))

    def test_counterfactual_factor_is_inconsistent_3(self):
        """Test #3 for whether a counterfactual factor is inconsistent.

        Source: RJC
        """
        event = [(Y @ [+X, -W, -Z], -Y), (W @ +X, -W), (X @ -Z, +X), (Z, -Z)]
        self.assertFalse(_counterfactual_factor_is_inconsistent(event=event))

    def test_counterfactual_factor_is_inconsistent_4(self):
        """Test #4 for whether a counterfactual factor is inconsistent.

        Source: RJC
        """
        event = [(Y @ [+X, +W, +Z], -Y), (W @ +X, -W), (X @ +Z, -X), (Z, -Z)]
        self.assertTrue(_counterfactual_factor_is_inconsistent(event=event))

    def test_counterfactual_factor_is_inconsistent_5(self):
        """Test #5 for whether a counterfactual factor is inconsistent.

        This is a case that is inconsistent according to Definition 4.1(ii)
        but not Definition 4.1(i).

        Source: RJC
        """
        # graph = NxMixedGraph.from_edges(
        #    directed=[
        #        (X, Y),
        #        (X, Z),
        #    ],
        #    undirected=[(Y, Z)],
        # )
        event = [(Y @ -X, -Y), (Z @ +X, -Z)]
        self.assertTrue(_counterfactual_factor_is_inconsistent(event=event))


# TODO: x 1. We need a test case that returns a probability of Zero() if the Simplify() algorithm
#          returns a probability 0 during transport_unconditional_counterfactual_query().
#       2. We need to test _transport_unconditional_counterfactual_query_line_2() for an input
#          event containing a variable that already has a value of None. This represents input
#          values for $\mathbf{y_{\ast}}$ corresponding to variables in $\mathbf{d_{\ast}}$ but
#          not $\mathbf{y_{\ast}}$ or $\mathbf{x_{\ast}}$ processed in Algorithm 3,
#          transport_conditional_counterfactual_query.
#       3. Test a query containing a variable with a value of None and the same variable with an
#          actual value (see test cases for TransportConditionalCounterfactualQuery for the reason).
class TestTransportUnconditionalCounterfactualQuery(cases.GraphTestCase):
    """Test [correa22a]_'s unconditional counterfactual transportability algorithm (Algorithm 2)."""

    def test_transport_unconditional_counterfactual_query_1(self):
        """Test of Algorithm 2 of [correa22a]_.

        Source: Example 4.2 from [correa22]_ (Equations 15, 17, and 19).
        """
        event = [(Y @ -X, -Y), (X, -X)]
        domain_graphs = [
            (
                figure_2_graph_domain_1_with_interventions,
                figure_2_graph_domain_1_with_interventions_topo,
            ),
            (
                figure_2_graph_domain_2,
                figure_2_graph_domain_2_topo,
            ),
        ]
        domain_data = [({X}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))]
        expected_result_part_1 = Product.safe(
            [PP[Pi1](Y | W, X, Z), PP[Pi1](W | X), PP[Pi2](X | Z), PP[Pi2](Z)]
        )
        expected_result = Sum.safe(expected_result_part_1, [Z, W])

        result_expr, result_event = transport_unconditional_counterfactual_query(
            event=event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )
        logger.debug("Result_expr = " + result_expr.to_latex())
        logger.debug("Result_event = " + str(result_event))
        self.assert_expr_equal(expected_result, result_expr)

    def test_transport_unconditional_counterfactual_query_2(self):
        """Test of Algorithm 2 of [correa22a]_.

        Checking Line 3: an inconsistent counterfactual factor.
        Source: RJC's mind.
        """
        event = [(Y @ -X, -Y), (W @ +X, -W), (X, -X)]
        domain_graphs = [
            (
                figure_2_graph_domain_1_with_interventions,
                figure_2_graph_domain_1_with_interventions_topo,
            ),
            (
                figure_2_graph_domain_2,
                figure_2_graph_domain_2_topo,
            ),
        ]
        domain_data = [({X}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))]
        # expected_result_part_1 = Product.safe(
        #    [PP[Pi1](Y | W, X, Z), PP[Pi1](W | X), PP[Pi2](X | Z), PP[Pi2](Z)]
        # )
        # expected_result = Sum.safe(expected_result_part_1, [Z, W])

        result = transport_unconditional_counterfactual_query(
            event=event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )
        self.assertIsNone(result)
        # self.assert_expr_equal(expected_result, result)

    def test_transport_unconditional_counterfactual_query_3(self):
        """Test of Algorithm 2 of [correa22a]_.

        Checking that the algorithm properly processes variables with values of None.

        Source: Example 4.2 from [correa22]_ (Equations 15, 17, and 19).
        """
        event = [(Y @ -X, -Y), (X, -X)]
        domain_graphs = [
            (
                figure_2_graph_domain_1_with_interventions,
                figure_2_graph_domain_1_with_interventions_topo,
            ),
            (
                figure_2_graph_domain_2,
                figure_2_graph_domain_2_topo,
            ),
        ]
        domain_data = [({X}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))]
        expected_result_part_1 = Product.safe(
            [PP[Pi1](Y | W, X, Z), PP[Pi1](W | X), PP[Pi2](X | Z), PP[Pi2](Z)]
        )
        expected_result = Sum.safe(expected_result_part_1, [Z, W])
        result_expr, result_event = transport_unconditional_counterfactual_query(
            event=event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )
        logger.debug("Result_expr = " + result_expr.to_latex())
        logger.debug("Result_event = " + str(result_event))
        self.assert_expr_equal(expected_result, result_expr)
        self.assertCountEqual(event, result_event)
        # Test sending variables with a value of None into this algorithm
        event_2 = [(Y @ -X, None), (X, -X)]
        event_3 = [(Y @ -X, -Y), (X, None)]
        event_4 = [(Y @ -X, -Y), (Y @ -X, None), (X, -X)]
        result_expr_2, result_event_2 = transport_unconditional_counterfactual_query(
            event=event_2,
            target_domain_graph=figure_2a_graph,
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )
        result_expr_3, result_event_3 = transport_unconditional_counterfactual_query(
            event=event_3,
            target_domain_graph=figure_2a_graph,
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )
        result_expr_4, result_event_4 = transport_unconditional_counterfactual_query(
            event=event_4,
            target_domain_graph=figure_2a_graph,
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )
        self.assert_expr_equal(expected_result, result_expr_2)
        self.assertCountEqual(event_2, result_event_2)
        self.assert_expr_equal(expected_result, result_expr_3)
        self.assertCountEqual(event_3, result_event_3)
        self.assert_expr_equal(expected_result, result_expr_4)
        # Simplify() should drop the redundant query variable that has a None value
        self.assertCountEqual(event, result_event_4)

    def test_transport_unconditional_counterfactual_query_line_2_1(self):
        """Test of Line 2 of Algorithm 2 of [correa22a]_.

        Source: Example 4.2 from [correa22]_ (Equations 15, 17, and 19).
        """
        event = [(Y @ -X, -Y), (X, -X)]
        target_domain_graph = figure_2a_graph
        expected_ancestors_with_values = {(Y @ -X, -Y), (W @ -X, None), (X, -X), (Z, None)}
        expected_counterfactual_factors_with_values = [
            {(Y @ [-X, -W, -Z], -Y), (W @ -X, None)},
            {(X @ -Z, -X), (Z, None)},
        ]
        (
            outcome_ancestors_with_values,
            counterfactual_factors_with_values,
        ) = _transport_unconditional_counterfactual_query_line_2(event, target_domain_graph)
        self.assertSetEqual(expected_ancestors_with_values, outcome_ancestors_with_values)
        self.assertCountEqual(
            expected_counterfactual_factors_with_values, counterfactual_factors_with_values
        )

    def test_transport_unconditional_counterfactual_query_line_2_2(self):
        """Second test of Line 2 of Algorithm 2 of [correa22a]_.

        This tests whether we properly get W @-X1 and W @ -X2 as two separate ancestors, and then
        merge them into one counterfactual factor W @ [-X1, -X2].

        Source: Example 4.2 from [correa22]_ (Equations 15, 17, and 19).
        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (X1, W),
                (X1, Y),
                (X2, W),
                (W, Y),
                (W, Z),
            ],
        )

        event = [(Y @ -X1, -Y), (Z @ -X2, -Z)]
        target_domain_graph = graph
        expected_ancestors_with_values = {
            (Y @ -X1, -Y),
            (W @ -X1, None),
            (W @ -X2, None),
            (Z @ -X2, -Z),
            (X1, None),
            (X2, None),
        }
        expected_counterfactual_factors_with_values = [
            {(Y @ [-X1, -W], -Y)},
            {(W @ [-X1, -X2], None)},
            {(Z @ -W, -Z)},
            {(X1, None)},
            {(X2, None)},
        ]
        (
            outcome_ancestors_with_values,
            counterfactual_factors_with_values,
        ) = _transport_unconditional_counterfactual_query_line_2(event, target_domain_graph)
        self.assertSetEqual(expected_ancestors_with_values, outcome_ancestors_with_values)
        self.assertCountEqual(
            expected_counterfactual_factors_with_values, counterfactual_factors_with_values
        )

    def test_transport_unconditional_counterfactual_query_simplify_returns_none(self):
        """Test of Line 1 of Algorithm 2 of [correa22a]_.

        This tests whether we properly return (Zero(), None) if a query, when simplified,
        returns a probability of zero.

        Source: Example 4.2 from [correa22]_, modified to make the simplification fail.
        """
        # 1. We need a test case that returns a probability of Zero() if the Simplify() algorithm
        #   returns a probability 0 during transport_unconditional_counterfactual_query().
        event = [(Y @ -Y, -Y), (Y, +Y)]
        # event = [(Y @ -X, -Y), (X, -X)]
        domain_graphs = [
            (
                figure_2_graph_domain_1_with_interventions,
                figure_2_graph_domain_1_with_interventions_topo,
            ),
            (
                figure_2_graph_domain_2,
                figure_2_graph_domain_2_topo,
            ),
        ]
        domain_data = [({X}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))]
        result_expr, result_event = transport_unconditional_counterfactual_query(
            event=event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )
        self.assert_expr_equal(result_expr, Zero())
        self.assertIsNone(result_event)

    def test_transport_unconditional_counterfactual_query_line_2_3(self):
        """Third test of Line 2 of Algorithm 2 of [correa22a]_.

        This tests whether we properly get W @ -X1 and W @ -X2 from two different variables but
        represent it as one ancestor, which we then merge into one counterfactual factor W @ -X2.

        Source: Example 4.2 from [correa22]_ (Equations 15, 17, and 19).
        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (X1, W),
                (X1, Y),
                (X2, W),
                (W, Y),
                (W, Z),
            ],
        )

        event = [(Y @ -X1, -Y), (Z @ -X2, -Z)]
        target_domain_graph = graph
        expected_ancestors_with_values = {
            (Y @ -X1, -Y),
            (W @ -X1, None),
            (W @ -X2, None),
            (Z @ -X2, -Z),
            (X1, None),
            (X2, None),
        }
        expected_counterfactual_factors_with_values = [
            {(Y @ [-X1, -W], -Y)},
            {(W @ [-X1, -X2], None)},
            {(Z @ -W, -Z)},
            {(X1, None)},
            {(X2, None)},
        ]
        (
            outcome_ancestors_with_values,
            counterfactual_factors_with_values,
        ) = _transport_unconditional_counterfactual_query_line_2(event, target_domain_graph)
        self.assertSetEqual(expected_ancestors_with_values, outcome_ancestors_with_values)
        self.assertCountEqual(
            expected_counterfactual_factors_with_values, counterfactual_factors_with_values
        )

    def test_transport_unconditional_counterfactual_query_line_2_4(self):
        """Fourth test of Line 2 of Algorithm 2 of [correa22a]_.

        This tests if the function properly parses input counterfactual variables with
        values of None.

        Source: Example 4.2 from [correa22]_ (Equations 15, 17, and 19).
        """
        event = [(Y @ -X, None), (X, -X)]
        target_domain_graph = figure_2a_graph
        expected_ancestors_with_values = {(Y @ -X, None), (W @ -X, None), (X, -X), (Z, None)}
        expected_counterfactual_factors_with_values = [
            {(Y @ [-X, -W, -Z], None), (W @ -X, None)},
            {(X @ -Z, -X), (Z, None)},
        ]
        (
            outcome_ancestors_with_values,
            counterfactual_factors_with_values,
        ) = _transport_unconditional_counterfactual_query_line_2(event, target_domain_graph)
        self.assertSetEqual(expected_ancestors_with_values, outcome_ancestors_with_values)
        self.assertCountEqual(
            expected_counterfactual_factors_with_values, counterfactual_factors_with_values
        )

    def test_transport_unconditional_counterfactual_query_line_2_5(self):
        """Fifth test of Line 2 of Algorithm 2 of [correa22a]_.

        This tests if the function properly parses input variables with
        values of None.

        Source: Example 4.2 from [correa22]_ (Equations 15, 17, and 19).
        """
        event = [(Y @ -X, -Y), (X, None)]
        target_domain_graph = figure_2a_graph
        expected_ancestors_with_values = {(Y @ -X, -Y), (W @ -X, None), (X, None), (Z, None)}
        expected_counterfactual_factors_with_values = [
            {(Y @ [-X, -W, -Z], -Y), (W @ -X, None)},
            {(X @ -Z, None), (Z, None)},
        ]
        (
            outcome_ancestors_with_values,
            counterfactual_factors_with_values,
        ) = _transport_unconditional_counterfactual_query_line_2(event, target_domain_graph)
        self.assertSetEqual(expected_ancestors_with_values, outcome_ancestors_with_values)
        self.assertCountEqual(
            expected_counterfactual_factors_with_values, counterfactual_factors_with_values
        )

    def test_transport_unconditional_counterfactual_query_line_5(self):
        """Test of Line 5 of Algorithm 2 of [correa22a]_.

        This test checks that if a counterfactual factor can't be transported, the algorithm
        returns None.

        Source: RJC's mind.
        """
        graph_1 = NxMixedGraph.from_edges(
            directed=[(Z, X), (X, Y), (Z, Y), (W, Y), (transport_variable(Y), Y)],
            undirected=[
                (Z, X),
            ],
        )
        graph_1_topo = list(graph_1.topological_sort())
        # Let's say graph 2 is a stochastic intervention where Y is a function of W and Z,
        # but not X.
        graph_2 = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (X, W),
                (Z, Y),
                (W, Y),
            ],
            undirected=[],
        )
        graph_2_topo = list(graph_2.topological_sort())
        domain_data = [(set(), PP[Pi1](W, X, Y, Z)), ({Y}, PP[Pi2](W, X, Y, Z))]
        query_result = transport_unconditional_counterfactual_query(
            event=[(Y @ -X, -Y), (X, -X)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=[(graph_1, graph_1_topo), (graph_2, graph_2_topo)],
            domain_data=domain_data,
        )
        self.assertIsNone(query_result)


class TestGetConditionedVariablesInAncestralSet(cases.GraphTestCase):
    r"""Identify conditioned variables that are ancestors of an input variable ($\mathbf{X_{\ast}}(W_{\mathbf{\ast}})$).

    Note: see the documentation for get_conditioned_variables_in_ancestral_set for a note about
    some ambiguity in the mathematical definition of that function in [correa22a]_.
    """

    def test_get_conditioned_variable_in_ancestral_set_1(self):
        """First test of the function to identify conditioned variables that are ancestors of an input variable.

        Source: Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_1 = {Z}
        result_1 = _get_conditioned_variables_in_ancestral_set(
            conditioned_variables={Z @ -X, X},
            ancestral_set_root_variable=Y @ -X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_1, result_1)

    def test_get_conditioned_variable_in_ancestral_set_2(self):
        """Second test of the function to identify conditioned variables that are ancestors of an input variable.

        Source: Slight modification of Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_2 = {Z}
        result_2 = _get_conditioned_variables_in_ancestral_set(
            conditioned_variables={Z @ -X, X},
            ancestral_set_root_variable=Z @ -X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_2, result_2)

    def test_get_conditioned_variable_in_ancestral_set_3(self):
        """Third test of the function to identify conditioned variables that are ancestors of an input variable.

        Source: Slight modification of Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_3 = {X}
        result_3 = _get_conditioned_variables_in_ancestral_set(
            conditioned_variables={Z @ -X, X},
            ancestral_set_root_variable=X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_3, result_3)

    def test_get_conditioned_variable_in_ancestral_set_4(self):
        """Fourth test of the function to identify conditioned variables that are ancestors of an input variable.

        Source: Slight modification of Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_4 = set()
        result_4 = _get_conditioned_variables_in_ancestral_set(
            conditioned_variables={Z @ -X},
            ancestral_set_root_variable=X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_4, result_4)

    def test_get_conditioned_variable_in_ancestral_set_5(self):
        """Fourth test of the function to identify conditioned variables that are ancestors of an input variable.

        Source: Slight modification of Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_5 = set()
        result_5 = _get_conditioned_variables_in_ancestral_set(
            conditioned_variables={X},
            ancestral_set_root_variable=Z @ -X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_5, result_5)

    def test_get_conditioned_variable_in_ancestral_set_6(self):
        """Fourth test of the function to identify conditioned variables that are ancestors of an input variable.

        Source: Slight modification of Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_6 = {X}
        result_6 = _get_conditioned_variables_in_ancestral_set(
            conditioned_variables={X},
            ancestral_set_root_variable=X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_6, result_6)

    def test_get_conditioned_variable_in_ancestral_set_7(self):
        """Fourth test of the function to identify conditioned variables that are ancestors of an input variable.

        Source: Slight modification of Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_7 = set()
        result_7 = _get_conditioned_variables_in_ancestral_set(
            conditioned_variables={X},
            ancestral_set_root_variable=Y @ -X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_7, result_7)


class TestGetAncestralSetAfterInterveningOnConditionedVariables(cases.GraphTestCase):
    """Intervene on conditioned variables in a variable's ancestral set and recompute the ancestral set."""

    def test_get_ancestral_set_after_intervening_on_conditioned_variables_1(self):
        """First test of the function to get an ancestral set after intervening on some conditioned variables.

        Note that we only intervene on the conditioned variables that are ancestors of the input root variable.

        Source: Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_1 = frozenset({Y @ -X})
        result_1 = _get_ancestral_set_after_intervening_on_conditioned_variables(
            conditioned_variables={Z @ -X, X},
            ancestral_set_root_variable=Y @ -X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_1, result_1)

    def test_get_ancestral_set_after_intervening_on_conditioned_variables_2(self):
        """Second test of the function to get an ancestral set after intervening on some conditioned variables.

        Note that we only intervene on the conditioned variables that are ancestors of the input root variable.

        Source: Slight modification of Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_2 = frozenset({Z @ -X})
        result_2 = _get_ancestral_set_after_intervening_on_conditioned_variables(
            conditioned_variables={Z @ -X, X},
            ancestral_set_root_variable=Z @ -X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_2, result_2)

    def test_get_ancestral_set_after_intervening_on_conditioned_variables_3(self):
        """Third test of the function to get an ancestral set after intervening on some conditioned variables.

        Note that we only intervene on the conditioned variables that are ancestors of the input root variable.

        Source: Slight modification of Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_3 = frozenset({X})
        result_3 = _get_ancestral_set_after_intervening_on_conditioned_variables(
            conditioned_variables={Z @ -X, X},
            ancestral_set_root_variable=X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_3, result_3)

    def test_get_ancestral_set_after_intervening_on_conditioned_variables_4(self):
        """Fourth test of the function to get an ancestral set after intervening on some conditioned variables.

        Note that we only intervene on the conditioned variables that are ancestors of the input root variable.

        Source: Slight modification of Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_4 = frozenset({Z @ -X})
        result_4 = _get_ancestral_set_after_intervening_on_conditioned_variables(
            conditioned_variables={Z @ -X},
            ancestral_set_root_variable=Z @ -X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_4, result_4)

    def test_get_ancestral_set_after_intervening_on_conditioned_variables_5(self):
        """Fifth test of the function to get an ancestral set after intervening on some conditioned variables.

        Note that we only intervene on the conditioned variables that are ancestors of the input root variable.

        Source: Slight modification of Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_5 = frozenset({Y @ -X, Z @ -X})
        result_5 = _get_ancestral_set_after_intervening_on_conditioned_variables(
            conditioned_variables={X},
            ancestral_set_root_variable=Y @ -X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_5, result_5)

    def test_get_ancestral_set_after_intervening_on_conditioned_variables_6(self):
        """Sixth test of the function to get an ancestral set after intervening on some conditioned variables.

        Note that we only intervene on the conditioned variables that are ancestors of the input root variable.

        Source: Slight modification of Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_6 = frozenset({X})
        result_6 = _get_ancestral_set_after_intervening_on_conditioned_variables(
            conditioned_variables={Z @ -X},
            ancestral_set_root_variable=X,
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_6, result_6)


# TODO: Add additional tests after checking code coverage.
class TestComputeAncestralComponentsFromAncestralSets(cases.GraphTestCase):
    """Test a function to combine ancestral sets if they share vertices or are joined by bidirected edges."""

    def test_compute_ancestral_components_from_ancestral_sets_1(self):
        """First test of a function to combine ancestral sets if they share vertices or are joined by bidirected edges.

        Source: Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_1 = frozenset({frozenset({Y @ -X}), frozenset({Z @ -X, X})})
        result_1 = _compute_ancestral_components_from_ancestral_sets(
            ancestral_sets=frozenset({frozenset({X}), frozenset({Z @ -X}), frozenset({Y @ -X})}),
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_1, result_1)

    def test_compute_ancestral_components_from_ancestral_sets_2(self):
        """Second test of a function to combine ancestral sets if they share vertices or are joined by bidirected edges.

        Source: RJC, using the graph for Figure 2a of [correa22a]_.
        """
        expected_result_2 = frozenset({frozenset({X, Z}), frozenset({W, Y})})
        result_2 = _compute_ancestral_components_from_ancestral_sets(
            ancestral_sets=frozenset(
                {frozenset({W}), frozenset({X}), frozenset({Y}), frozenset({Z})}
            ),
            graph=figure_2a_graph,
        )
        self.assertSetEqual(expected_result_2, result_2)

    def test_compute_ancestral_components_from_ancestral_sets_3(self):
        """Third test of a function to combine ancestral sets if they share vertices or are joined by bidirected edges.

        Source: RJC, using the graph for Figure 2a of [correa22a]_.
        """
        expected_result_3 = frozenset({frozenset({W, X, Y, Z})})
        result_3 = _compute_ancestral_components_from_ancestral_sets(
            ancestral_sets=frozenset({frozenset({W, X}), frozenset({Y}), frozenset({Z})}),
            graph=figure_2a_graph,
        )
        self.assertSetEqual(expected_result_3, result_3)

    def test_compute_ancestral_components_from_ancestral_sets_4(self):
        """Fourth test of a function to combine ancestral sets if they share vertices or are joined by bidirected edges.

        Source: RJC, based the graph for Figure 2a of [correa22a]_ with no bidirectional edges.
        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (Z, Y),
                (X, Y),
                (X, W),
                (W, Y),
            ],
        )
        expected_result_4 = frozenset({frozenset({W, X, Y}), frozenset({Z})})
        result_4 = _compute_ancestral_components_from_ancestral_sets(
            ancestral_sets=frozenset({frozenset({W, X}), frozenset({X, Y}), frozenset({Z})}),
            graph=graph,
        )
        self.assertSetEqual(expected_result_4, result_4)

    def test_compute_ancestral_components_from_ancestral_sets_5(self):
        """Fifth test of a function to combine ancestral sets if they share vertices or are joined by bidirected edges.

        Source: Example 1.1 of [correa22a]_, modified to include ancestral sets that are not disjoint and
           bidirected edges joining two ancestral sets.
        """
        expected_result_5 = frozenset({frozenset({Y @ -X, Y, Z @ -X, X})})
        result_5 = _compute_ancestral_components_from_ancestral_sets(
            ancestral_sets=frozenset({frozenset({X, Y}), frozenset({Z @ -X}), frozenset({Y @ -X})}),
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_5, result_5)

    def test_compute_ancestral_components_from_ancestral_sets_6(self):
        """Sixth test of a function to combine ancestral sets if they share vertices or are joined by bidirected edges.

        Source: Example 1.1 of [correa22a]_, modified to include ancestral sets that are not disjoint and
           bidirected edges joining two ancestral sets.
        """
        expected_result_6 = frozenset({frozenset({Y @ -X, Y @ -Z, Z @ -X, X})})
        result_6 = _compute_ancestral_components_from_ancestral_sets(
            ancestral_sets=frozenset(
                {frozenset({X, Y @ -Z}), frozenset({Z @ -X}), frozenset({Y @ -X})}
            ),
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_6, result_6)


# TODO: Add additional tests covering more complex scenarios.
class TestGetAncestralComponents(cases.GraphTestCase):
    """Test a function to compute ancestral components given a set of vertices and conditioned variables."""

    def test_get_ancestral_components_1(self):
        """First test of a function to compute ancestral components for a graph.

        Source: Example 4.5 and Figure 6 of [correa22a]_.
        """
        expected_result_1 = frozenset({frozenset({Y @ -X}), frozenset({Z @ -X, X})})
        result_1 = get_ancestral_components(
            conditioned_variables=frozenset({X, Z @ -X}),
            root_variables=frozenset({Y @ -X, Z @ -X, X}),
            graph=figure_1_graph_no_transportability_nodes,
        )
        self.assertSetEqual(expected_result_1, result_1)


# TODO: 1. We need a test case that returns a probability of Zero() if the Simplify() algorithm
#          returns a probability 0 during transport_unconditional_counterfactual_query().
#       2. Test a case that sends a query to TransportUnconditionalCounterfactualQuery containing
#          a variable with a value of None and the same variable with an actual value, due to its
#          status as both an outcome variable and an ancestor of an outcome variable.
class TestTransportConditionalCounterfactualQuery(cases.GraphTestCase):
    """Test a function to transport a conditional counterfactual query (Algorithm 3 of [correa22a]_)."""

    @classmethod
    def setUpClass(cls):
        """Set up the class."""
        cls.example_1_outcomes = [(Y @ -X, -Y)]
        cls.example_1_conditions = [(Z @ -X, -Z), (X, +X)]
        cls.example_1_target_domain_graph = figure_1_graph_no_transportability_nodes
        cls.example_1_domain_graphs = [
            (
                figure_1_graph_no_transportability_nodes,
                figure_1_graph_no_transportability_nodes_topo,
            ),
            (
                figure_1_graph_domain_1_with_interventions,
                figure_1_graph_domain_1_with_interventions_topo,
            ),
        ]
        cls.example_2_outcomes = [(Y @ -X1, -Y), (W @ -X2, -W)]
        cls.example_2_conditions = [(X1, -X1)]
        cls.example_2_target_domain_graph = NxMixedGraph.from_edges(
            directed=[
                (X1, Z),
                (X2, Z),
                (Z, W),
                (W, Y),
            ],
            undirected=[(Z, W)],
        )
        cls.example_2_target_domain_graph_topo = (
            cls.example_2_target_domain_graph.topological_sort()
        )
        cls.example_2_domain_1_graph = NxMixedGraph.from_edges(
            directed=[(X1, Z), (X2, Z), (Z, W), (W, Y), (transport_variable(X2), X2)],
            undirected=[],
        )
        cls.example_2_domain_1_graph_topo = cls.example_2_domain_1_graph.topological_sort()
        cls.example_2_domain_graphs = [
            (
                cls.example_2_target_domain_graph,
                cls.example_2_target_domain_graph_topo,
            ),
            (
                cls.example_2_domain_1_graph,
                cls.example_2_domain_1_graph_topo,
            ),
        ]

    def test_transport_conditional_counterfactual_query_1(self):
        """First test of Algorithm 3 of [correa22a], transporting a conditional counterfactual query.

        Source: Example 4.5 and Figure 6 of [correa22a]_.
        """
        # DSL isn't smart enough to replace the denominator with 1
        domain_data = [(set(), PP[TARGET_DOMAIN](X, Y, Z)), ({X}, PP[Pi1](X, Y, Z))]
        expected_result_expr, expected_result_event = (
            Fraction(PP[TARGET_DOMAIN](Y | X, Z), Sum.safe(PP[TARGET_DOMAIN](Y | X, Z), {Y})),
            [(Y, -Y), (X, +X), (Z, -Z)],
        )
        result_expr, result_event = transport_conditional_counterfactual_query(
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=domain_data,
        )
        logger.debug("expected_result_expr = " + expected_result_expr.to_latex())
        logger.debug("expected_result_event = " + str(expected_result_event))
        logger.debug("Result_expr = " + result_expr.to_latex())
        logger.debug("Result_event = " + str(result_event))
        self.assert_expr_equal(expected_result_expr, result_expr)
        self.assertCountEqual(expected_result_event, result_event)

    def test_transport_conditional_counterfactual_query_2(self):
        """Second test of Algorithm 3 of [correa22a], transporting a conditional counterfactual query.

        Source: RC's mind. Designed to test merging outcome ancestral sets into ancestral components.
        """
        # DSL isn't smart enough to replace the denominator with 1
        # @cthoyt: The code base would be simpler were we to instantiate these as static class-level objects,
        # but then we get the following MyPy error:
        # error: Value of type "type[PopulationProbabilityBuilderType]" is not indexable
        expected_result_expr, expected_result_event = (
            Fraction(
                Sum.safe(
                    Product.safe(
                        [
                            PP[TARGET_DOMAIN](Y | W, Z, X2, X1),
                            PP[TARGET_DOMAIN](W | Z, X2, X1),
                            PP[TARGET_DOMAIN](Z | X2, X1),
                            PP[TARGET_DOMAIN](X2 | X1),
                        ],
                    ),
                    {Z, X2},
                ),
                Sum.safe(
                    Product.safe(
                        [
                            PP[TARGET_DOMAIN](Y | W, Z, X2, X1),
                            PP[TARGET_DOMAIN](W | Z, X2, X1),
                            PP[TARGET_DOMAIN](Z | X2, X1),
                            PP[TARGET_DOMAIN](X2 | X1),
                        ],
                    ),
                    {Z, X2, Y, W},
                ),
            ),
            [(Y, -Y), (W, -W), (X1, -X1)],
        )
        domain_data = [
            (set(), PP[TARGET_DOMAIN](X1, X2, W, Y, Z)),
            ({W}, PP[Pi1](X1, X2, W, Y, Z)),
        ]
        result_expr, result_event = transport_conditional_counterfactual_query(
            outcomes=self.example_2_outcomes,
            conditions=self.example_2_conditions,
            target_domain_graph=self.example_2_target_domain_graph,
            domain_graphs=self.example_2_domain_graphs,
            domain_data=domain_data,
        )
        logger.debug("expected_result_expr = " + expected_result_expr.to_latex())
        logger.debug("expected_result_event = " + str(expected_result_event))
        logger.debug("Result_expr = " + result_expr.to_latex())
        logger.debug("Result_event = " + str(result_event))
        self.assert_expr_equal(expected_result_expr, result_expr)
        self.assertCountEqual(expected_result_event, result_event)

    def test_transport_conditional_counterfactual_query_subroutines_1(self):
        """Test _initialize_conditional_transportability_data_structures().

        Source: RC's mind.
        """
        outcomes = [(Y @ -X1, -Y), (W @ -X2, -W)]
        conditions = [(X1, -X1), (X2, -X2)]
        (
            conditioned_variables,
            outcome_variables,
            outcome_and_conditioned_variables,
            outcome_variable_to_value_mappings,
            outcome_and_conditioned_variable_names_to_values,
            outcome_and_conditioned_variable_names,
            conditioned_variable_names,
        ) = _initialize_conditional_transportability_data_structures(
            outcomes=outcomes, conditions=conditions
        )
        self.assertSetEqual(conditioned_variables, {X1, X2})
        self.assertSetEqual(outcome_variables, {Y @ -X1, W @ -X2})
        self.assertSetEqual(outcome_and_conditioned_variables, {X1, X2, Y @ -X1, W @ -X2})
        self.assertDictEqual(outcome_variable_to_value_mappings, {Y @ -X1: {-Y}, W @ -X2: {-W}})
        self.assertDictEqual(
            outcome_and_conditioned_variable_names_to_values,
            {X1: {-X1}, X2: {-X2}, Y: {-Y}, W: {-W}},
        )
        self.assertSetEqual(outcome_and_conditioned_variable_names, {X1, X2, W, Y})
        self.assertSetEqual(conditioned_variable_names, {X1, X2})

    def test_transport_conditional_counterfactual_query_subroutines_2(self):
        """Test _transport_conditional_counterfactual_query_line_2().

        Source: RC's mind.
        """
        ancestral_components = frozenset(
            {frozenset({Y @ -X1, W @ -X1, Z @ -X1, X2, W @ -X2, Z @ -X2}), frozenset({X1})}
        )
        outcome_variables = {Y @ -X1, W @ -X2}
        outcome_variable_to_value_mappings = {Y @ -X1: {-Y}, W @ -X2: {-W}}
        expected_outcome_ancestral_component_query_in_counterfactual_factor_form = [
            (Y @ -W, -Y),
            (W @ -Z, -W),
            (W @ -Z, None),
            (Z @ -X1 @ -X2, None),
            (Z @ -X1 @ -X2, None),
            (X2, None),
        ]
        expected_outcome_variable_ancestral_component_variable_names = {Y, W, Z, X2}
        (
            outcome_ancestral_component_query_in_counterfactual_factor_form,
            outcome_variable_ancestral_component_variable_names,
        ) = _transport_conditional_counterfactual_query_line_2(
            ancestral_components=ancestral_components,
            outcome_variables=outcome_variables,
            outcome_variable_to_value_mappings=outcome_variable_to_value_mappings,
            target_domain_graph=self.example_2_target_domain_graph,
        )
        self.assertCountEqual(
            expected_outcome_ancestral_component_query_in_counterfactual_factor_form,
            outcome_ancestral_component_query_in_counterfactual_factor_form,
        )
        self.assertSetEqual(
            outcome_variable_ancestral_component_variable_names,
            expected_outcome_variable_ancestral_component_variable_names,
        )

    def test_transport_conditional_counterfactual_query_subroutines_3(self):
        """Test _transport_conditional_counterfactual_query_line_4().

        Source: RC's mind.
        """
        outcomes = [(Y @ -X1, -Y), (W @ -X2, -W)]
        conditions = [(X1, -X1)]
        outcome_variable_ancestral_component_variable_names = {Y, W, Z, X2}
        outcome_and_conditioned_variable_names = {Y, W, X1}
        conditioned_variable_names = {X1}
        transported_unconditional_query_expression = Product.safe(
            [
                PP[TARGET_DOMAIN](Y | W, Z, X2, X1),
                PP[TARGET_DOMAIN](W | Z, X2, X1),
                PP[TARGET_DOMAIN](Z | X2, X1),
                PP[TARGET_DOMAIN](X2 | X1),
            ],
        )
        simplified_event = [
            (Y @ -W, -Y),
            (W @ -Z, -W),
            (Z @ -X1 @ -X2, None),
            (X2, None),
        ]
        # @cthoyt: The code base would be simpler were we to instantiate these as static class-level objects,
        # but then we get the following MyPy error:
        # error: Value of type "type[PopulationProbabilityBuilderType]" is not indexable
        expected_result_expr, expected_result_event = (
            Fraction(
                Sum.safe(
                    Product.safe(
                        [
                            PP[TARGET_DOMAIN](Y | W, Z, X2, X1),
                            PP[TARGET_DOMAIN](W | Z, X2, X1),
                            PP[TARGET_DOMAIN](Z | X2, X1),
                            PP[TARGET_DOMAIN](X2 | X1),
                        ],
                    ),
                    {Z, X2},
                ),
                Sum.safe(
                    Product.safe(
                        [
                            PP[TARGET_DOMAIN](Y | W, Z, X2, X1),
                            PP[TARGET_DOMAIN](W | Z, X2, X1),
                            PP[TARGET_DOMAIN](Z | X2, X1),
                            PP[TARGET_DOMAIN](X2 | X1),
                        ],
                    ),
                    {Z, X2, Y, W},
                ),
            ),
            [(Y, -Y), (W, -W), (X1, -X1)],
        )
        domain_data = [
            (set(), PP[TARGET_DOMAIN](X1, X2, W, Y, Z)),
            ({W}, PP[Pi1](X1, X2, W, Y, Z)),
        ]
        outcome_and_conditioned_variable_names_to_values = {X1: {-X1}, Y: {-Y}, W: {-W}}
        result_expression, result_event = _transport_conditional_counterfactual_query_line_4(
            outcome_variable_ancestral_component_variable_names=outcome_variable_ancestral_component_variable_names,
            outcome_and_conditioned_variable_names=outcome_and_conditioned_variable_names,
            conditioned_variable_names=conditioned_variable_names,
            transported_unconditional_query_expression=transported_unconditional_query_expression,
            simplified_event=simplified_event,
            outcome_and_conditioned_variable_names_to_values=outcome_and_conditioned_variable_names_to_values,
            outcomes=outcomes,
            conditions=conditions,
            domain_data=domain_data,
        )
        self.assert_expr_equal(result_expression, expected_result_expr)
        self.assertCountEqual(result_event, expected_result_event)

    def test_transport_conditional_counterfactual_query_subroutines_4(self):
        """Test _validate_transport_conditional_counterfactual_query_line_4_output().

        Source: RC's mind.
        """
        simplified_event = [
            (Y @ -W, -Y),
            (W @ -Z, -W),
            (Z @ -X1 @ -X2, None),
            (X2, None),
        ]
        outcome_and_conditioned_variable_names = {Y, W, X1}
        outcome_and_conditioned_variable_names_to_values = {X1: {-X1}, Y: {-Y}, W: {-W}}
        outcome_ancestral_component_variables_with_no_values = {Z, X2}  # {Y, W, Z, X2} - {Y, W, X1}
        result_expression, result_event = (
            Fraction(
                Sum.safe(
                    Product.safe(
                        [
                            PP[TARGET_DOMAIN](Y | W, Z, X2, X1),
                            PP[TARGET_DOMAIN](W | Z, X2, X1),
                            PP[TARGET_DOMAIN](Z | X2, X1),
                            PP[TARGET_DOMAIN](X2 | X1),
                        ],
                    ),
                    {Z, X2},
                ),
                Sum.safe(
                    Product.safe(
                        [
                            PP[TARGET_DOMAIN](Y | W, Z, X2, X1),
                            PP[TARGET_DOMAIN](W | Z, X2, X1),
                            PP[TARGET_DOMAIN](Z | X2, X1),
                            PP[TARGET_DOMAIN](X2 | X1),
                        ],
                    ),
                    {Z, X2, Y, W},
                ),
            ),
            [(Y, -Y), (W, -W), (X1, -X1)],
        )
        domain_data = [
            (set(), PP[TARGET_DOMAIN](X1, X2, W, Y, Z)),
            ({W}, PP[Pi1](X1, X2, W, Y, Z)),
        ]
        # 1. Make sure in the simplified event we got back, all of the variables
        #    that have values are either variables in the outcomes or variables
        #    in the conditions.
        self.assertRaises(
            KeyError,
            _validate_transport_conditional_counterfactual_query_line_4_output,
            simplified_event=[
                (Y @ -W, -Y),
                (W @ -Z, -W),
                (Z @ -X1 @ -X2, -Z),  # Testing the first check
                (X2, None),
            ],
            outcome_and_conditioned_variable_names=outcome_and_conditioned_variable_names,
            outcome_and_conditioned_variable_names_to_values=outcome_and_conditioned_variable_names_to_values,
            outcome_ancestral_component_variables_with_no_values=outcome_ancestral_component_variables_with_no_values,
            result_expression=result_expression,
            result_event=result_event,
            domain_data=domain_data,
        )
        # 2. Make sure the values for those variables in the simplified event match
        #    the values for their corresponding outcome or condition variables
        #    in the input for this function.
        self.assertRaises(
            KeyError,
            _validate_transport_conditional_counterfactual_query_line_4_output,
            simplified_event=[
                (Y @ -W, +Y),  # Testing the second check
                (W @ -Z, -W),
                (Z @ -X1 @ -X2, None),
                (X2, None),
            ],
            outcome_and_conditioned_variable_names=outcome_and_conditioned_variable_names,
            outcome_and_conditioned_variable_names_to_values=outcome_and_conditioned_variable_names_to_values,
            outcome_ancestral_component_variables_with_no_values=outcome_ancestral_component_variables_with_no_values,
            result_expression=result_expression,
            result_event=result_event,
            domain_data=domain_data,
        )
        # 3. Make sure all the variables in the expression this function will return are either
        #    in the outcomes, the conditions, one of the expressions passed in with the
        #    domain data, or the outcome ancestral component variables excluding outcomes and conditions.
        #    See also test_transport_conditional_counterfactual_query_5 below.
        self.assertRaises(
            KeyError,
            _validate_transport_conditional_counterfactual_query_line_4_output,
            simplified_event=simplified_event,
            outcome_and_conditioned_variable_names=outcome_and_conditioned_variable_names,
            outcome_and_conditioned_variable_names_to_values=outcome_and_conditioned_variable_names_to_values,
            outcome_ancestral_component_variables_with_no_values=outcome_ancestral_component_variables_with_no_values,
            result_expression=Fraction(
                Sum.safe(
                    Product.safe(
                        [
                            PP[TARGET_DOMAIN](Y | W, Z, X2, X1),
                            PP[TARGET_DOMAIN](W | Z, X2, X1),
                            PP[TARGET_DOMAIN](Z | X2, X1),
                            PP[TARGET_DOMAIN](X2 | X1),
                        ],
                    ),
                    {Z, X2},
                ),
                Sum.safe(
                    Product.safe(
                        [
                            PP[TARGET_DOMAIN](Y | W, Z, X2, X1),
                            PP[TARGET_DOMAIN](W | Z, X2, X1),
                            PP[TARGET_DOMAIN](Z | X2, X1),
                            PP[TARGET_DOMAIN](X2 | X1, R),  # Testing the third check
                        ],
                    ),
                    {Z, X2, Y, W},
                ),
            ),
            result_event=result_event,
            domain_data=domain_data,
        )
        self.assertRaises(
            KeyError,
            _validate_transport_conditional_counterfactual_query_line_4_output,
            simplified_event=simplified_event,
            outcome_and_conditioned_variable_names=outcome_and_conditioned_variable_names,
            outcome_and_conditioned_variable_names_to_values=outcome_and_conditioned_variable_names_to_values,
            outcome_ancestral_component_variables_with_no_values=outcome_ancestral_component_variables_with_no_values,
            result_expression=Fraction(
                Sum.safe(
                    Product.safe(
                        [
                            PP[TARGET_DOMAIN](Y, R | W, Z, X2, X1),  # Testing the third check
                            PP[TARGET_DOMAIN](W | Z, X2, X1),
                            PP[TARGET_DOMAIN](Z | X2, X1),
                            PP[TARGET_DOMAIN](X2 | X1),
                        ],
                    ),
                    {Z, X2},
                ),
                Sum.safe(
                    Product.safe(
                        [
                            PP[TARGET_DOMAIN](Y | W, Z, X2, X1),
                            PP[TARGET_DOMAIN](W | Z, X2, X1),
                            PP[TARGET_DOMAIN](Z | X2, X1),
                            PP[TARGET_DOMAIN](X2 | X1),
                        ],
                    ),
                    {Z, X2, Y, W},
                ),
            ),
            result_event=result_event,
            domain_data=domain_data,
        )
        # 4. Make sure we're not going to return any variables with None values. TODO: This check
        #    won't be necessary after we verify that the input outcomes and conditions have no
        #    None values.
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_line_4_output,
            simplified_event=simplified_event,
            outcome_and_conditioned_variable_names=outcome_and_conditioned_variable_names,
            outcome_and_conditioned_variable_names_to_values=outcome_and_conditioned_variable_names_to_values,
            outcome_ancestral_component_variables_with_no_values=outcome_ancestral_component_variables_with_no_values,
            result_expression=result_expression,
            result_event=[(Y, -Y), (W, -W), (X1, None)],
            domain_data=domain_data,
        )
        # 5. Make sure the result expression and result event have the same variables.
        self.assertRaises(
            KeyError,
            _validate_transport_conditional_counterfactual_query_line_4_output,
            simplified_event=simplified_event,
            outcome_and_conditioned_variable_names=outcome_and_conditioned_variable_names,
            outcome_and_conditioned_variable_names_to_values=outcome_and_conditioned_variable_names_to_values,
            outcome_ancestral_component_variables_with_no_values=outcome_ancestral_component_variables_with_no_values,
            result_expression=result_expression,
            result_event=[(Y, -Y), (W, -W), (X1, -X1), (R, -R)],
            domain_data=domain_data,
        )

    def test_transport_conditional_counterfactual_query_5(self):
        """Fifth test of Algorithm 3 of [correa22a], transporting a conditional counterfactual query.

        Here we test the notion that the joint probability of the vertices sent in as input conditions
        on other variables we don't see in the graph. A user may wish to transport a subgraph,
        for example.

        Source: Example 4.5 and Figure 6 of [correa22a]_.
        """
        domain_data = [(set(), PP[TARGET_DOMAIN](X, Y, Z | R)), ({X}, PP[Pi1](X, Y, Z | R))]
        expected_result_expr, expected_result_event = (
            Fraction(PP[TARGET_DOMAIN](Y | X, Z, R), Sum.safe(PP[TARGET_DOMAIN](Y | X, Z, R), {Y})),
            [(Y, -Y), (X, +X), (Z, -Z)],
        )
        result_expr, result_event = transport_conditional_counterfactual_query(
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=domain_data,
        )
        logger.debug("expected_result_expr = " + expected_result_expr.to_latex())
        logger.debug("expected_result_event = " + str(expected_result_event))
        logger.debug("Result_expr = " + result_expr.to_latex())
        logger.debug("Result_event = " + str(result_event))
        self.assert_expr_equal(expected_result_expr, result_expr)
        self.assertCountEqual(expected_result_event, result_event)

    def test_transport_conditional_counterfactual_query_6(self):
        """Sixth test of Algorithm 3 of [correa22a], transporting a conditional counterfactual query.

        Here we test the notion that the joint probability of the vertices sent in as input conditions
        on other variables we don't see in the graph. A user may wish to transport a subgraph,
        for example.

        Source: Example 4.5 and Figure 6 of [correa22a]_.
        """
        domain_data = [(set(), PP[TARGET_DOMAIN](X, Y, Z, R)), ({X}, PP[Pi1](X, Y, Z, R))]
        # What happens here is the extra variable R in the domain_data graph probability expression
        # doesn't get used during processing because it's outside the graph and we don't condition on it.
        # Leaving out R doesn't happen explicitly: it happens implicitly during compute_c_factor() called
        # from tian_pearl_identify(). compute_c_factor() leaves out R because it's not part of the graph
        # and it's not a conditioned variable, and that's expected behavior.
        expected_result_expr, expected_result_event = (
            Fraction(PP[TARGET_DOMAIN](Y | X, Z), Sum.safe(PP[TARGET_DOMAIN](Y | X, Z), {Y})),
            [(Y, -Y), (X, +X), (Z, -Z)],
        )
        result_expr, result_event = transport_conditional_counterfactual_query(
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=domain_data,
        )
        logger.debug("expected_result_expr = " + expected_result_expr.to_latex())
        logger.debug("expected_result_event = " + str(expected_result_event))
        logger.debug("Result_expr = " + result_expr.to_latex())
        logger.debug("Result_event = " + str(result_event))
        self.assert_expr_equal(expected_result_expr, result_expr)
        self.assertCountEqual(expected_result_event, result_event)

    def test_transport_conditional_counterfactual_query_7(self):
        """Seventh test of Algorithm 3 of [correa22a], transporting a conditional counterfactual query.

        This test checks that if a counterfactual factor can't be transported, the algorithm
        returns None.

        Source: RJC's mind.
        """
        target_domain_graph = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (Z, Y),
                (X, Y),
                (X, W),
                (W, Y),
                (Y, R),
            ],
            undirected=[(Z, X), (W, Y)],
        )
        graph_1 = NxMixedGraph.from_edges(
            directed=[(Z, X), (X, Y), (Z, Y), (W, Y), (transport_variable(Y), Y), (Y, R)],
            undirected=[
                (Z, X),
            ],
        )
        graph_1_topo = graph_1.topological_sort()
        # Let's say graph 2 is a stochastic intervention where Y is a function of W and Z,
        # but not X.
        graph_2 = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (X, W),
                (Z, Y),
                (W, Y),
                (Y, R),
            ],
            undirected=[],
        )
        graph_2_topo = graph_2.topological_sort()
        domain_data = [(set(), PP[Pi1](W, X, Y, Z, R)), ({Y}, PP[Pi2](W, X, Y, Z, R))]
        query_result = transport_conditional_counterfactual_query(
            outcomes=[(Y @ -X, -Y), (X, -X)],
            conditions=[(R, -R)],
            target_domain_graph=target_domain_graph,
            domain_graphs=[(graph_1, graph_1_topo), (graph_2, graph_2_topo)],
            domain_data=domain_data,
        )
        self.assertIsNone(query_result)

    def test_transport_conditional_counterfactual_query_8(self):
        """Eigth test of Algorithm 3 of [correa22a], transporting a conditional counterfactual query.

        This tests whether we properly return (Zero(), None) if a query, when simplified,
        returns a probability of zero.

        Source: Example 4.2 from [correa22]_, modified to make the simplification fail.
        """
        # 1. We need a test case that returns a probability of Zero() if the Simplify() algorithm
        #   returns a probability 0 during transport_unconditional_counterfactual_query().
        outcomes = [(Y @ -Y, -Y), (Y, +Y)]
        conditions = [(R, -R)]
        # event = [(Y @ -X, -Y), (X, -X)]
        target_domain_graph = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (Z, Y),
                (X, Y),
                (X, W),
                (W, Y),
                (Y, R),
            ],
            undirected=[(Z, X), (W, Y)],
        )
        # From [correa22a]_, Figures 3a and 4, with an extra vertex R that is a descendent of Y.
        graph_1 = NxMixedGraph.from_edges(
            directed=[(X, Y), (X, W), (W, Y), (Z, Y), (transport_variable(Z), Z), (Y, R)],
            undirected=[
                (W, Y),
            ],
        )
        graph_1_topo = list(graph_1.topological_sort())

        # From [correa22a]_, Figure 3b, with an extra vertex R that is a descendent of Y.
        graph_2 = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (Z, Y),
                (X, Y),
                (X, W),
                (W, Y),
                (transport_variable(W), W),
                (Y, R),
            ],
            undirected=[(Z, X), (W, Y)],
        )
        graph_2_topo = list(graph_2.topological_sort())
        domain_graphs = [
            (
                graph_1,
                graph_1_topo,
            ),
            (
                graph_2,
                graph_2_topo,
            ),
        ]
        domain_data = [({X}, PP[Pi1](W, X, Y, Z, R)), (set(), PP[Pi2](W, X, Y, Z, R))]
        result_expr, result_event = transport_conditional_counterfactual_query(
            outcomes=outcomes,
            conditions=conditions,
            target_domain_graph=target_domain_graph,
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )
        self.assert_expr_equal(result_expr, Zero())
        self.assertIsNone(result_event)

    def test_transport_conditional_counterfactual_query_preprocessing(self):
        """Tests of input validation for transport_conditional_counterfactual_query() [correa22a].

        Here are all the checks (numbering is just based on convenience during implementation, and
        the numbered order is not necessarily the order of implementation):
        1. Type checking for outcomes and conditions
        2. Type checking for target_domain_graph
        3. Type checking for domain_graphs
        4. Type checking for domain_data
        4.5. Make sure probabilistic expressions in domain_data aren't Zero() or One()
        5. Make sure conditions and outcomes aren't empty
        6. (Skipped for the conditional transportability algorithm, included for unconditional
        transportability) Make sure at least one event element has a non-None value
        7. Check domain_graphs and domain_data aren't empty lists
        8. Check all graphs in domain_graphs have nodes
        9. Check all topologically sorted lists have entries
        9.2. Check that the target domain graph contains no transportability nodes and is a directed acyclic graph
        9.5. Check that the domain_graphs and domain_data list lengths are equal
        9.7. Check that every domain graph is a directed acyclic graph
        10. Check that every topological order list in domain_graphs is a valid topological order,
            given the corresponding graph
        11. Check the domain graph vertices are all the same as the target domain graph vertices
        12. Check the event vertices are in the target domain graph (given check #11, that
            means they're in every graph)
        13. Check the conditioned and outcome variables have the same base variable
            as the base variable of their corresponding values
        14. Domain graphs: make sure the vertex set of the topologically sorted vertex order matches
            the set of vertices in each corresponding domain graph
        15. It's possible for a graph probability expression to contain vertices not in the graph
            due to conditioning on vertices outside the graph. But the graph vertices must all be
            represented in that graph probability expression.
        16. If the target domain graph is also in the domain_graphs list (i.e., data were collected for
            the target domain), then the target domain graph in the domain_graphs list must be
            identical to the target_domain_graph parameter.
        17. Make sure the conditioned variable and outcome variable sets don't share graph vertices

        Source: RJC.
        """
        # 1. Type check the outcomes
        example_1_domain_data = [(set(), PP[TARGET_DOMAIN](X, Y, Z)), ({X}, PP[Pi1](X, Y, Z))]
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=1,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=[1],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=1,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=[1],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=[(Y @ -X, -Y, None)],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=[(Y @ -X, -Y, None)],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=[(1, -Y)],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=[(Y @ -X, None)],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=[(1, -Y)],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=[(Y @ -X, None)],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        # 1. Type check the conditions
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=1,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=[1],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=1,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=[1],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=[(Z @ -X, -Z), (5, +X), None],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=[(Z @ -X, -Z), (5, +X), None],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=[(Z @ -X, -Z), (5, +X)],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=[(Z @ -X, -Z), (5, +X)],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=[(Z @ -X, -Z), (X, None)],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=[(Z @ -X, -Z), (X, None)],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        # Check we have no empty inputs, for inputs unique to Algorithm 3
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=[],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=[],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=[],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=[],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=NxMixedGraph(),
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=NxMixedGraph(),
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        # Type checking for inputs consistent with both Algorithms 2 and 3
        # 2.
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=1,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=1,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=1,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=1,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[None],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[1],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    figure_1_graph_no_transportability_nodes_topo,
                ),
                (
                    1,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    figure_1_graph_no_transportability_nodes_topo,
                ),
                (
                    1,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    1,
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    1,
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    [1],
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    [1],
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        # 4.
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=1,
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=1,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[1],
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[1],
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[(None, PP[TARGET_DOMAIN](X, Y, Z)), ({X}, PP[Pi1](X, Y, Z))],
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[({1}, PP[TARGET_DOMAIN](X, Y, Z)), ({X}, PP[Pi1](X, Y, Z))],
        )
        self.assertRaises(
            TypeError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[(set(), 1), ({X}, PP[Pi1](X, Y, Z))],
        )
        self.assertRaises(
            TypeError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[(set(), 1), ({X}, PP[Pi1](X, Y, Z))],
        )
        # 4.5.
        self.assertRaises(
            NotImplementedError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[(set(), Zero()), ({X}, PP[Pi1](X, Y, Z))],
        )
        self.assertRaises(
            NotImplementedError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[(set(), Zero()), ({X}, PP[Pi1](X, Y, Z))],
        )
        self.assertRaises(
            NotImplementedError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[(set(), One()), ({X}, PP[Pi1](X, Y, Z))],
        )
        self.assertRaises(
            NotImplementedError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[(set(), One()), ({X}, PP[Pi1](X, Y, Z))],
        )
        # 17.
        # self.assertRaises(
        #    NotImplementedError,
        #    _validate_transport_conditional_counterfactual_query_input,
        #    outcomes=[(Y @ -X, -Y)],
        #    conditions=[(Y @ -Z, -Y), (X, +X)],
        #    target_domain_graph=self.example_1_target_domain_graph,
        #    domain_graphs=self.example_1_domain_graphs,
        #    domain_data=[(set(), PP[TARGET_DOMAIN](X, Y, Z)), ({X}, PP[Pi1](X, Y, Z))],
        # )
        # self.assertRaises(
        #    NotImplementedError,
        #    transport_conditional_counterfactual_query,
        #    outcomes=[(Y @ -X, -Y)],
        #    conditions=[(Y @ -Z, -Y), (X, +X)],
        #    target_domain_graph=self.example_1_target_domain_graph,
        #    domain_graphs=self.example_1_domain_graphs,
        #    domain_data=[(set(), PP[TARGET_DOMAIN](X, Y, Z)), ({X}, PP[Pi1](X, Y, Z))],
        # )
        # 7.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[],
            domain_data=[(set(), PP[TARGET_DOMAIN](X, Y, Z)), ({X}, PP[Pi1](X, Y, Z))],
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[],
            domain_data=[(set(), PP[TARGET_DOMAIN](X, Y, Z)), ({X}, PP[Pi1](X, Y, Z))],
        )
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[],
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[],
        )
        # 8.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    figure_1_graph_no_transportability_nodes_topo,
                ),
                (
                    NxMixedGraph(),
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    figure_1_graph_no_transportability_nodes_topo,
                ),
                (
                    NxMixedGraph(),
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        # 9.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    figure_1_graph_no_transportability_nodes_topo,
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    [],
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    figure_1_graph_no_transportability_nodes_topo,
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    [],
                ),
            ],
            domain_data=example_1_domain_data,
        )
        # 9.5.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    figure_1_graph_no_transportability_nodes_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    figure_1_graph_no_transportability_nodes_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        # 9.2.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=figure_1_graph_domain_1_with_interventions,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=figure_1_graph_domain_1_with_interventions,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=figure_1_graph_domain_1_with_interventions,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=figure_1_graph_domain_1_with_interventions,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=NxMixedGraph.from_edges(
                directed=[
                    (X, Z),
                    (Z, Y),
                    (Y, X),
                ],
            ),
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=NxMixedGraph.from_edges(
                directed=[
                    (X, Z),
                    (Z, Y),
                    (Y, X),
                ],
            ),
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        # 11.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=NxMixedGraph.from_edges(
                directed=[
                    (X, Z),
                    (Z, Y),
                    (X, R),
                ],
            ),
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=NxMixedGraph.from_edges(
                directed=[
                    (X, Z),
                    (Z, Y),
                    (X, R),
                ],
            ),
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    figure_1_graph_no_transportability_nodes_topo,
                ),
                (
                    NxMixedGraph.from_edges(
                        directed=[(X, Z), (X, Y), (R, Y)],
                        undirected=[],
                    ),
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    figure_1_graph_no_transportability_nodes,
                    figure_1_graph_no_transportability_nodes_topo,
                ),
                (
                    NxMixedGraph.from_edges(
                        directed=[(X, Z), (X, Y), (R, Y)],
                        undirected=[],
                    ),
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        # 12.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=[(Z @ -X, -Z), (R, -R)],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=[(Z @ -X, -Z), (R, -R)],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=[(R @ -X, -R)],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=[(R @ -X, -R)],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        # 13.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=[(Z @ -X, -Z), (X, -Z)],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=[(Z @ -X, -Z), (X, -Z)],
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=[(Y @ -X, -R)],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=[(Y @ -X, -Z)],
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        # 14.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (X, Z),
                            (Z, Y),
                            (X, Y),
                        ],
                        undirected=[(Z, X)],
                    ),
                    [X, Z, Y, R],
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (X, Z),
                            (Z, Y),
                            (X, Y),
                        ],
                        undirected=[(Z, X)],
                    ),
                    [X, Z, Y, R],
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        # 15.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[(set(), PP[TARGET_DOMAIN](X, Y)), ({X}, PP[Pi1](X, Y, Z))],
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[(set(), PP[TARGET_DOMAIN](X, Y)), ({X}, PP[Pi1](X, Y, Z))],
        )
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[(set(), PP[TARGET_DOMAIN](X, Y, Z)), ({X, R}, PP[Pi1](X, Y, Z))],
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=self.example_1_domain_graphs,
            domain_data=[(set(), PP[TARGET_DOMAIN](X, Y, Z)), ({X, R}, PP[Pi1](X, Y, Z))],
        )
        # 10.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (X, Z),
                            (Z, Y),
                            (X, Y),
                        ],
                        undirected=[(Z, X)],
                    ),
                    [Y, Z, X],
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (X, Z),
                            (Z, Y),
                            (X, Y),
                        ],
                        undirected=[(Z, X)],
                    ),
                    [Y, Z, X],
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        # 16.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=NxMixedGraph.from_edges(
                directed=[
                    (X, Z),
                    (Z, Y),
                    (X, Y),
                ],
                undirected=[],
            ),
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=NxMixedGraph.from_edges(
                directed=[
                    (X, Z),
                    (Z, Y),
                    (X, Y),
                ],
                undirected=[],
            ),
            domain_graphs=self.example_1_domain_graphs,
            domain_data=example_1_domain_data,
        )
        # 9.7.
        self.assertRaises(
            ValueError,
            _validate_transport_conditional_counterfactual_query_input,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (X, Z),
                            (Z, Y),
                            (Y, X),
                        ],  # not a directed acyclic graph
                        undirected=[(Z, X)],
                    ),
                    [X, Z, Y],
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_conditional_counterfactual_query,
            outcomes=self.example_1_outcomes,
            conditions=self.example_1_conditions,
            target_domain_graph=self.example_1_target_domain_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (X, Z),
                            (Z, Y),
                            (Y, X),
                        ],  # not a directed acyclic graph
                        undirected=[(Z, X)],
                    ),
                    [X, Z, Y],
                ),
                (
                    figure_1_graph_domain_1_with_interventions,
                    figure_1_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=example_1_domain_data,
        )


class TestTransportConditionalCounterfactualQueryUtils(cases.GraphTestCase):
    """Test utility functions to transport a conditional counterfactual query."""

    def test_valid_topo_list(self):
        r"""Test the valid_topo_list() utility function.

        Source: RC's mind.
        """
        graph = NxMixedGraph.from_edges(
            directed=[
                (X1, Z),
                (X2, Z),
                (X2, W),
                (W, Y),
                (Z, Y),
            ],
        )
        topo = list(graph.topological_sort())
        self.assertTrue(_valid_topo_list([X1, X2, Z, W, Y], graph))
        self.assertTrue(_valid_topo_list(topo, graph))
        self.assertFalse(_valid_topo_list([X1, Z, X2, W, Y], graph))
        self.assertTrue(_valid_topo_list([X2, W, X1, Z, Y], graph))


class TestTransportUnconditionalCounterfactualQueryPreprocessing(cases.GraphTestCase):
    """Test input validation for [correa22a]_'s unconditional counterfactual transportability algorithm."""

    @classmethod
    def setUpClass(cls):
        """Set up the class."""
        cls.event = [(Y @ -X, -Y), (X, -X)]
        cls.domain_graphs = [
            (
                figure_2_graph_domain_1_with_interventions,
                figure_2_graph_domain_1_with_interventions_topo,
            ),
            (
                figure_2_graph_domain_2,
                figure_2_graph_domain_2_topo,
            ),
        ]

    def test_unconditional_counterfactual_query_preprocessing(self):
        """Tests of input validation for transport_conditional_counterfactual_query() [correa22a].

        Here are all the checks (numbering is just based on convenience during implementation, and
        the numbered order is not necessarily the order of implementation):
        1. Type checking for the input event
        2. Type checking for target_domain_graph
        3. Type checking for domain_graphs
        4. Type checking for domain_data
        4.5. Make sure probabilistic expressions in domain_data aren't Zero() or One()
        5. Make sure the input event isn't empty
        6. Make sure at least one event element has a non-None value
        7. Check domain_graphs and domain_data aren't empty lists
        8. Check all graphs in domain_graphs have nodes
        9. Check all topologically sorted lists have entries
        9.2. Check that the target domain graph contains no transportability nodes and is a directed acyclic graph
        9.5. Check that the domain_graphs and domain_data list lengths are equal
        9.7. Check that every domain graph is a directed acyclic graph
        10. Check that every topological order list in domain_graphs is a valid topological order,
            given the corresponding graph
        11. Check the domain graph vertices are all the same as the target domain graph vertices
        12. Check the event vertices are in the target domain graph (given check #11, that
            means they're in every graph)
        13. Check the event variables have the same base variable as the base variable of their
            corresponding values, if those values aren't None
        14. Domain graphs: make sure the vertex set of the topologically sorted vertex order matches
            the set of vertices in each corresponding domain graph
        15. It's possible for a graph probability expression to contain vertices not in the graph
            due to conditioning on vertices outside the graph. But the graph vertices must all be
            represented in that graph probability expression
        15.5. Make sure policy vertices are in the target domain graph
        16. If the target domain graph is also in the domain_graphs list (i.e., data were collected for
            the target domain), then the target domain graph in the domain_graphs list must be
            identical to the target_domain_graph parameter

        Source: RJC.
        """
        # 1. Type check the outcomes
        domain_data = [({X}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))]
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=1,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=1,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=[1],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=[1],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=[(Y @ -X, -Y), (X, -X), None],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=[(Y @ -X, -Y), (X, -X), None],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=[(Y @ -X, -Y, None), (X, -X)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=[(Y @ -X, -Y, None), (X, -X)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=[(1, -Y), (X, -X)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=[(1, -Y), (X, -X)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=[(Y @ -X, 1), (X, -X)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=[(Y @ -X, 1), (X, -X)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        # None values for event interventions are allowed for Algorithm 2 but not Algorithm 3.
        # So the next two (commented out) tests fail and are supposed to fail.
        # self.assertRaises(
        #    TypeError,
        #    _validate_transport_unconditional_counterfactual_query_input,
        #    event=[(Y @ -X, None), (X, -X)],
        #    target_domain_graph=figure_2a_graph,
        #    domain_graphs=self.domain_graphs,
        #    domain_data=domain_data,
        # )
        # self.assertRaises(
        #    TypeError,
        #    transport_unconditional_counterfactual_query,
        #    event=[(Y @ -X, None), (X, -X)],
        #    target_domain_graph=figure_2a_graph,
        #    domain_graphs=self.domain_graphs,
        #    domain_data=domain_data,
        # )
        # 5.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=[],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=[],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        # 2.
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=1,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=None,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        # 3.
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=1,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=1,
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[1],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[None],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    1,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    1,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    1,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    1,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    [1],
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    [1],
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=1,
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=1,
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[1],
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[1],
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[(None, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[(1, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({1}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({1}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            TypeError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({X}, 1), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            TypeError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({X}, 1), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            NotImplementedError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({X}, Zero()), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            NotImplementedError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({X}, Zero()), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            NotImplementedError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({X}, One()), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            NotImplementedError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({X}, One()), (set(), PP[Pi2](W, X, Y, Z))],
        )
        # 6. At least one event element has a non-None value
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=[(Y @ -X, None), (X, None)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=[(Y @ -X, None), (X, None)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        # 7.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[],
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[],
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[],
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[],
        )
        # 8.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=NxMixedGraph(),
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=NxMixedGraph(),
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    NxMixedGraph(),
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    NxMixedGraph(),
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        # 9.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    [],
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    [],
                ),
            ],
            domain_data=domain_data,
        )
        # 9.5.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    figure_2_graph_domain_1_with_interventions,
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
            ],
            domain_data=domain_data,
        )
        # 9.2
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2_graph_domain_2,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2_graph_domain_2,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=NxMixedGraph.from_edges(
                directed=[
                    (Z, X),
                    (Y, Z),
                    (X, Y),
                    (X, W),
                    (W, Y),
                ],
                undirected=[(Z, X), (W, Y)],
            ),
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=NxMixedGraph.from_edges(
                directed=[
                    (Z, X),
                    (Y, Z),
                    (X, Y),
                    (X, W),
                    (W, Y),
                ],
                undirected=[(Z, X), (W, Y)],
            ),
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        # 11.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=NxMixedGraph.from_edges(
                directed=[
                    (Z, X),
                    (Z, R),
                    (X, Y),
                    (X, W),
                    (W, Y),
                ],
                undirected=[(Z, X), (W, Y)],
            ),
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=NxMixedGraph.from_edges(
                directed=[
                    (Z, X),
                    (Z, R),
                    (X, Y),
                    (X, W),
                    (W, Y),
                ],
                undirected=[(Z, X), (W, Y)],
            ),
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[(X, Y), (X, W), (W, R), (Z, Y), (transport_variable(Z), Z)],
                        undirected=[
                            (W, Y),
                        ],
                    ),
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[(X, Y), (X, W), (W, R), (Z, Y), (transport_variable(Z), Z)],
                        undirected=[
                            (W, Y),
                        ],
                    ),
                    figure_2_graph_domain_1_with_interventions_topo,
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        # 12.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=[(Y @ -X, -Y), (R, -R)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=[(Y @ -X, -Y), (R, -R)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        # 13.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=[(Y @ -X, -Y), (X, -R)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=[(Y @ -X, -Y), (X, -R)],
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=domain_data,
        )
        # 14.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[(X, Y), (X, W), (W, Y), (Z, Y), (transport_variable(Z), Z)],
                        undirected=[
                            (W, Y),
                        ],
                    ),
                    [Z, X, W, R],
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[(X, Y), (X, W), (W, Y), (Z, Y), (transport_variable(Z), Z)],
                        undirected=[
                            (W, Y),
                        ],
                    ),
                    [Z, X, W, R],
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        # 15.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({X}, PP[Pi1](W, X, Y)), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({X}, PP[Pi1](W, X, Y)), (set(), PP[Pi2](W, X, Y, Z))],
        )
        # 15.5.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({X}, PP[Pi1](W, X, Y, Z)), ({R}, PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=self.domain_graphs,
            domain_data=[({X}, PP[Pi1](W, X, Y, Z)), ({R}, PP[Pi2](W, X, Y, Z))],
        )
        # 10.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (X, Y),
                            (X, W),
                            (W, Y),
                            (Z, Y),
                            (Z, X),
                            (transport_variable(Z), Z),
                        ],
                        undirected=[
                            (W, Y),
                        ],
                    ),
                    [transport_variable(Z), X, Z, W, Y],
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (X, Y),
                            (X, W),
                            (W, Y),
                            (Z, Y),
                            (Z, X),
                            (transport_variable(Z), Z),
                        ],
                        undirected=[
                            (W, Y),
                        ],
                    ),
                    [transport_variable(Z), X, Z, W, Y],
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=domain_data,
        )
        # 16.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (Z, X),
                            (Z, Y),
                            (X, Y),
                            (X, W),
                            (W, Y),
                        ],
                        undirected=[(Z, X)],  # Missing an undirected edge
                    ),
                    [Z, X, W, Y],
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=[({X}, PP[TARGET_DOMAIN](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (Z, X),
                            (Z, Y),
                            (X, Y),
                            (X, W),
                            (W, Y),
                        ],
                        undirected=[(Z, X)],  # Missing an undirected edge
                    ),
                    [Z, X, W, Y],
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=[({X}, PP[TARGET_DOMAIN](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))],
        )
        # 9.7.
        self.assertRaises(
            ValueError,
            _validate_transport_unconditional_counterfactual_query_input,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (Z, X),
                            (Y, Z),
                            (X, Y),
                            (X, W),
                            (W, Y),
                        ],  # Not a DAG
                        undirected=[(Z, X), (W, Y)],
                    ),
                    [Z, X, W, Y],
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=[({X}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))],
        )
        self.assertRaises(
            ValueError,
            transport_unconditional_counterfactual_query,
            event=self.event,
            target_domain_graph=figure_2a_graph,
            domain_graphs=[
                (
                    NxMixedGraph.from_edges(
                        directed=[
                            (Z, X),
                            (Y, Z),
                            (X, Y),
                            (X, W),
                            (W, Y),
                        ],  # Not a DAG
                        undirected=[(Z, X)],
                    ),
                    [Z, X, W, Y],
                ),
                (
                    figure_2_graph_domain_2,
                    figure_2_graph_domain_2_topo,
                ),
            ],
            domain_data=[({X}, PP[Pi1](W, X, Y, Z)), (set(), PP[Pi2](W, X, Y, Z))],
        )


class TestGetCounterfactualFactorsRetainingVariableValues(cases.GraphTestCase):
    """Test function to retrieve counterfactual factors from a graph."""

    def test_get_counterfactual_factors_retaining_variable_values(self):
        r"""Test a function to retrieve counterfactual factors, retaining variable values.

        Source: RC's mind.
        """
        event = [(Y @ -X, -Y), (X @ -Z, -X)]
        self.assertRaises(
            ValueError,
            get_counterfactual_factors_retaining_variable_values,
            event=event,
            graph=figure_2a_graph,
        )
        # The next test is expected to fail and it does
        event_2 = [(Y @ -X @ -W @ -Z, -Y), (X @ -Z, -X)]
        expected_result = [{(Y @ -X @ -W @ -Z, -Y)}, {(X @ -Z, -X)}]
        # expected_result[frozenset({W, Y})] = {(Y @ -X @ -W @ -Z, -Y)}
        # expected_result[frozenset({X, Z})] = {(X @ -Z, -X)}
        self.assertCountEqual(
            expected_result,
            get_counterfactual_factors_retaining_variable_values(
                event=event_2, graph=figure_2a_graph
            ),
        )

    # district_mappings: DefaultDict[
    #    frozenset[Variable], set[tuple[Variable, Intervention | None]]
    # ] = defaultdict(set)
    # for variable, value in event:
    #    district_mappings[graph.get_district(variable.get_base())].add((variable, value))
    # self.assertRaises(
    #   ValueError,
    #   get_counterfactual_factors_retaining_variable_values,
    #   event=event_2,
    #   graph=figure_2a_graph
    # )


class TestMergeFrozenSetsWithCommonElements(cases.GraphTestCase):
    """Utility function to merge frozen sets that have common elements."""

    def test_merge_frozen_sets_with_common_vertices(self):
        r"""Test a helper function to merge ancestral components based on common vertices.

        Source: RC's mind.
        """
        test_1_inputs = {
            frozenset([W, X]),
            frozenset([X, Y]),
            frozenset([W, Z]),
            frozenset([X1, X2]),
            frozenset([X1, R, W1]),
        }
        result = _merge_frozen_sets_with_common_vertices(test_1_inputs)
        expected_result = {frozenset([W, X, Y, Z]), frozenset([R, W1, X1, X2])}
        self.assertSetEqual(result, expected_result)
        test_2_inputs = {
            frozenset([W]),
            frozenset(),
            frozenset([X]),
            frozenset([W, Y]),
            frozenset([Z, R]),
            frozenset([W1]),
        }
        result_2 = _merge_frozen_sets_with_common_vertices(test_2_inputs)
        expected_result_2 = {frozenset([W, Y]), frozenset([X]), frozenset([W1]), frozenset([R, Z])}
        self.assertSetEqual(result_2, expected_result_2)

    def test_merge_frozen_sets_linked_by_bidirectional_edges(self):
        r"""Test a helper function to merge ancestral components linked by bidirectional edges.

        Source: RC's mind.
        """
        graph_1 = NxMixedGraph.from_edges(
            directed=[],  # Not a DAG
            undirected=[(W, Y), (W, X), (W1, R)],
        )
        test_1_inputs = {frozenset([W, Y]), frozenset([X]), frozenset([W1]), frozenset([R, Z])}
        result = _merge_frozen_sets_linked_by_bidirectional_edges(
            input_sets=test_1_inputs, graph=graph_1
        )
        expected_result = {frozenset([W, X, Y]), frozenset([R, W1, Z])}
        self.assertSetEqual(result, expected_result)
        graph_2 = NxMixedGraph.from_edges(
            directed=[],
            undirected=[],
        )
        test_2_inputs = {frozenset([X]), frozenset([W, Y]), frozenset([Z, R]), frozenset([W1])}
        result_2 = _merge_frozen_sets_linked_by_bidirectional_edges(
            input_sets=test_2_inputs, graph=graph_2
        )
        expected_result_2 = {frozenset([W, Y]), frozenset([X]), frozenset([W1]), frozenset([R, Z])}
        logger.debug(str(expected_result_2))
        logger.debug(str(result_2))
        self.assertSetEqual(result_2, expected_result_2)
        graph_3 = NxMixedGraph.from_edges(directed=[], undirected=[(W, X), (X, Y), (R, Z), (W1, Y)])
        result_3 = _merge_frozen_sets_linked_by_bidirectional_edges(
            input_sets=test_2_inputs, graph=graph_3
        )
        expected_result_3 = {frozenset([W, Y, X, W1]), frozenset([R, Z])}
        self.assertSetEqual(result_3, expected_result_3)

    def test_convert_counterfactual_variables_to_base_variables(self):
        r"""Test a helper function to merge ancestral components linked by bidirectional edges.

        Source: RC's mind.
        """
        test_1_input = frozenset([X])
        result_1 = get_base_variables(test_1_input)
        self.assertSetEqual(result_1, frozenset([X]))
        test_2_input = frozenset([X @ -Y])
        result_2 = get_base_variables(test_2_input)
        self.assertSetEqual(result_2, frozenset([X]))
        test_3_input = frozenset([])
        self.assertSetEqual(get_base_variables(test_3_input), frozenset([]))
