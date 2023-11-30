"""Tests for counterfactual transportability."""

import unittest

from y0.algorithm.counterfactual_transportability import simplify, get_ancestors_of_counterfactual, is_ctf_factor
from y0.graph import NxMixedGraph
from y0.dsl import X, Y, W, Z, Variable, CounterfactualVariable, Intervention, Zero
from y0.algorithm.transport import make_selection_diagram, transport_variable

class TestGetAncestorsOfCounterfactual(unittest.TestCase):
    """Test getting the ancestors of a counterfactual following Correa, Lee, and Bareinboim 2022, Definition 2.1 and Example 2.1"""
    ## TODO: It seems important to be able to represent Y_{x_0}, not just Y_{X}. (That is, the value of Y were we to intervene on X 
    ## and set its value to x0). Ask how we can do this with Y0 syntax.

    def __init__(cls):
        cls.figure_2a_graph = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (Z, Y),
                (X, Y),
                (X, W),
                (W, Y),
            ],
            undirected=[
                (Z, X),
                (W, Y)
            ]
        )

    def test_example_2_1(cls):
        test1_in = CounterfactualVariable(name=Y, star=None, interventions=(Intervention(name=X,star=False),)) # Y @ -X
        test1_out = set(CounterfactualVariable(name=Y, star=None, interventions=(Intervention(name=X,star=False),)), # Y @ -X)
                        CounterfactualVariable(name=W, star=None, interventions=(Intervention(name=X,star=False),)), # W @ -Z
                        Variable(name=Z) # Z
        )
        result = get_ancestors_of_counterfactual(
            tuple[test1_in, cls.figure_2a_graph]
        )
        cls.assertTrue(variable in test1_out for variable in result)
        cls.assertTrue(test1_in in result) # Every outcome is its own ancestor

    def test_example_2_2(cls):
        test2_in = CounterfactualVariable(name=W, star=None, interventions=(Intervention(name=Y,star=False),Intervention(name=Z,star=False))) # W @ -Y @ -Z
        test2_out = set(CounterfactualVariable(name=W, star=None, interventions=(Intervention(name=Z,star=False),)), # W @ -Z
                        CounterfactualVariable(name=X, star=None, interventions=(Intervention(name=Z,star=False),)), # X @ -Z
        )
        result = get_ancestors_of_counterfactual(
            tuple[test2_in, cls.figure_2a_graph]
        )
        cls.assertTrue(variable in test2_out for variable in result)
        cls.assertTrue(test2_in in result) # Every outcome is its own ancestor

    def test_example_2_3(cls):
        test3_in = CounterfactualVariable(name=Y, star=None, interventions=(Intervention(name=W,star=False),)) # Y @ -W
        test3_out = set(CounterfactualVariable(name=Y, star=None, interventions=(Intervention(name=W,star=False),)), # Y @ -W
                        Variable(name=X), # X
                        Variable(name=Z) # Z
        )
        result = get_ancestors_of_counterfactual(
            tuple[test3_in, cls.figure_2a_graph]
        )
        cls.assertTrue(variable in test3_out for variable in result)
        cls.assertTrue(test3_in in result) # Every outcome is its own ancestor


class TestSimplify(unittest.TestCase):
    """Test the simplify algorithm from counterfactual transportability."""
    ##TODO: Incorporate a test involving counterfactual unnesting.

    def test_inconsistent_1(cls):
        """Test simplifying an inconsistent event. Correa et al. specify the output should be 0 if the 
        counterfactual event is guaranteed to have probability 0.
        """
        event = [
            (Y @ -X, -Y),
            (Y @ -X, +Y)
        ]
        cls.assertEquals(simplify(event), Zero())

    def test_inconsistent_2(cls):
        """Second test for simplifying an inconsistent event."""
        event = [
            (Y @ -Y, +Y)
        ]
        cls.assertEquals(simplify(event), Zero())

    def test_redundant_1(cls):
        """First test for simplifying an event with redundant subscripts."""
        event = [
            (Y @ -X, -Y),
            (Y @ -X, -Y)
        ]
        result = simplify(event)
        cls.assertEqual(result, [(Y @ -X, -Y)])

    def test_redundant_2(cls):
        """Second test for simplifying an event with redundant subscripts."""
        event = [
            (Y @ -X, -Y),
            (Y @ -X, -Y),
            (X @ -W, -X)
        ]
        cls.assertEqual(simplify(event), [(Y @ -X, -Y), (X @ -W, -X)])

    def test_redundant_3(cls):
        """Third test for simplifying an event with redundant subscripts."""
        event = [
            (Y @ -Y, -Y),
            (X @ -W, -X),
        ]
        cls.assertEqual(simplify(event), [(X @ -W, -X)])

    def test_redundant_4(cls):
        """Fourth test for simplifying an event with redundant subscripts."""
        event = [
            (Y @ -Y, -Y),
        ]
        cls.assertIsNone(simplify(event))

class TestIsCTFFactor(unittest.TestCase):
    """Test properties of counterfactual transportability factors."""
    ##TODO: Incorporate a test involving counterfactual unnesting.

    def __init__(cls):
        cls.figure_2a_graph = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (Z, Y),
                (X, Y),
                (X, W),
                (W, Y)
            ],
            undirected=[
                (Z, X),
                (W, Y)
            ]
        )

    def test_is_ctf_factor(cls):
        """From Example 3.3 of Correa, Lee, and Barenboim 2022."""
        ## TODO: More tests, including asserting that some inputs are not counterfactual factors.
        event1 = [
            (Y @ -Z @ -W),
            (W @ -X)
        ]
        cls.assertTrue(is_ctf_factor(event = event1, graph = cls.figure_2a_graph))

        event2 = [
            (W @ X),
            (Z)
        ]
        cls.assertTrue(is_ctf_factor(event = event2, graph = cls.figure_2a_graph))

class TestMakeSelectionDiagram(unittest.TestCase):
    """Test the results of creating a selection diagram that is an amalgamation of selection diagrams for specific domains."""

    def __init__(cls):
        cls.figure_2a_graph = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (Z, Y),
                (X, Y),
                (X, W),
                (W, Y),
            ],
            undirected=[
                (Z, X),
                (W, Y)
            ]
        )

    def test_make_selection_diagram(cls):
        """Create Figure 2(b) of Correa, Lee, and Barenboim 2022 from Figures 3(a) and 3(b)."""
        selection_nodes = dict({1:iter(Z),2:iter(W)})
        selection_diagram = make_selection_diagram(
            graph = cls.figure_2a_graph,
            selection_nodes=selection_nodes
        )
        expected_selection_diagram = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (Z, Y),
                (X, Y),
                (X, W),
                (W, Y),
                (transport_variable(Z),Z), # How do we indicate with a superscript that this is from domain 1? 
                (transport_variable(W),W) # How do we indicate with a superscript that this is from domain 2?
            ],
            undirected=[
                (Z, X),
                (W, Y)
            ]
        )
        cls.assertEquals(selection_diagram, expected_selection_diagram)
