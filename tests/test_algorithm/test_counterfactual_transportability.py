"""Tests for counterfactual transportability."""

import unittest

from y0.algorithm.counterfactual_transportability import simplify, get_ancestors_of_counterfactual
from y0.graph import NxMixedGraph
from y0.dsl import X, Y, W, Z, Variable, CounterfactualVariable, Intervention


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

    def test_inconsistent(self):
        """Test simplifying an inconsistent event."""
        event = [
            (Y @ -X, +Y),
            (Y @ -X, -Y),
        ]
        result = simplify(event)
        self.assertIsNone(result)
    
