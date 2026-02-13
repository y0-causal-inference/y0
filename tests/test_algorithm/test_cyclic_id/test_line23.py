"""Tests for the line 23 fix in the cyclic ID algorithm"""



import unittest
from tests.test_algorithm import cases
from y0.examples import napkin
from y0.algorithm.identify.cyclic_id import identify_district_variables_cyclic
from y0.dsl import X, Y, Z1, Z2, P, Sum, Fraction, Variable
from y0.graph import NxMixedGraph


class TestIdentifyDistrictVariablesCyclic(cases.GraphTestCase):
    """Tests for the generalized identify_district_variables function"""
    
    def test_function_exists_and_accepts_parameters(self):
        """Verify that the function exists and accepts the correct parameters."""
        
        # simple graph to test function signature
        simple_graph = NxMixedGraph.from_edges(
            directed=[],
            undirected=[]
        )
        
        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y}),
            input_district=frozenset({Y}),
            district_probability=P(Y),
            graph=simple_graph,
            topo=[Y],
        )
        
        self.assertIsNotNone(result)
        
    def test_base_case_1(self):
        """
        When A=C, it should return Q[C] = Sum[T\C]Q.
        """
        graph = NxMixedGraph.from_edges(
            directed=[],
            undirected=[]
        )
        
        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y}),
            input_district=frozenset({Y}),
            district_probability=P(Y),
            graph=graph,
            topo=[Y],
        )
        
        self.assert_expr_equal(P(Y), result)
        
    def test_base_case_1_marginalization(self):
        graph = NxMixedGraph.from_edges(
            directed=[(X, Y)],
            undirected=[]
        )
     
        graph.add_node(Z1)  # Add disconnected node Z1
    
        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y}),
            input_district=frozenset({Y, Z1}),
            district_probability=P(Y, Z1),
            graph=graph,
            topo=[X, Y, Z1],
        )
    
        expected = Sum[Z1](P(Y, Z1))
        self.assert_expr_equal(expected, result)
        
    def test_base_case_1_multiple_variables(self):
        graph = NxMixedGraph.from_edges(
            directed=[(X, Y), (X, Z1)],
            undirected=[]
        )
        
        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y, Z1}),
            input_district=frozenset({Y, Z1}),
            district_probability=P(Y, Z1),
            graph=graph,
            topo=[X, Y, Z1],
        )
        
        
        self.assert_expr_equal(P(Y, Z1), result)
        
    def test_base_case_2_no_confounding_returns_none(self):
        
        graph = NxMixedGraph.from_edges(
            directed=[(X, Y)],
            undirected=[]
        )
        
        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y}),
            input_district=frozenset({X, Y}),
            district_probability=P(X, Y),
            graph=graph,
            topo=[X, Y],
        )
        
        self.assertIsNone(result)
        
    def test_base_case_2_with_confounding_returns_fraction(self):
        
        graph = NxMixedGraph.from_edges(
            directed=[(X, Y)],
            undirected=[(X, Y)]
        )
        
        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y}),
            input_district=frozenset({X, Y}),
            district_probability=P(X, Y),
            graph=graph,
            topo=[X, Y],
        )
        
        self.assertIsInstance(result, Fraction)
        
        
    def test_case_3_triggers_recursion(self):
        
        
        graph = NxMixedGraph.from_edges(
            directed=[(Z2, Z1), (Z1, X), (X, Y)],
            undirected=[(Z2, X), (Z2, Y)]
        )
        
        result = identify_district_variables_cyclic(
            input_variables=frozenset({Y}),
            input_district=frozenset({Z2, X, Y}),
            district_probability=P(Z2, X, Y),
            graph=graph,
            topo=[Z2, Z1, X, Y],
        )
        
        
        self.assertIsInstance(result, Fraction)
    

    def test_napkin_full_integration(self):
        """Integration test: full cyclic_id() on napkin graph gives correct formula."""
        from y0.algorithm.identify.cyclic_id import cyclic_id
    
        graph = NxMixedGraph.from_edges(
            directed=[(Z2, Z1), (Z1, X), (X, Y)],
            undirected=[(Z2, X), (Z2, Y)]
        )
    
        result = cyclic_id(
            graph=graph,
            outcomes={Y},
            interventions={X},
        )
        
        print(f"result = {result}")
        print(f"result latex = {result.to_latex()}")
    

        # P(Y | do(X)) = Sum[Z2](P(Y,X|Z1,Z2) * P(Z2)) / Sum[Z2](P(X|Z1,Z2) * P(Z2))
        inner_term = P(X, Y, Z1, Z2) * P(Z2) / P(Z1, Z2)
        numerator = Sum[Z2](inner_term)
        denominator = Sum[Y](Sum[Z2](inner_term))
        expected = Fraction(numerator, denominator)
        
        
        self.assert_expr_equal(expected, result)