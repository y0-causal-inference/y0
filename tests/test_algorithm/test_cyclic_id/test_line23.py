"""Tests for the line 23 fix in the cyclic ID algorithm

This module tests:
1. identify_district_variables_cyclic()
2. initial_distribution parameter - an optional parameter to add interventional data
"""



from y0 import graph
from IPython.display import display, Math
from tests.test_algorithm import cases
from y0.examples import napkin
from y0.algorithm.identify.cyclic_id import cyclic_id
from y0.algorithm.identify.cyclic_id import identify_district_variables_cyclic, _get_projected_subgraph
from y0.dsl import W3, X, Y, Z1, Z2, P, Sum, Fraction, Variable, Z
from y0.graph import NxMixedGraph
from y0.algorithm.identify.utils import Unidentifiable
import networkx as nx


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
        When the ancestral set = input variables, return the district probability.
        Expected: P(Y) - no marginzalization needed.
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
        """
        Graph: X -> Y, disconnected Z1
        District T = {Y, Z1}, Target = {Y}
        Expected: Sum[Z1](P(Y, Z1)) - marginalize out Z1
        """
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
        """Base Case 1: Multiple Target variables (|C| > 1)"""
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
        
        # No marginalization, ancestral set = input variables = input district
        self.assert_expr_equal(P(Y, Z1), result)
        
    def test_base_case_2_no_confounding_returns_none(self):
        """Base Case 2: Ancestral set = input district, should return None"""
        
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
        
        # if ancestral set is equal to the input district, should return a FAIL/None
        self.assertIsNone(result)
        
    def test_case_3_triggers_recursion(self):
        """Base Case 3: C ⊂ A ⊂ T triggers recursion with Lemma 4.
        """
        
        
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
        
        graph = NxMixedGraph.from_edges(
            directed=[(Z2, Z1), (Z1, X), (X, Y)],
            undirected=[(Z2, X), (Z2, Y)]
        )
    
        result = cyclic_id(
            graph=graph,
            outcomes={Y},
            interventions={X},
        )    
        
        inner_term = P(X, Y, Z1, Z2) * P(Z2) / P(Z1, Z2)
        numerator = Sum[Z2](inner_term)
        denominator = Sum[Y](Sum[Z2](inner_term))
        expected = Fraction(numerator, denominator)
        
        self.assert_expr_equal(expected, result)
        print(result)
        
   
    def test_acyclic_no_confounding_simple_dag(self):
        """Regression test: Simple DAG with no confounding still works with no errors."""
        
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
        # no confounding = None
        self.assertIsNone(result)
           
    def test_projected_subgraph_bidirected_chain(self):
        
        Za = Variable("Za")
        Zb = Variable("Zb")
        
        graph = NxMixedGraph.from_edges(
            directed=[],
            undirected=[(X, Za), (Za, Zb), (Zb, Y)]
        )
        
        projected = _get_projected_subgraph(graph, frozenset({X, Y}))
        
        self.assertTrue(projected.undirected.has_edge(X, Y))
        
        self.assertNotIn(Za, projected.nodes())
        self.assertNotIn(Zb, projected.nodes())
        
    def test_incorrect_input_c_not_subset_of_t(self):
        """Input Validation: Input variables must be a subset of T"""
        
        graph = NxMixedGraph.from_edges(
            directed=[(X, Y)],
            undirected=[]
        )
        
        with self.assertRaises(nx.NetworkXError):
            identify_district_variables_cyclic(
                input_variables=frozenset({Z1}),
                input_district=frozenset({X, Y}),
                district_probability=P(X, Y),
                graph=graph,
                topo=[X, Y, Z1],
            )
    
    
    def test_bow_arc_unidentifiable(self):
        """Regression test: Bow arc structure is unidentifiable."""
        
        graph = NxMixedGraph.from_edges(
            directed=[(X, Y)],
            undirected=[(X, Y)]
        )
        
        input_district = frozenset({X, Y})
        target = frozenset({Y})
        q_t = P(X, Y)
        
        result = identify_district_variables_cyclic(
            input_variables=target,
            input_district=input_district,
            district_probability=q_t,
            graph=graph,
            topo=[X, Y]
        )
        
        self.assertIsNone(result)
        
        
    def test_tian_pearl_figure_9_deep_recursion(self):
        """Uses figure 9 of the Tian & Pearl algorithm."""
        
        
        W1 = Variable("W1")
        W2 = Variable("W2")
        W3 = Variable("W3")
        W4 = Variable("W4")
        W5 = Variable("W5")
    
        graph = NxMixedGraph.from_edges(
            directed=[(W1, W2), (W2, X), (W3, W4), (W4, X), (X, Y)],
            undirected=[(W1, W3), (W3, W5), (W4, W5), (W2, W3), (W1, X), (W1, Y)]
        )
        
        topo = [W3, W5, W4, W1, W2, X, Y]
        input_district = frozenset(graph.nodes())
        target = frozenset({Y})
        
        # Initial distribution Q[V] = P(V)
        q_v = P(W3, W5, W4, W1, W2, X, Y)

        result = identify_district_variables_cyclic(
            input_variables=target,
            input_district=input_district,
            district_probability=q_v,
            graph=graph,
            topo=topo
        )

        # Expected: Non-None result (identifiable)
        # Should be a Fraction (ratio formula from 3 levels of recursion)
        self.assertIsNotNone(result, "Figure 9 should be identifiable")
        self.assertIsInstance(result, Fraction, "Should produce ratio formula")         
        
class TestInitialDistributionParameter(cases.GraphTestCase):
    
    def test_ecoli_bow_arc_unidentifiable_without_intervention(self):
        """E.coli: Bow-arc (soxR → soxS with soxR ↔ soxS) is unidentifiable.
    
        Direct confounding between intervention and outcome makes the query fail.
        """
    
        fur = Variable("fur")
        fnr = Variable("fnr")
        soxR = Variable("soxR")
        oxyR = Variable("oxyR")
        soxS = Variable("soxS")
    
        # E.coli network with bow-arc: soxR → soxS AND soxR ↔ soxS
        graph = NxMixedGraph.from_edges(
            directed=[
                (fur, soxR),
                (soxR, oxyR),
                (oxyR, fur),
                (fnr, fnr),
                (fnr, soxR),
                (soxR, soxS),  # Directed
            ],
            undirected=[
                (soxR, soxS),  # Confounded (bow-arc!)
            ]
        )
    
        # Query P(soxS | do(soxR)) - should FAIL due to bow-arc
        with self.assertRaises(Unidentifiable):
                cyclic_id(
                    graph=graph,
                    outcomes={soxS},
                    interventions={soxR},
                )
   
    
    def test_broken_bow_arc_different_interventions(self):
        """Bow-arc: P(Y|do(X)) fails observationally, succeeds with P[do(Z)](V).
    
        Graph: Z → X, Z → Y, X → Y (Z is the confounder)
        Intervention on Z breaks the X-Y confounding path.
        """
 
        
        # Z confounds X and Y
        graph = NxMixedGraph.from_edges(
            directed=[(Z, X), (Z, Y), (X, Y)],
            undirected=[]
        )
    
        # Observational: Should succeed (Z is observed, back-door criterion)
        # But let's test with interventional data
    
        # With P[do(Z)](V): breaks confounding!
        interventional_dist = P[Z](X, Y, Z)
    
        result = cyclic_id(
            graph=graph,
            outcomes={Y},
            interventions={X},  # ← Different from Z!
            base_distribution=interventional_dist,
        )
    
        self.assertIsNotNone(result)
        print(f"\nResult with do(Z) data: {result}")
        print(f"LaTeX: {result.to_latex()}")
    
        # Should be non-trivial (not just "1")
        variables = result.get_variables()
        self.assertTrue(len(variables) > 0, "Result should contain variables")
        
    def test_manual_vs_automatic_graph_mutilation(self):
        """Verify automatic mutilation matches manual graph surgery."""
    
  
        # Original graph with Z
        graph_with_z = NxMixedGraph.from_edges(
            directed=[(Z, X), (Z, Y), (X, Y)],
            undirected=[]
        )
    
        # Manually mutilated graph (remove Z)
        graph_without_z = NxMixedGraph.from_edges(
            directed=[(X, Y)],  # Z edges removed!
            undirected=[]
        )
    
        # METHOD 1: Automatic (with interventional data)
        interventional_dist = P[Z](X, Y, Z)
        result_automatic = cyclic_id(
            graph=graph_with_z,
            outcomes={Y},
            interventions={X},
            base_distribution=interventional_dist,
        )
    
        # METHOD 2: Manual (query on pre-mutilated graph with observational data)
        # Use P(X,Y) as if Z never existed
        result_manual = cyclic_id(
            graph=graph_without_z,
            outcomes={Y},
            interventions={X},
            # No base_distribution - using observational P(X,Y)
        )
    
        print(f"\nAutomatic (with P[do(Z)](V)): {result_automatic}")
        print(f"Manual (pre-mutilated graph):  {result_manual}")
    
        # They should be equivalent!
        self.assertEqual(
            result_automatic, 
            result_manual,
            "Automatic mutilation should match manual graph surgery"
        )
    def test_scc_cycle_breaker_with_interventional_data(self):
            """SCC cycle breaker: X→Y→Z→X unidentifiable, identifiable with P[do(Z)](V)."""
            from y0.algorithm.identify.cyclic_id import cyclic_id
            from y0.algorithm.identify import Unidentifiable
        
            # Simple 3-node cycle
            graph = NxMixedGraph.from_edges(
                directed=[(X, Y), (Y, Z), (Z, X)]
            )
            
            # Observational: FAIL (all in same SCC)
            with self.assertRaises(Unidentifiable):
                cyclic_id(graph, outcomes={Y}, interventions={X})
            
            # With do(Z) data: breaks cycle!
            int_data = P[Z](X, Y, Z)
            result = cyclic_id(
                graph, 
                outcomes={Y}, 
                interventions={X}, 
                base_distribution=int_data
            )
            
            self.assertIsNotNone(result)
            print(f"\nSCC cycle break result: {result}")


    def test_mediated_bow_arc_break_with_interventional_data(self):
            """Mediated bow-arc: X→Z→Y with X↔Z unidentifiable, identifiable with P[do(Z)](V)."""
            from y0.algorithm.identify.cyclic_id import cyclic_id
            from y0.algorithm.identify import Unidentifiable
            
            # X→Z→Y with X↔Z
            graph = NxMixedGraph.from_edges(
                directed=[(X, Z), (Z, Y)],
                undirected=[(X, Z)]
            )
            
            # Observational: FAIL (bow-arc on mediator)
            with self.assertRaises(Unidentifiable):
                cyclic_id(graph, outcomes={Y}, interventions={X})
            
            # With do(Z) data: breaks bow-arc!
            int_data = P[Z](X, Y, Z)
            result = cyclic_id(
                graph,
                outcomes={Y},
                interventions={X},
                base_distribution=int_data
            )
            
            self.assertIsNotNone(result)
            print(f"\nMediated bow-arc break result: {result}")
            
            
    def test_overlapping_interventions_raises_error(self):
        """Verify error when J ∩ W ≠ ∅"""
        graph = NxMixedGraph.from_edges(directed=[(X, Y), (Y, Z)])
    
        # Try to query P(Z|do(X)) with P[do(X)](V) - X in both!
        with self.assertRaises(ValueError) as cm:
            cyclic_id(
                graph,
                outcomes={Z},
                interventions={X},  # Same as intervention in data!
                base_distribution=P[X](X, Y, Z)
            )
    
        self.assertIn("must be disjoint", str(cm.exception))