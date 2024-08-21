"""Tests for estimation workflows and tools."""

import unittest

import pandas as pd

from y0.algorithm.estimation import ananke_average_causal_effect, df_covers_graph
from y0.algorithm.estimation.estimators import get_primal_ipw_ace, get_state_space_map
from y0.dsl import Variable
from y0.examples import examples, frontdoor, napkin, napkin_example


class TestEstimation(unittest.TestCase):
    """A test case for estimation workflows and tools."""

    def test_data_covers_graph(self):
        """Test the data coverage utility."""
        df = napkin_example.generate_data(1000)
        self.assertTrue(df_covers_graph(graph=napkin, data=df))
        self.assertFalse(df_covers_graph(graph=frontdoor, data=df))

    def test_beta_primal(self):
        """Test beta primal on example graphs that have data generators."""
        for example in examples:
            if not example.generate_data:
                continue
            queries = [
                query
                for query in example.example_queries or []
                if (
                    len(query.treatments) != 1
                    or len(query.outcomes) != 1
                    or len(query.conditions) > 0
                )
            ]
            if not queries:
                continue
            data = example.generate_data(1000)
            for example_query in queries:
                with self.subTest(name=example.name, query=str(example_query)):
                    treatment = example_query.treatments[0]
                    outcome = example_query.outcomes[0]
                    ananke_results = ananke_average_causal_effect(
                        graph=example.graph,
                        treatment=treatment,
                        outcome=outcome,
                        data=data,
                        estimator="p-ipw",
                        bootstraps=0,  # be explicit not to do bootstrapping
                    )
                    y0_results = get_primal_ipw_ace(
                        graph=example.graph, data=data, treatment=treatment, outcome=outcome
                    )
                    self.assertAlmostEqual(ananke_results, y0_results)

    def test_get_state_space_map(self):
        """Test the state space map creation for the variables in the data."""
        data = pd.DataFrame.from_dict(
            data={"test1": [0, 0, 1, 0, 1], "test2": [1, 2, 3, 4, 5], "test3": [0, 1, 2, 3, 4]}
        )
        computed_state_space_map = get_state_space_map(data)
        expected_state_space_map = {
            Variable("test1"): "binary",
            Variable("test2"): "continuous",
            Variable("test3"): "continuous",
        }
        self.assertEqual(computed_state_space_map, expected_state_space_map)
