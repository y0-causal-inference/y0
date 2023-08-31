"""Tests for estimation workflows and tools."""

import unittest

import pandas as pd

from y0.algorithm.estimation import (
    _ananke_compute_effect,
    df_covers_graph,
    estimate_ate,
    get_primal_ipw_ace,
    get_state_space_map,
)
from y0.dsl import Variable, X, Y
from y0.examples import frontdoor, napkin, napkin_example


class TestEstimation(unittest.TestCase):
    """A test case for estimation workflows and tools."""

    def test_data_covers_graph(self):
        """Test the data coverage utility."""
        df = napkin_example.generate_data(1000)
        self.assertTrue(df_covers_graph(graph=napkin, df=df))
        self.assertFalse(df_covers_graph(graph=frontdoor, df=df))

    @unittest.skip(reason="Turn this test on before finishing the PR")
    def test_estimate_ate(self):
        """Run a simple test for ATE on the napkin graph."""
        df = napkin_example.generate_data(1000)
        expected_result = 0.0005
        result = estimate_ate(graph=napkin, data=df, treatment=X, outcome=Y)
        self.assertAlmostEqual(expected_result, result, delta=1e-5)

    def test_beta_primal(self):
        """Test beta primal on the Napkin graph."""
        example = napkin_example  # FIXME need one that is p-fixable

        data = example.generate_data(1000)
        ananke_results = _ananke_compute_effect(
            graph=napkin, treatment=X, outcome=Y, data=data, estimator="p-ipw"
        )
        y0_results = get_primal_ipw_ace(graph=napkin, data=data, treatment=X, outcome=Y)
        self.assertAlmostEqual(ananke_results, y0_results, delta=0.1)

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
