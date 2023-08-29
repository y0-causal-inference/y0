"""Tests for estimation workflows and tools."""
import logging
import unittest

import pandas as pd

from tests.constants import NAPKIN_OBSERVATIONAL_PATH
from y0.algorithm.estimation import (
    _ananke_compute_effect,
    df_covers_graph,
    estimate_ate,
    get_primal_ipw_ace,
    get_state_space_map,
    is_a_fixable,
    is_markov_blanket_shielded,
    is_p_fixable,
)
from y0.dsl import Variable, X, Y
from y0.examples import SARS_SMALL_GRAPH, frontdoor, napkin, napkin_example
from y0.graph import NxMixedGraph


class TestEstimation(unittest.TestCase):
    """A test case for estimation workflows and tools."""

    def assert_mb_shielded(self, graph: NxMixedGraph):
        """Assert the graph is mb-shielded."""
        self.assertTrue(graph.to_admg().mb_shielded())
        self.assertTrue(is_markov_blanket_shielded(graph))

    def assert_mb_unshielded(self, graph: NxMixedGraph):
        """Assert the graph is not mb-shielded."""
        self.assertFalse(graph.to_admg().mb_shielded())
        self.assertFalse(is_markov_blanket_shielded(graph))

    def test_is_mb_shielded(self):
        """Test checking for markov blanket shielding."""
        # Adapted from https://gitlab.com/causal/ananke/-/blob/dev/tests/estimation/test_automated_if.py#L80-92
        graph_unshielded = NxMixedGraph.from_str_edges(
            directed=[("T", "M"), ("M", "L"), ("L", "Y")], undirected=[("M", "Y")]
        )
        with self.subTest(name="Graph 1"):
            self.assert_mb_unshielded(graph_unshielded)

        # Second test: Napkin model
        with self.subTest(name="Napkin"):
            self.assert_mb_unshielded(napkin)

        # Third test
        with self.subTest(name="Graph 3"):
            self.assert_mb_unshielded(SARS_SMALL_GRAPH)

        # Fourth test
        graph_4 = NxMixedGraph.from_str_edges(
            directed=[
                ("SOS", "Ras"),
                ("Ras", "PI3K"),
                ("Ras", "Raf"),
                ("PI3K", "AKT"),
                ("AKT", "Raf"),
            ],
            undirected=[("SOS", "PI3K")],
        )
        with self.subTest(name="Graph 4"):
            self.assert_mb_shielded(graph_4)

        # Fifth test
        graph_5 = NxMixedGraph.from_str_edges(
            directed=[
                ("Z1", "X"),
                ("X", "M1"),
                ("M1", "Y"),
                ("Z1", "Z2"),
                ("Z2", "Z3"),
                ("Z3", "Y"),
                ("Z2", "M1"),
            ],
            undirected=[("Z1", "X"), ("Z2", "M1")],
        )
        with self.subTest(name="Graph 5"):
            self.assert_mb_unshielded(graph_5)

        # Sixth test
        graph_6 = NxMixedGraph.from_str_edges(
            directed=[("Z1", "X"), ("X", "M1"), ("M1", "Y"), ("Z1", "Y")]
        )
        with self.subTest(name="Graph 6"):
            self.assert_mb_shielded(graph_6)

        # Seventh test
        graph_7 = NxMixedGraph.from_str_edges(directed=[("A", "B"), ("B", "C")])
        with self.subTest(name="Graph 6"):
            self.assert_mb_shielded(graph_7)

    def test_is_a_fixable(self):
        """Test checking for a-fixability.

        Graphs 9, 10, and 11 are from https://gitlab.com/causal/ananke/-/blob/dev/tests/\
        estimation/test_counterfactual_mean.py?ref_type=heads#L20-47
        """
        graph_1 = NxMixedGraph.from_str_edges(
            directed=[("T", "M"), ("M", "L"), ("L", "Y")], undirected=[("M", "Y")]
        )
        treatment_1 = Variable("T")
        self.assertTrue(is_a_fixable(graph_1, treatment_1))

        self.assertFalse(is_a_fixable(napkin, X))

        graph_3 = NxMixedGraph.from_str_edges(
            directed=[
                ("ADAM17", "EGFR"),
                ("ADAM17", "TNF"),
                ("ADAM17", "Sil6r"),
                ("EGFR", "cytok"),
                ("TNF", "cytok"),
                ("Sil6r", "IL6STAT3"),
                ("IL6STAT3", "cytok"),
            ],
            undirected=[
                ("ADAM17", "cytok"),
                ("ADAM17", "Sil6r"),
                ("EGFR", "TNF"),
                ("EGFR", "IL6STAT3"),
            ],
        )
        treatment_3 = Variable("EGFR")
        self.assertTrue(is_a_fixable(graph_3, treatment_3))

        graph_4 = NxMixedGraph.from_str_edges(
            directed=[
                ("SOS", "Ras"),
                ("Ras", "PI3K"),
                ("Ras", "Raf"),
                ("PI3K", "AKT"),
                ("AKT", "Raf"),
            ],
            undirected=[("SOS", "PI3K")],
        )
        treatment_4 = Variable("PI3K")
        self.assertTrue(is_a_fixable(graph_4, treatment_4))

        graph_5 = NxMixedGraph.from_str_edges(
            directed=[
                ("Z1", "X"),
                ("X", "M1"),
                ("M1", "Y"),
                ("Z1", "Z2"),
                ("Z2", "Z3"),
                ("Z3", "Y"),
                ("Z2", "M1"),
            ],
            undirected=[("Z1", "X"), ("Z2", "M1")],
        )
        self.assertTrue(is_a_fixable(graph_5, X))

        graph_6 = NxMixedGraph.from_str_edges(
            directed=[("Z1", "X"), ("X", "M1"), ("M1", "Y"), ("Z1", "Y")], undirected=[]
        )
        self.assertTrue(is_a_fixable(graph_6, X))

        graph_7 = NxMixedGraph.from_str_edges(directed=[("A", "B"), ("B", "C")], undirected=[])
        treatment_7 = Variable("A")
        self.assertTrue(is_a_fixable(graph_7, treatment_7))

        graph_8 = graph_4
        treatment_8 = Variable("SOS")
        self.assertFalse(is_a_fixable(graph_8, treatment_8))

    def test_is_p_fixable(self):
        """Test checking for p-fixability.

        .. seealso:: https://gitlab.com/causal/ananke/-/blob/dev/tests/\
        estimation/test_counterfactual_mean.py?ref_type=heads#L151-212
        """
        graph_1 = NxMixedGraph.from_str_edges(
            directed=[("T", "M"), ("M", "L"), ("L", "Y")], undirected=[("M", "Y")]
        )
        treatment_1 = Variable("T")
        with self.subTest(name="Graph 1"):
            self.assertTrue(is_p_fixable(graph_1, treatment_1))

        with self.subTest(name="Napkin"):
            self.assertFalse(is_p_fixable(napkin, X))

        treatment_3 = Variable("EGFR")
        with self.subTest(name="Graph 3"):
            self.assertTrue(is_p_fixable(SARS_SMALL_GRAPH, treatment_3))

        graph_4 = NxMixedGraph.from_str_edges(
            directed=[
                ("SOS", "Ras"),
                ("Ras", "PI3K"),
                ("Ras", "Raf"),
                ("PI3K", "AKT"),
                ("AKT", "Raf"),
            ],
            undirected=[("SOS", "PI3K")],
        )
        treatment_4 = Variable("PI3K")
        with self.subTest(name="Graph 4"):
            self.assertTrue(is_p_fixable(graph_4, treatment_4))

        graph_5 = NxMixedGraph.from_str_edges(
            directed=[
                ("Z1", "X"),
                ("X", "M1"),
                ("M1", "Y"),
                ("Z1", "Z2"),
                ("Z2", "Z3"),
                ("Z3", "Y"),
                ("Z2", "M1"),
            ],
            undirected=[("Z1", "X"), ("Z2", "M1")],
        )
        with self.subTest(name="Graph 5"):
            self.assertTrue(is_p_fixable(graph_5, X))

        graph_6 = NxMixedGraph.from_str_edges(
            directed=[("Z1", "X"), ("X", "M1"), ("M1", "Y"), ("Z1", "Y")], undirected=[]
        )
        with self.subTest(name="Graph 6"):
            self.assertTrue(is_p_fixable(graph_6, X))

        graph_7 = NxMixedGraph.from_str_edges(directed=[("A", "B"), ("B", "C")], undirected=[])
        treatment_7 = Variable("A")
        with self.subTest(name="Graph 7"):
            self.assertTrue(is_p_fixable(graph_7, treatment_7))

        graph_8 = graph_4
        treatment_8 = Variable("SOS")
        with self.subTest(name="Graph 8"):
            self.assertTrue(is_p_fixable(graph_8, treatment_8))

        graph_9 = NxMixedGraph.from_str_edges(
            directed=[
                ("C", "T"),
                ("C", "M"),
                ("C", "L"),
                ("C", "Y"),
                ("T", "M"),
                ("M", "L"),
                ("M", "Y"),
                ("L", "Y"),
            ],
            undirected=[("T", "L"), ("T", "Y")],
        )
        treatment_9 = Variable("T")
        with self.subTest(name="Graph 9"):
            self.assertTrue(is_p_fixable(graph_9, treatment_9))

        graph_10 = NxMixedGraph.from_str_edges(
            directed=[
                ("C", "T"),
                ("C", "M"),
                ("C", "L"),
                ("C", "Y"),
                ("T", "M"),
                ("M", "L"),
                ("T", "Y"),
                ("L", "Y"),
            ],
            undirected=[("T", "L"), ("M", "Y")],
        )
        treatment_10 = Variable("T")
        with self.subTest(name="Graph 10"):
            self.assertTrue(is_p_fixable(graph_10, treatment_10))

        graph_11 = NxMixedGraph.from_str_edges(
            directed=[
                ("C1", "T"),
                ("C1", "L"),
                ("C2", "M"),
                ("C2", "L"),
                ("C2", "Y"),
                ("T", "M"),
                ("M", "L"),
                ("L", "Y"),
            ],
            undirected=[("Z1", "C1"), ("Z2", "C2"), ("T", "L")],
        )
        treatment_11 = Variable("T")
        with self.subTest(name="Graph 11"):
            self.assertTrue(is_p_fixable(graph_11, treatment_11))

        graph_12 = NxMixedGraph.from_str_edges(
            directed=[
                ("C2", "T"),
                ("C2", "M1"),
                ("C2", "M2"),
                ("C2", "Y"),
                ("C1", "T"),
                ("C1", "M2"),
                ("T", "M1"),
                ("M1", "M2"),
                ("M2", "Y"),
                ("T", "Y"),
            ],
            undirected=[("T", "M2"), ("M1", "Y")],
        )
        treatment_12 = Variable("T")
        with self.subTest(name="Graph 12"):
            self.assertTrue(is_p_fixable(graph_12, treatment_12))

    def test_data_covers_graph(self):
        """Test the data coverage utility."""
        df = pd.read_csv(NAPKIN_OBSERVATIONAL_PATH, sep="\t")
        self.assertTrue(df_covers_graph(graph=napkin, df=df))
        self.assertFalse(df_covers_graph(graph=frontdoor, df=df))

    @unittest.skip(reason="Turn this test on before finishing the PR")
    def test_estimate_ate(self):
        """Run a simple test for ATE on the napkin graph."""
        df = pd.read_csv(NAPKIN_OBSERVATIONAL_PATH, sep="\t")
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
