import unittest

from y0.algorithm.estimation import (
    is_a_fixable,
    is_markov_blanket_shielded,
    is_p_fixable,
)
from y0.examples import napkin
from y0.graph import NxMixedGraph


class TestEstimation(unittest.TestCase):
    def test_is_mb_shielded(self):
        # Adapted from https://gitlab.com/causal/ananke/-/blob/dev/tests/estimation/test_automated_if.py#L80-92
        graph_unshielded = NxMixedGraph.from_str_edges(
            directed=[("T", "M"), ("M", "L"), ("L", "Y")], undirected=[("M", "Y")]
        )
        # use Ananke method for sanity check
        self.assertFalse(graph_unshielded.to_admg().mb_shielded())
        # test our method
        self.assertFalse(is_markov_blanket_shielded(graph_unshielded))

        # Second test: Napkin model
        graph_napkin = napkin.graph
        # use Ananke method for sanity check
        self.assertFalse(graph_napkin.to_admg().mb_shielded())
        # test our method
        self.assertFalse(is_markov_blanket_shielded(graph_napkin))

        # Third test
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
        # use Ananke method for sanity check
        self.assertFalse(graph_3.to_admg().mb_shielded())
        # test our method
        self.assertFalse(is_markov_blanket_shielded(graph_3))

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
        # use Ananke method for sanity check
        self.assertTrue(graph_4.to_admg().mb_shielded())
        # test our method
        self.assertTrue(is_markov_blanket_shielded(graph_4))

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
        # use Ananke method for sanity check
        self.assertFalse(graph_5.to_admg().mb_shielded())
        # test our method
        self.assertFalse(is_markov_blanket_shielded(graph_5))

        # Sixth test
        graph_6 = NxMixedGraph.from_str_edges(
            directed=[("Z1", "X"), ("X", "M1"), ("M1", "Y"), ("Z1", "Y")], undirected=[]
        )
        # use Ananke method for sanity check
        self.assertTrue(graph_6.to_admg().mb_shielded())
        # test our method
        self.assertTrue(is_markov_blanket_shielded(graph_6))

        # Seventh test
        graph_7 = NxMixedGraph.from_str_edges(directed=[("A", "B"), ("B", "C")], undirected=[])
        # use Ananke method for sanity check
        self.assertTrue(graph_7.to_admg().mb_shielded())
        # test our method
        self.assertTrue(is_markov_blanket_shielded(graph_7))

    def test_is_a_fixable(self):
        """Test checking for a-fixability."""
        # see examples at https://gitlab.com/causal/ananke/-/blob/dev/tests/estimation/test_counterfactual_mean.py?ref_type=heads#L20-47
        graph_1 = NxMixedGraph.from_str_edges(
            directed=[("T", "M"), ("M", "L"), ("L", "Y")], undirected=[("M", "Y")]
        )
        treatment_1 = "T"
        self.assertTrue(is_a_fixable(graph_1, treatment_1))

        graph_2 = napkin.graph
        treatment_2 = "X"
        self.assertFalse(is_a_fixable(graph_2, treatment_2))

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
        treatment_3 = "EGFR"
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
        treatment_4 = "PI3K"
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
        treatment_5 = "X"
        self.assertTrue(is_a_fixable(graph_5, treatment_5))

        graph_6 = NxMixedGraph.from_str_edges(
            directed=[("Z1", "X"), ("X", "M1"), ("M1", "Y"), ("Z1", "Y")], undirected=[]
        )
        treatment_6 = "X"
        self.assertTrue(is_a_fixable(graph_6, treatment_6))

        graph_7 = NxMixedGraph.from_str_edges(directed=[("A", "B"), ("B", "C")], undirected=[])
        treatment_7 = "A"
        self.assertTrue(is_a_fixable(graph_7, treatment_7))

        graph_8 = graph_4
        treatment_8 = "SOS"
        self.assertFalse(is_a_fixable(graph_8, treatment_8))

    def test_is_p_fixable(self):
        """Test checking for p-fixability."""
        # see examples at https://gitlab.com/causal/ananke/-/blob/dev/tests/estimation/test_counterfactual_mean.py?ref_type=heads#L151-212
        graph_1 = ...
        treatment_1 = ...
        self.assertFalse(is_p_fixable(graph_1, treatment_1))

        graph_2 = ...
        treatment_2 = ...
        self.assertTrue(is_p_fixable(graph_2, treatment_2))
