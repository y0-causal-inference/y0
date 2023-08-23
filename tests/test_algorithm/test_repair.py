"""Data-driven graph repair."""

import unittest
from y0.algorithm.repair import repair_graph
from y0.graph import NxMixedGraph
import pandas as pd

class TestRepair(unittest.TestCase):
    """Test case for data-driven graph repair."""

    def test_repair(self):
        graph: NxMixedGraph = ...
        data:pd.DataFrame = ...
        new_graph = repair_graph(graph=graph, data=data)

        # TODO implement test
        raise NotImplementedError