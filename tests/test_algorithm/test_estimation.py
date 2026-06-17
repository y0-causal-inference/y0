"""Tests for estimation workflows and tools."""

import unittest

import numpy as np
import pandas as pd

from y0.algorithm.estimation import ananke_average_causal_effect, df_covers_graph
from y0.algorithm.estimation.estimators import get_primal_ipw_ace, get_state_space_map
from y0.algorithm.estimation.linear_scm import get_sigma_single_door, get_single_door
from y0.dsl import Variable
from y0.examples import examples, frontdoor, napkin, napkin_example
from y0.graph import ANANKE_REQUIRED, NxMixedGraph, is_p_fixable
from y0.simulation import LinearSCM, example_graph

TOLERANCE = 0.1


class TestEstimation(unittest.TestCase):
    """A test case for estimation workflows and tools."""

    def test_data_covers_graph(self):
        """Test the data coverage utility."""
        df = napkin_example.generate_data(1000)
        self.assertTrue(df_covers_graph(graph=napkin, data=df))
        self.assertFalse(df_covers_graph(graph=frontdoor, data=df))

    @ANANKE_REQUIRED
    def test_beta_primal(self):
        """Test beta primal on example graphs that have data generators."""
        usable_examples = []
        for example in examples:
            if example.name in {"Frontdoor / Backdoor", "SARS-CoV-2 Small Graph"}:
                continue  # FIXME something wrong is going on there
            if example.generate_data is None:
                continue
            queries = [
                query
                for query in example.example_queries or []
                if 1 == len(query.treatments)
                and 1 == len(query.outcomes)
                and 0 == len(query.conditions)
                and is_p_fixable(example.graph, next(iter(query.treatments)))
            ]
            if not queries:
                continue
            usable_examples.append(example)
        self.assertNotEqual(
            [], usable_examples, msg="No examples had a simple query and a generator"
        )
        for example in usable_examples:
            data = example.generate_data(1000)
            for query in example.example_queries or []:
                treatment = next(iter(query.treatments))
                outcome = next(iter(query.outcomes))
                with self.subTest(
                    name=example.name, treatment=treatment.name, outcome=outcome.name
                ):
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
                    self.assertAlmostEqual(ananke_results, y0_results, delta=TOLERANCE)

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


class TestGetSingleDoor(unittest.TestCase):
    """Regression tests pinning get_single_door output (safety net for refactor).

    ``get_single_door`` uses back-door adjustment to estimate **total causal
    effects**, not direct edge weights.  For edges with mediators (e.g.
    V4→V6 where V4→V5→V6 is also a path), the expected value is the sum of
    all directed-path contributions, not just the direct edge weight.

    Example graph: V1→V2, V1→V4, V2→V5, V4→V5, V4→V6, V3→V5, V5→V6
    All edge weights = 0.5.  Total causal effects:
      - V4→V6: 0.5 (direct) + 0.5*0.5 (via V5) = 0.75
      - All other edges: 0.5 (no mediators)
    """

    #: Fixed seed for reproducibility
    SEED = 42
    #: Number of samples — large enough for stable estimates
    N = 2000
    #: Tolerance for coefficient comparison
    TOLERANCE = 0.15
    #: Direct edge weight used in the SCM
    EDGE_WEIGHT = 0.5

    def setUp(self) -> None:
        """Generate a fixed dataset from the example graph with known weights."""
        from y0.dsl import V4, V6

        rng = np.random.default_rng(self.SEED)
        # Use fixed weights so the expected values are deterministic
        self.weights = dict.fromkeys(example_graph.directed.edges(), self.EDGE_WEIGHT)

        def _uniform() -> float:
            return float(rng.uniform(-1.0, 1.0))

        scm = LinearSCM(
            example_graph,
            generators=dict.fromkeys(example_graph.nodes(), _uniform),
            weights=self.weights,
        )
        rows = [
            {variable.name: value for variable, value in scm.trial().items()}
            for _ in range(self.N)
        ]
        self.data = pd.DataFrame(rows)

        # Build expected total causal effects.
        # V4->V6 has an indirect path V4->V5->V6 (weight 0.5*0.5=0.25).
        self.expected = dict.fromkeys(example_graph.directed.edges(), self.EDGE_WEIGHT)
        self.expected[V4, V6] = self.EDGE_WEIGHT + self.EDGE_WEIGHT * self.EDGE_WEIGHT

    def test_get_single_door_returns_all_edges(self) -> None:
        """get_single_door should return a coefficient for every directed edge."""
        result = get_single_door(example_graph, self.data)
        expected_edges = set(example_graph.directed.edges())
        self.assertEqual(expected_edges, set(result.keys()))

    def test_get_single_door_coefficients_near_total_effects(self) -> None:
        """get_single_door coefficients should be close to the true total causal effects.

        Note: get_single_door uses back-door adjustment, which estimates total
        causal effects (not direct edge weights).  For V4→V6, the expected
        total effect is 0.75 (direct 0.5 + indirect via V5: 0.5*0.5=0.25).
        """
        result = get_single_door(example_graph, self.data)
        for edge, coef in result.items():
            expected = self.expected[edge]
            with self.subTest(edge=edge):
                self.assertAlmostEqual(
                    coef,
                    expected,
                    delta=self.TOLERANCE,
                    msg=f"Coefficient for {edge} = {coef:.4f}, expected total effect ~{expected}",
                )


class TestGetSigmaSingleDoor(unittest.TestCase):
    """Tests for get_sigma_single_door.

    Covers:
    - On a DAG, both functions identify all edges (same key set).
    - Edges with no admissible Z are omitted from the result.
    - Coefficients are close to the true direct edge weights on the example DAG.

    Note on semantics: ``get_sigma_single_door`` uses sigma-separation admissibility,
    which allows conditioning on mediators (descendants of the treatment that are
    non-descendants of the outcome).  This means it can estimate **direct** causal
    effects, whereas ``get_single_door`` (back-door criterion) estimates **total**
    causal effects.  For edges with mediators (e.g. V4→V6 where V4→V5→V6 is also
    a path), the two functions will disagree: sigma gives the direct effect (~0.5)
    while back-door gives the total effect (~0.75).
    """

    #: Fixed seed for reproducibility
    SEED = 7
    #: Number of samples
    N = 2000
    #: Tolerance for coefficient comparison
    TOLERANCE = 0.20
    #: Direct edge weight used in the SCM
    EDGE_WEIGHT = 0.5

    def setUp(self) -> None:
        """Generate a fixed dataset from the example DAG with known weights."""
        rng = np.random.default_rng(self.SEED)
        self.weights = dict.fromkeys(example_graph.directed.edges(), self.EDGE_WEIGHT)

        def _uniform() -> float:
            return float(rng.uniform(-1.0, 1.0))

        scm = LinearSCM(
            example_graph,
            generators=dict.fromkeys(example_graph.nodes(), _uniform),
            weights=self.weights,
        )
        rows = [
            {variable.name: value for variable, value in scm.trial().items()}
            for _ in range(self.N)
        ]
        self.data = pd.DataFrame(rows)

    def test_dag_identifies_all_edges(self) -> None:
        """On a DAG, get_sigma_single_door should identify all directed edges."""
        sigma_result = get_sigma_single_door(example_graph, self.data)
        expected_edges = set(example_graph.directed.edges())
        self.assertEqual(expected_edges, set(sigma_result.keys()))

    def test_dag_coefficients_near_direct_edge_weights(self) -> None:
        """get_sigma_single_door coefficients should be close to the true direct edge weights.

        Unlike get_single_door (back-door, total effects), sigma-single-door
        conditions on mediators when available, recovering direct edge weights.
        For V4→V6, sigma finds {V5} as an admissible set (V5 is a non-descendant
        of V6), which blocks the indirect path V4→V5→V6 and recovers the direct
        weight ~0.5.
        """
        result = get_sigma_single_door(example_graph, self.data)
        for edge, coef in result.items():
            with self.subTest(edge=edge):
                self.assertAlmostEqual(
                    coef,
                    self.EDGE_WEIGHT,
                    delta=self.TOLERANCE,
                    msg=f"sigma-coefficient for {edge} = {coef:.4f}, expected direct weight ~{self.EDGE_WEIGHT}",
                )

    def test_non_identifiable_edge_omitted(self) -> None:
        """An edge with no sigma-admissible Z should be absent from the result.

        2-cycle X<->Y: neither X->Y nor Y->X is identifiable via single-door.
        """
        cx = Variable("CX")
        cy = Variable("CY")
        cycle_graph = NxMixedGraph.from_edges(directed=[(cx, cy), (cy, cx)])
        # Generate trivial data (values don't matter — the graph check fires first)
        data = pd.DataFrame({"CX": [0.0, 1.0, 2.0], "CY": [1.0, 2.0, 3.0]})
        result = get_sigma_single_door(cycle_graph, data)
        self.assertEqual({}, result, msg="2-cycle edges should be non-identifiable")
