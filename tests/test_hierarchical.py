"""Test hierarchical graphical model creation and manipulation."""

import unittest

from y0.dsl import A, B, C, D, E, F, G, M, U, Y, Z
from y0.examples import hierarchical as hcm_examples
from y0.examples.hierarchical import (
    get_compl_subgraph_hcm,
    get_confounder_hcgm,
    get_confounder_hscm,
    get_instrument_hcgm,
    get_instrument_hcm,
    get_instrument_hscm,
    get_instrument_subunit_graph,
    get_school_confounder_interference_hcgm,
    get_school_confounder_interference_hcm,
    get_school_confounder_interference_hscm,
)
from y0.graph import NxMixedGraph
from y0.hierarchical import (
    HierarchicalCausalModel,
    HierarchicalStructuralCausalModel,
    QVariable,
    _create_qvar,
    augment_collapsed_model,
    augment_from_mechanism,
    augmentation_mechanism,
    get_ancestors,
    marginalize_augmented_model,
)

Q_A = QVariable(name="A")
Q_Y = QVariable(name="Y")
Q_Y_A = QVariable(name="Y", parents=frozenset([A]))
Q_A_Z = QVariable(name="A", parents=frozenset([Z]))
Q_Z = QVariable(name="Z")
Q_Y_A_Z = QVariable(name="Y", parents=frozenset([A, Z]))

confounder_hcgm = get_confounder_hcgm()
confounder_hscm = get_confounder_hscm()
confounder_interference_hcm = get_school_confounder_interference_hcm()
confounder_interference_hcgm = get_school_confounder_interference_hcgm()
confounder_interference_hscm = get_school_confounder_interference_hscm()
instrument_hcm = get_instrument_hcm()
instrument_subunit_graph = get_instrument_subunit_graph()
instrument_hcgm = get_instrument_hcgm()
instrument_hscm = get_instrument_hscm()
compl_subgraph_hcm = get_compl_subgraph_hcm()


class TestStructure(unittest.TestCase):
    """Test copy HCM for Confounder fixture."""

    def assert_hcm_equal(
        self, expected: HierarchicalCausalModel, actual: HierarchicalCausalModel
    ) -> None:
        """Test all parts of two HCMs are equal."""
        self.assertEqual(
            expected.get_observed(), actual.get_observed(), msg="observed are different"
        )
        self.assertEqual(
            expected.get_unobserved(), actual.get_unobserved(), msg="unobserved are different"
        )
        self.assertEqual(expected.get_units(), actual.get_units(), msg="units are different")
        self.assertEqual(
            expected.get_subunits(), actual.get_subunits(), msg="subunits are different"
        )
        self.assertEqual(set(expected.edges()), set(actual.edges()), msg="edges are different")

    def assert_hscm_equal(
        self, expected: HierarchicalStructuralCausalModel, actual: HierarchicalStructuralCausalModel
    ) -> None:
        """Test all parts of two HSCMs are equal."""
        self.assertEqual(
            expected.get_observed(), actual.get_observed(), msg="observed are different"
        )
        self.assertEqual(
            expected.get_unobserved(), actual.get_unobserved(), msg="unobserved are different"
        )
        self.assertEqual(expected.get_units(), actual.get_units(), msg="units are different")
        self.assertEqual(
            expected.get_subunits(), actual.get_subunits(), msg="subunits are different"
        )
        self.assertEqual(set(expected.edges()), set(actual.edges()), msg="edges are different")
        self.assertEqual(expected.get_exogenous_noise(), actual.get_exogenous_noise())

    def test_instrument_hcm_from_list(self) -> None:
        """Test for correct observed variables."""
        hcm = HierarchicalCausalModel.from_lists(
            observed_subunits=[A, Z],
            observed_units=[Y],
            unobserved_units=[U],
            edges=[(Z, A), (A, Y), (U, Y), (U, A)],
        )
        self.assert_hcm_equal(instrument_hcm, hcm)

    def test_confounder_inference(self) -> None:
        """Pytest fixture from lists, to compre against 'by-hand' fixture."""
        hcm = HierarchicalCausalModel.from_lists(
            observed_subunits=[A, Y],
            observed_units=[Z],
            unobserved_units=[U],
            edges=[(U, A), (A, Y), (U, Y), (A, Z), (Z, Y)],
        )
        self.assert_hcm_equal(confounder_interference_hcm, hcm)

    def test_list_construction(self) -> None:
        """Test for correct observed variables."""
        hcm = HierarchicalCausalModel.from_lists(
            observed_subunits=[A, Y],
            unobserved_units=[U],
            edges=[(U, A), (A, Y), (U, Y)],
        )
        self.assert_hcm_equal(hcm_examples.confounder_hcm, hcm)

    def test_copy(self) -> None:
        """Test copy HCM for Confounder fixture."""
        for hcm in [
            hcm_examples.confounder_hcm,
            confounder_interference_hcm,
            instrument_hcm,
        ]:
            self.assert_hcm_equal(hcm, hcm.copy_hcm())

    def test_get_subunit_graph(self) -> None:
        """Test getting the subunit graph."""
        expected = instrument_subunit_graph._graph
        actual = instrument_hcm.get_subunit_graph()
        self.assertEqual(set(expected.nodes()), set(actual.nodes()))
        self.assertEqual(set(expected.edges()), set(actual.edges()))

    def test_get_observed(self) -> None:
        """Test getting observed nodes."""
        # Test observed variables in Confounder HCM fixture."""
        self.assertEqual({A, Y}, hcm_examples.confounder_hcm.get_observed())

        # Test observed variables in Confounder Interference HCM fixture."""
        self.assertEqual({A, Y, Z}, confounder_interference_hcm.get_observed())

        # Test observed variables in Instrument HCM fixture
        self.assertEqual({A, Y, Z}, instrument_hcm.get_observed())

    def test_get_unobserved(self) -> None:
        """Test unobserved variables in Confounder HCM fixture."""
        self.assertEqual({U}, hcm_examples.confounder_hcm.get_unobserved())
        self.assertEqual({U}, confounder_interference_hcm.get_unobserved())
        self.assertEqual({U}, instrument_hcm.get_unobserved())

    def test_get_units(self) -> None:
        """Test unit variables in Confounder HCM fixture."""
        self.assertEqual({U}, hcm_examples.confounder_hcm.get_units())
        self.assertEqual({U, Z}, confounder_interference_hcm.get_units())
        self.assertEqual({U, Y}, instrument_hcm.get_units())

    def test_get_subunits(self) -> None:
        """Test subunit variables in Confounder HCM fixture."""
        self.assertEqual({A, Y}, hcm_examples.confounder_hcm.get_subunits())
        self.assertEqual({A, Y}, confounder_interference_hcm.get_subunits())
        self.assertEqual({A, Z}, instrument_hcm.get_subunits())

    def test_get_parents(self) -> None:
        """Test getting parents."""
        self.assertEqual({U, A, Z}, confounder_interference_hcm.get_parents(Y))
        self.assertEqual(set(), confounder_interference_hcm.get_parents(U))

    def test_get_ancestors(self) -> None:
        """Test getting ancestors."""
        subgraph = compl_subgraph_hcm.get_subunit_graph()

        # non-direct subunit ancestors are not included
        self.assertEqual(set(), get_ancestors(subgraph, C))
        # non-parent direct subunit ancestors are included
        self.assertEqual({A, B, C}, get_ancestors(subgraph, Y))

    def test_to_hgcm(self) -> None:
        """Test converting to hierarchical causal graphical models (HCGMs)."""
        for hcm, hcgm in [
            (instrument_hcm, instrument_hcgm),
            (hcm_examples.confounder_hcm, confounder_hcgm),
            (confounder_interference_hcm, confounder_interference_hcgm),
        ]:
            self.assert_hcm_equal(hcgm, hcm.to_hcgm())

    def test_to_hscm(self) -> None:
        """Test converting to hierarchical structural causal models (HSCMs)."""
        for hcm, hscm in [
            (hcm_examples.confounder_hcm, confounder_hscm),
            (confounder_interference_hcm, confounder_interference_hscm),
            (instrument_hcm, instrument_hscm),
        ]:
            self.assert_hscm_equal(hscm, hcm.to_hscm())

    def test_to_hcm(self) -> None:
        """Test converting to hierarchical structural causal models (HSCMs)."""
        for hcm, hscm in [
            (hcm_examples.confounder_hcm, confounder_hscm),
            (confounder_interference_hcm, confounder_interference_hscm),
            (instrument_hcm, instrument_hscm),
        ]:
            self.assert_hcm_equal(hscm.to_hcm(), hcm)

    def test_duds(self) -> None:
        """Test case for multiple direct unit descendants."""
        obs_sub = [A, B, C, D, F, G]
        obs_units = [E, M]
        edges = [(B, C), (C, D), (D, E), (B, M), (A, M), (F, G), (M, E)]
        hcm = HierarchicalCausalModel.from_lists(
            observed_subunits=obs_sub, observed_units=obs_units, edges=edges
        )

        self.assertEqual({E, M}, hcm.get_direct_unit_descendants(B))

        # Test that non-direct unit descendants are excluded
        self.assertEqual({M}, hcm.get_direct_unit_descendants(A))

        # Test case for path to unit descendant goes through multiple subunit descendants.
        self.assertEqual({E}, hcm.get_direct_unit_descendants(C))

        # Test case when there are no direct unit descendants.
        self.assertEqual(set(), hcm.get_direct_unit_descendants(F))

    def test_q_variables(self) -> None:
        """Test q-variable construction."""
        # Test generic case with multiple subunit parents
        hcm_1 = HierarchicalCausalModel.from_lists(
            observed_subunits=[A, Y, Z], edges=[(A, Y), (Z, Y)]
        )
        self.assertEqual(Q_Y_A_Z, _create_qvar(hcm_1, Y))

        # Test case when there are no subunit parents
        hcm_2 = confounder_interference_hcm.get_subunit_graph()
        self.assertEqual(Q_A, _create_qvar(hcm_2, A))


class TestADMG(unittest.TestCase):
    """Tests for collapsing to ADMGs."""

    def check_graph_equal(self, expected: NxMixedGraph, actual: NxMixedGraph) -> None:
        """Check two graphs are equal based on their nodes and edges."""
        self.assertEqual(set(expected.nodes()), set(actual.nodes()))
        self.assertEqual(expected.directed.edges(), actual.directed.edges())
        self.assertEqual(expected.undirected.edges(), actual.undirected.edges())

    def test_to_admg(self) -> None:
        """Test conversion to ADMG."""
        for hcm, admg in [
            (hcm_examples.confounder_hcm, hcm_examples.confounder_collapsed_admg),
            (
                confounder_interference_hcm,
                hcm_examples.confounder_interference_collapsed_admg,
            ),
            (instrument_hcm, hcm_examples.instrument_collapsed_admg),
        ]:
            self.check_graph_equal(admg, hcm.to_admg())

    # For Algorithm 2

    def test_augment_confounder_from_mech(self) -> None:
        """Test that augmenting Figure 2 (c) fixture gives Figure 2 (d) fixture."""
        self.check_graph_equal(
            hcm_examples.confounder_augmented_admg,
            augment_from_mechanism(
                hcm_examples.confounder_collapsed_admg,
                Q_Y,
                (Q_A, Q_Y_A),
            ),
        )

    def test_confounder_aug_mech(self) -> None:
        """Test the augmentation mechanism for the confounder HCM."""
        mechanism = augmentation_mechanism(hcm_examples.confounder_hcm.get_subunit_graph(), Q_Y)
        self.assertEqual({Q_A, Q_Y_A}, set(mechanism))

    def test_augment_confounder_interference_from_mech(self) -> None:
        """Test that augmenting Figure 2 (g) fixture gives Figure 2 (h)."""
        self.check_graph_equal(
            hcm_examples.confounder_interference_augmented_admg,
            augment_from_mechanism(
                hcm_examples.confounder_interference_collapsed_admg,
                Q_Y,
                (Q_A, Q_Y_A),
            ),
        )

    def test_confounder_interference_aug_mech(self) -> None:
        """Test the augmentation mechanism for the confounder interence HCM."""
        mechanism = augmentation_mechanism(confounder_interference_hcm.get_subunit_graph(), Q_Y)
        self.assertEqual({Q_A, Q_Y_A}, set(mechanism))

    def test_augment_confounder_interference_no_mech(self) -> None:
        """Test augmenting Figure 2 (g) fixture with no mechanism provided."""
        self.check_graph_equal(
            hcm_examples.confounder_interference_augmented_admg,
            augment_collapsed_model(
                hcm_examples.confounder_interference_collapsed_admg,
                confounder_interference_hcm.get_subunit_graph(),
                Q_Y,
            ),
        )

    def test_augment_confounder_interference_with_mech(self) -> None:
        """Test augmenting Figure 2 (h) fixture with mechanism provided."""
        self.check_graph_equal(
            hcm_examples.confounder_interference_augmented_admg,
            augment_collapsed_model(
                hcm_examples.confounder_interference_collapsed_admg,
                confounder_interference_hcm.get_subunit_graph(),
                Q_Y,
                (Q_A, Q_Y_A),
            ),
        )

    def test_augment_instrument_from_mech(self) -> None:
        """Test that augmenting Figure 2 (k) fixture gives Figure A2 fixture."""
        self.check_graph_equal(
            hcm_examples.instrument_augmented_admg,
            augment_from_mechanism(
                hcm_examples.instrument_collapsed_admg,
                Q_A,
                [Q_Z, Q_A_Z],
            ),
        )

    def test_instrument_aug_mech(self) -> None:
        """Test the augmentation mechanism for the instrument HCM."""
        mechanism = augmentation_mechanism(instrument_hcm.get_subunit_graph(), Q_A)
        self.assertEqual({Q_Z, Q_A_Z}, set(mechanism))

    def test_compl_sub_aug_mech(self) -> None:
        """Test the augmentation mechanism for the complicated subgraph HCM."""
        mechanism = augmentation_mechanism(compl_subgraph_hcm.get_subunit_graph(), Q_Y)
        self.assertEqual(
            set(mechanism),
            {
                QVariable.parse_str("Q^{y|b,c}".upper()),
                QVariable.parse_str("Q^c".upper()),
                QVariable.parse_str("Q^{b|a}".upper()),
                Q_A,
            },
        )

    # For Algorithm 3

    def test_marginalized_instrument(self) -> None:
        """Test that marginalizing the Figure A2 fixture gives the Figure 2 (l) fixture."""
        self.check_graph_equal(
            hcm_examples.instrument_marginalized_admg,
            marginalize_augmented_model(hcm_examples.instrument_augmented_admg, Q_A, [Q_Z]),
        )
