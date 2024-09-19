"""Test hierarchical graphical model creation and manipulation."""

import pygraphviz as pgv
import pytest

from y0.dsl import Variable
from y0.graph import NxMixedGraph
from y0.hierarchical import (
    HCM_from_lists,
    augment_collapsed_model,
    collapse_HCM,
    convert_to_HCGM,
    copy_HCM,
    create_Qvar,
    direct_unit_descendents,
    get_observed,
    get_subunits,
    get_units,
    get_unobserved,
    marginalize_augmented_model,
    parents,
)


@pytest.fixture
def confounder_HCM_pygraphviz():
    """Pytest fixture for the Confounder HCM in Figure 2 (a)."""
    HCM = pgv.AGraph(directed=True)
    HCM.add_node("A", style="filled", color="lightgrey")
    HCM.add_node("Y", style="filled", color="lightgrey")
    HCM.add_edge("U", "A")
    HCM.add_edge("A", "Y")
    HCM.add_edge("U", "Y")
    HCM.add_subgraph(["A", "Y"], name="cluster_subunits", style="dashed", label="m")
    return HCM


@pytest.fixture
def confounder_HCGM_pygraphviz():
    """Pytest fixture for the Confounder HCGM in Figure 2 (b)."""
    HCM = pgv.AGraph(directed=True)
    HCM.add_node("A", style="filled", color="lightgrey")
    HCM.add_node("Y", style="filled", color="lightgrey")
    HCM.add_node("Q_a", style="filled", color="lightgrey")
    HCM.add_node("Q_{y|a}", style="filled", color="lightgrey")
    HCM.add_edge("U", "Q_a")
    HCM.add_edge("Q_a", "A")
    HCM.add_edge("A", "Y")
    HCM.add_edge("U", "Q_{y|a}")
    HCM.add_edge("Q_{y|a}", "Y")
    HCM.add_subgraph(["A", "Y"], name="cluster_subunits", label="m")
    return HCM


@pytest.fixture
def confounder_interference_HCM_pygraphviz():
    """Pytest fixture for the Confounder Interference HCM in Figure 2 (e)."""
    HCM = pgv.AGraph(directed=True)
    HCM.add_node("A", style="filled", color="lightgrey")
    HCM.add_node("Y", style="filled", color="lightgrey")
    HCM.add_node("Z", style="filled", color="lightgrey")
    HCM.add_edge("U", "A")
    HCM.add_edge("A", "Y")
    HCM.add_edge("U", "Y")
    HCM.add_edge("A", "Z")
    HCM.add_edge("Z", "Y")
    HCM.add_subgraph(["A", "Y"], name="cluster_subunits", style="dashed", label="m")
    return HCM


@pytest.fixture
def confounder_interference_HCGM_pygraphviz():
    """Pytest fixture for the Confounder Interference HCGM in FIgure 2 (f)."""
    HCM = pgv.AGraph(directed=True)
    HCM.add_node("A", style="filled", color="lightgrey")
    HCM.add_node("Y", style="filled", color="lightgrey")
    HCM.add_node("Z", style="filled", color="lightgrey")
    HCM.add_node("Q_a", style="filled", color="lightgrey")
    HCM.add_node("Q_{y|a}", style="filled", color="lightgrey")
    HCM.add_edge("U", "Q_a")
    HCM.add_edge("Q_a", "A")
    HCM.add_edge("A", "Y")
    HCM.add_edge("U", "Q_{y|a}")
    HCM.add_edge("Q_{y|a}", "Y")
    HCM.add_edge("A", "Z")
    HCM.add_edge("Z", "Q_{y|a}")
    HCM.add_subgraph(["A", "Y"], name="cluster_subunits", label="m")
    return HCM


@pytest.fixture
def instrument_HCM_pygraphviz():
    """Pytest fixture for the Instrument HCM in Figure 2 (i)."""
    HCM = pgv.AGraph(directed=True)
    HCM.add_node("A", style="filled", color="lightgrey")
    HCM.add_node("Y", style="filled", color="lightgrey")
    HCM.add_node("Z", style="filled", color="lightgrey")
    HCM.add_edge("U", "A")
    HCM.add_edge("A", "Y")
    HCM.add_edge("U", "Y")
    HCM.add_edge("Z", "A")
    HCM.add_subgraph(["A", "Z"], name="cluster_subunits", style="dashed", label="m")
    return HCM


@pytest.fixture
def instrument_HCGM_pygraphviz():
    """Pytest fixture for the Instrument HCGM in Figure 2(j)."""
    HCM = pgv.AGraph(directed=True)
    HCM.add_node("A", style="filled", color="lightgrey")
    HCM.add_node("Y", style="filled", color="lightgrey")
    HCM.add_node("Z", style="filled", color="lightgrey")
    HCM.add_node("Q_z", style="filled", color="lightgrey")
    HCM.add_node("Q_{a|z}", style="filled", color="lightgrey")
    HCM.add_edge("U", "Q_{a|z}")
    HCM.add_edge("Q_{a|z}", "A")
    HCM.add_edge("Q_z", "Z")
    HCM.add_edge("A", "Y")
    HCM.add_edge("U", "Y")
    HCM.add_edge("Z", "A")
    HCM.add_subgraph(["A", "Z"], name="cluster_subunits", label="m")
    return HCM


def test_get_observed_confounder(confounder_HCM_pygraphviz: pgv.AGraph):
    """Test observed variables in Confounder HCM fixture."""
    assert get_observed(confounder_HCM_pygraphviz) == {"A", "Y"}


def test_get_observed_confounder_interference(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    """Test observed variables in Confounder Interference HCM fixture."""
    assert get_observed(confounder_interference_HCM_pygraphviz) == {"A", "Y", "Z"}


def test_get_observed_instrument(instrument_HCM_pygraphviz: pgv.AGraph):
    """Test observed variables in Instrument HCM fixture."""
    assert get_observed(instrument_HCM_pygraphviz) == {"A", "Y", "Z"}


def test_get_unobserved_confounder(confounder_HCM_pygraphviz: pgv.AGraph):
    """Test unobserved variables in Confounder HCM fixture."""
    assert get_unobserved(confounder_HCM_pygraphviz) == {"U"}


def test_get_unobserved_confounder_interference(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    """Test unobserved variables in Confounder Interference HCM fixture."""
    assert get_unobserved(confounder_interference_HCM_pygraphviz) == {"U"}


def test_get_unobserved_instrument(instrument_HCM_pygraphviz: pgv.AGraph):
    """Test unobserved variables in Instrument HCM fixture."""
    assert get_unobserved(instrument_HCM_pygraphviz) == {"U"}


def test_get_units_confounder(confounder_HCM_pygraphviz: pgv.AGraph):
    """Test unit variables in Confounder HCM fixture."""
    assert get_units(confounder_HCM_pygraphviz) == {"U"}


def test_get_units_confounder_interference(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    """Test unit variables in Confounder Interference HCM fixture."""
    assert get_units(confounder_interference_HCM_pygraphviz) == {"U", "Z"}


def test_get_units_instrument(instrument_HCM_pygraphviz: pgv.AGraph):
    """Test unit variables in Instrument HCM fixture."""
    assert get_units(instrument_HCM_pygraphviz) == {"U", "Y"}


def test_get_subunits_confounder(confounder_HCM_pygraphviz: pgv.AGraph):
    """Test subunit variables in Confounder HCM fixture."""
    assert get_subunits(confounder_HCM_pygraphviz) == {"A", "Y"}


def test_get_subunits_confounder_interference(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    """Test subunit variables in Confounder Interference HCM fixture."""
    assert get_subunits(confounder_interference_HCM_pygraphviz) == {"A", "Y"}


def test_get_subunits_instrument(instrument_HCM_pygraphviz: pgv.AGraph):
    """Test subunit variables in Instrument HCM fixture."""
    assert get_subunits(instrument_HCM_pygraphviz) == {"A", "Z"}


class TestFromListsConfounder:
    """Test HCM construction from lists for Confounder fixture."""

    @pytest.fixture(autouse=True)
    def HCM_fixt(self):
        """Pytest fixture from lists, to compare against 'by-hand' fixture."""
        obs_sub = ["A", "Y"]
        unobs_unit = ["U"]
        edges = [("U", "A"), ("A", "Y"), ("U", "Y")]
        self.HCM = HCM_from_lists(obs_subunits=obs_sub, unobs_units=unobs_unit, edges=edges)

    def test_observed_nodes(self, confounder_HCM_pygraphviz: pgv.AGraph):
        """Test for correct observed variables."""
        assert get_observed(self.HCM) == get_observed(confounder_HCM_pygraphviz)

    def test_unobserved_nodes(self, confounder_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unobserved variables."""
        assert get_unobserved(self.HCM) == get_unobserved(confounder_HCM_pygraphviz)

    def test_units(self, confounder_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unit variables."""
        assert get_units(self.HCM) == get_units(confounder_HCM_pygraphviz)

    def test_subunits(self, confounder_HCM_pygraphviz: pgv.AGraph):
        """Test for correct subunit variables."""
        assert get_subunits(self.HCM) == get_subunits(confounder_HCM_pygraphviz)

    def test_edges(self, confounder_HCM_pygraphviz: pgv.AGraph):
        """Test for correct edges."""
        assert set(self.HCM.edges()) == set(confounder_HCM_pygraphviz.edges())


class TestFromListsConfounderInterference:
    """Test HCM construction from lists for Confounder Interference fixture."""

    @pytest.fixture(autouse=True)
    def HCM_fixt(self):
        """Pytest fixture from lists, to compre against 'by-hand' fixture."""
        obs_sub = ["A", "Y"]
        obs_units = ["Z"]
        unobs_units = ["U"]
        edges = [("U", "A"), ("A", "Y"), ("U", "Y"), ("A", "Z"), ("Z", "Y")]
        self.HCM = HCM_from_lists(
            obs_subunits=obs_sub, obs_units=obs_units, unobs_units=unobs_units, edges=edges
        )

    def test_observed_nodes(self, confounder_interference_HCM_pygraphviz: pgv.AGraph):
        """Test for correct observed variables."""
        assert get_observed(self.HCM) == get_observed(confounder_interference_HCM_pygraphviz)

    def test_unobserved_nodes(self, confounder_interference_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unobserved variables."""
        assert get_unobserved(self.HCM) == get_unobserved(confounder_interference_HCM_pygraphviz)

    def test_units(self, confounder_interference_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unit variables."""
        assert get_units(self.HCM) == get_units(confounder_interference_HCM_pygraphviz)

    def test_subunits(self, confounder_interference_HCM_pygraphviz: pgv.AGraph):
        """Test for correct subunit variables."""
        assert get_subunits(self.HCM) == get_subunits(confounder_interference_HCM_pygraphviz)

    def test_edges(self, confounder_interference_HCM_pygraphviz: pgv.AGraph):
        """Test for correct edges."""
        assert set(self.HCM.edges()) == set(confounder_interference_HCM_pygraphviz.edges())


class TestFromListsInstrument:
    """Test HCM construction from lists for Instrument fixture."""

    @pytest.fixture(autouse=True)
    def HCM_fixt(self):
        """Pytest fixture from lists, to compare against 'by-hand' fixture."""
        obs_sub = ["A", "Z"]
        obs_units = ["Y"]
        unobs_units = ["U"]
        edges = [("Z", "A"), ("A", "Y"), ("U", "Y"), ("U", "A")]
        self.HCM = HCM_from_lists(
            obs_subunits=obs_sub, obs_units=obs_units, unobs_units=unobs_units, edges=edges
        )

    def test_observed_nodes(self, instrument_HCM_pygraphviz: pgv.AGraph):
        """Test for correct observed variables."""
        assert get_observed(self.HCM) == get_observed(instrument_HCM_pygraphviz)

    def test_unobserved_nodes(self, instrument_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unobserved variables."""
        assert get_unobserved(self.HCM) == get_unobserved(instrument_HCM_pygraphviz)

    def test_units(self, instrument_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unit variables."""
        assert get_units(self.HCM) == get_units(instrument_HCM_pygraphviz)

    def test_subunits(self, instrument_HCM_pygraphviz: pgv.AGraph):
        """Test for correct subunit variables."""
        assert get_subunits(self.HCM) == get_subunits(instrument_HCM_pygraphviz)

    def test_edges(self, instrument_HCM_pygraphviz: pgv.AGraph):
        """Test for correct edges."""
        assert set(self.HCM.edges()) == set(instrument_HCM_pygraphviz.edges())


class TestCopyConfounder:
    """Test copy HCM for Confounder fixture."""

    @pytest.fixture(autouse=True)
    def HCM_fixt(self, confounder_HCM_pygraphviz: pgv.AGraph):
        """Pytest fixture from lists, to compare against 'by-hand' fixture."""
        self.HCM = copy_HCM(confounder_HCM_pygraphviz)

    def test_observed_nodes(self, confounder_HCM_pygraphviz: pgv.AGraph):
        """Test for correct observed variables."""
        assert get_observed(self.HCM) == get_observed(confounder_HCM_pygraphviz)

    def test_unobserved_nodes(self, confounder_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unobserved variables."""
        assert get_unobserved(self.HCM) == get_unobserved(confounder_HCM_pygraphviz)

    def test_units(self, confounder_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unit variables."""
        assert get_units(self.HCM) == get_units(confounder_HCM_pygraphviz)

    def test_subunits(self, confounder_HCM_pygraphviz: pgv.AGraph):
        """Test for correct subunit variables."""
        assert get_subunits(self.HCM) == get_subunits(confounder_HCM_pygraphviz)

    def test_edges(self, confounder_HCM_pygraphviz: pgv.AGraph):
        """Test for correct edges."""
        assert set(self.HCM.edges()) == set(confounder_HCM_pygraphviz.edges())


class TestCopyConfounderInterference:
    """Test copy HCM for Confounder Interference fixture."""

    @pytest.fixture(autouse=True)
    def HCM_fixt(self, confounder_interference_HCM_pygraphviz: pgv.AGraph):
        """Pytest fixture from lists, to compare against 'by-hand' fixture."""
        self.HCM = copy_HCM(confounder_interference_HCM_pygraphviz)

    def test_observed_nodes(self, confounder_interference_HCM_pygraphviz: pgv.AGraph):
        """Test for correct observed variables."""
        assert get_observed(self.HCM) == get_observed(confounder_interference_HCM_pygraphviz)

    def test_unobserved_nodes(self, confounder_interference_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unobserved variables."""
        assert get_unobserved(self.HCM) == get_unobserved(confounder_interference_HCM_pygraphviz)

    def test_units(self, confounder_interference_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unit variables."""
        assert get_units(self.HCM) == get_units(confounder_interference_HCM_pygraphviz)

    def test_subunits(self, confounder_interference_HCM_pygraphviz: pgv.AGraph):
        """Test for correct subunit variables."""
        assert get_subunits(self.HCM) == get_subunits(confounder_interference_HCM_pygraphviz)

    def test_edges(self, confounder_interference_HCM_pygraphviz: pgv.AGraph):
        """Test for correct edges."""
        assert set(self.HCM.edges()) == set(confounder_interference_HCM_pygraphviz.edges())


class TestCopyInstrument:
    """Test copy HCM for Instrument fixture."""

    @pytest.fixture(autouse=True)
    def HCM_fixt(self, instrument_HCM_pygraphviz: pgv.AGraph):
        """Pytest fixture from lists, to compare against 'by-hand' fixture."""
        self.HCM = copy_HCM(instrument_HCM_pygraphviz)

    def test_observed_nodes(self, instrument_HCM_pygraphviz: pgv.AGraph):
        """Test for correct observed variables."""
        assert get_observed(self.HCM) == get_observed(instrument_HCM_pygraphviz)

    def test_unobserved_nodes(self, instrument_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unobserved variables."""
        assert get_unobserved(self.HCM) == get_unobserved(instrument_HCM_pygraphviz)

    def test_units(self, instrument_HCM_pygraphviz: pgv.AGraph):
        """Test for correct unit variables."""
        assert get_units(self.HCM) == get_units(instrument_HCM_pygraphviz)

    def test_subunits(self, instrument_HCM_pygraphviz: pgv.AGraph):
        """Test for correct subunit variables."""
        assert get_subunits(self.HCM) == get_subunits(instrument_HCM_pygraphviz)

    def test_edges(self, instrument_HCM_pygraphviz: pgv.AGraph):
        """Test for correct edges."""
        assert set(self.HCM.edges()) == set(instrument_HCM_pygraphviz.edges())


# For Algorithm 1


@pytest.fixture
def confounder_collapsed_nxmixedgraph():
    """Pytest fixture for collapsed Confounder HCM in Figure 2 (c)."""
    Qa = Variable("Q_a")
    Qya = Variable("Q_{y|a}")
    return NxMixedGraph.from_edges(undirected=[(Qa, Qya)])


@pytest.fixture
def confounder_interference_collapsed_nxmixedgraph():
    """Pytest fixture for collapsed Confounder Interference HCM in Figure 2 (g)."""
    Qa = Variable("Q_a")
    Qya = Variable("Q_{y|a}")
    Z = Variable("Z")
    return NxMixedGraph.from_edges(undirected=[(Qa, Qya)], directed=[(Qa, Z), (Z, Qya)])


@pytest.fixture
def instrument_collapsed_nxmixedgraph():
    """Pytest fixture for collapsed Instrument HCM in Figure 2 (k)."""
    Qaz = Variable("Q_{a|z}")
    Y = Variable("Y")
    Qz = Variable("Q_z")
    return NxMixedGraph.from_edges(undirected=[(Qaz, Y)], directed=[(Qaz, Y), (Qz, Y)])


@pytest.fixture
def direct_unit_descendents_fixt():
    """Pytest fixture to test generic and edge cases of direct unit descendents."""
    obs_sub = ["1", "2", "3", "4", "6", "7"]
    obs_units = ["5", "8"]
    edges = [("2", "3"), ("3", "4"), ("4", "5"), ("2", "8"), ("1", "8"), ("6", "7"), ("8", "5")]
    return HCM_from_lists(obs_subunits=obs_sub, obs_units=obs_units, edges=edges)


def test_parents_three(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    """Test generic case with three parents."""
    HCM = confounder_interference_HCM_pygraphviz
    assert parents(HCM, "Y") == {"U", "A", "Z"}


def test_parents_empty(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    """Test case for no parents."""
    HCM = confounder_interference_HCM_pygraphviz
    assert parents(HCM, "U") == set()


def test_Qvar_with_parents():
    """Test generic case with multiple subunit parents."""
    HCM = HCM_from_lists(obs_subunits=["A", "Y", "Z"], edges=[("A", "Y"), ("Z", "Y")])
    # don't care about order of a,z so include both because sets are unordered
    assert create_Qvar(HCM, "Y") in (Variable("Q_{y|a,z}"), Variable("Q_{y|z,a}"))


def test_Qvar_no_parents(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    """Test case when there are no subunit parents."""
    HCM = confounder_interference_HCM_pygraphviz
    assert create_Qvar(HCM, "A") == Variable("Q_a")


def test_confounder_convert_HCGM_units(
    confounder_HCM_pygraphviz: pgv.AGraph, confounder_HCGM_pygraphviz: pgv.AGraph
):
    """Test that unit variables match between Figure 2(a) conversion and Figure 2 (b)."""
    assert get_units(convert_to_HCGM(confounder_HCM_pygraphviz)) == get_units(
        confounder_HCGM_pygraphviz
    )


def test_confounder_convert_HCGM_subunits(
    confounder_HCM_pygraphviz: pgv.AGraph, confounder_HCGM_pygraphviz: pgv.AGraph
):
    """Test that subunit variables match between Figure 2(a) conversion and Figure 2 (b)."""
    assert get_subunits(convert_to_HCGM(confounder_HCM_pygraphviz)) == get_subunits(
        confounder_HCGM_pygraphviz
    )


def test_confounder_convert_HCGM_observed(
    confounder_HCM_pygraphviz: pgv.AGraph, confounder_HCGM_pygraphviz: pgv.AGraph
):
    """Test that observed variables match between Figure 2(a) conversion and Figure 2 (b)."""
    assert get_observed(convert_to_HCGM(confounder_HCM_pygraphviz)) == get_observed(
        confounder_HCGM_pygraphviz
    )


def test_confounder_convert_HCGM_edges(
    confounder_HCM_pygraphviz: pgv.AGraph, confounder_HCGM_pygraphviz: pgv.AGraph
):
    """Test that edges match between Figure 2(a) conversion and Figure 2 (b)."""
    assert set((convert_to_HCGM(confounder_HCM_pygraphviz)).edges()) == set(
        confounder_HCGM_pygraphviz.edges()
    )


def test_confounder_interference_convert_HCGM_units(
    confounder_interference_HCM_pygraphviz: pgv.AGraph,
    confounder_interference_HCGM_pygraphviz: pgv.AGraph,
):
    """Test that unit variables match between Figure 2(e) conversion and Figure 2 (f)."""
    assert get_units(convert_to_HCGM(confounder_interference_HCM_pygraphviz)) == get_units(
        confounder_interference_HCGM_pygraphviz
    )


def test_confounder_interference_convert_HCGM_subunits(
    confounder_interference_HCM_pygraphviz: pgv.AGraph,
    confounder_interference_HCGM_pygraphviz: pgv.AGraph,
):
    """Test that subunit variables match between Figure 2(e) conversion and Figure 2 (f)."""
    assert get_subunits(convert_to_HCGM(confounder_interference_HCM_pygraphviz)) == get_subunits(
        confounder_interference_HCGM_pygraphviz
    )


def test_confounder_interference_convert_HCGM_observed(
    confounder_interference_HCM_pygraphviz: pgv.AGraph,
    confounder_interference_HCGM_pygraphviz: pgv.AGraph,
):
    """Test that observed variables match between Figure 2(e) conversion and Figure 2 (f)."""
    assert get_observed(convert_to_HCGM(confounder_interference_HCM_pygraphviz)) == get_observed(
        confounder_interference_HCGM_pygraphviz
    )


def test_confounder_interference_convert_HCGM_edges(
    confounder_interference_HCM_pygraphviz: pgv.AGraph,
    confounder_interference_HCGM_pygraphviz: pgv.AGraph,
):
    """Test that edges match between Figure 2(e) conversion and Figure 2 (f)."""
    assert set((convert_to_HCGM(confounder_interference_HCM_pygraphviz)).edges()) == set(
        confounder_interference_HCGM_pygraphviz.edges()
    )


def test_instrument_convert_HCGM_units(
    instrument_HCM_pygraphviz: pgv.AGraph, instrument_HCGM_pygraphviz: pgv.AGraph
):
    """Test that unit variables match between Figure 2(i) conversion and Figure 2 (j)."""
    assert get_units(convert_to_HCGM(instrument_HCM_pygraphviz)) == get_units(
        instrument_HCGM_pygraphviz
    )


def test_instrument_convert_HCGM_subunits(
    instrument_HCM_pygraphviz: pgv.AGraph, instrument_HCGM_pygraphviz: pgv.AGraph
):
    """Test that subunit variables match between Figure 2(i) conversion and Figure 2 (j)."""
    assert get_subunits(convert_to_HCGM(instrument_HCM_pygraphviz)) == get_subunits(
        instrument_HCGM_pygraphviz
    )


def test_instrument_convert_HCGM_observed(
    instrument_HCM_pygraphviz: pgv.AGraph, instrument_HCGM_pygraphviz: pgv.AGraph
):
    """Test that observed variables match between Figure 2(i) conversion and Figure 2 (j)."""
    assert get_observed(convert_to_HCGM(instrument_HCM_pygraphviz)) == get_observed(
        instrument_HCGM_pygraphviz
    )


def test_instrument_convert_HCGM_edges(
    instrument_HCM_pygraphviz: pgv.AGraph, instrument_HCGM_pygraphviz: pgv.AGraph
):
    """Test that edges match between Figure 2(i) conversion and Figure 2 (j)."""
    assert set((convert_to_HCGM(instrument_HCM_pygraphviz)).edges()) == set(
        instrument_HCGM_pygraphviz.edges()
    )


def test_direct_unit_descends_multi(direct_unit_descendents_fixt: pgv.AGraph):
    """Test case for multiple direct unit descendents."""
    assert direct_unit_descendents(direct_unit_descendents_fixt, "2") == {"5", "8"}


def test_direct_unit_descends_no_unit_to_unit(direct_unit_descendents_fixt: pgv.AGraph):
    """Test that non-direct unit descendents are excluded."""
    assert direct_unit_descendents(direct_unit_descendents_fixt, "1") == {"8"}


def test_direct_unit_descends_multisub_path(direct_unit_descendents_fixt: pgv.AGraph):
    """Test case for path to unit descendent goes through multiple subunit descendents."""
    assert direct_unit_descendents(direct_unit_descendents_fixt, "3") == {"5"}


def test_direct_unit_descends_empty(direct_unit_descendents_fixt: pgv.AGraph):
    """Test case when there are no direct unit descendents."""
    assert direct_unit_descendents(direct_unit_descendents_fixt, "6") == set()


def test_collapse_confounder(
    confounder_HCM_pygraphviz: pgv.AGraph, confounder_collapsed_nxmixedgraph: NxMixedGraph
):
    """Test that collapsing Figure 2 (a) HCM fixture gives Figure 2 (c) fixture."""
    assert collapse_HCM(confounder_HCM_pygraphviz) == confounder_collapsed_nxmixedgraph


def test_collapse_confounder_interference(
    confounder_interference_HCM_pygraphviz: pgv.AGraph,
    confounder_interference_collapsed_nxmixedgraph: NxMixedGraph,
):
    """Test that collapsing Figure 2 (e) HCM fixture gives Figure 2 (g) fixture."""
    assert (
        collapse_HCM(confounder_interference_HCM_pygraphviz)
        == confounder_interference_collapsed_nxmixedgraph
    )


def test_collapse_instrument(
    instrument_HCM_pygraphviz: pgv.AGraph, instrument_collapsed_nxmixedgraph: NxMixedGraph
):
    """Test that collapsing Figure 2 (i) HCM fixture gives Figure 2 (k) fixture."""
    assert collapse_HCM(instrument_HCM_pygraphviz) == instrument_collapsed_nxmixedgraph


# For Algorithm 2


@pytest.fixture
def confounder_augmented_nxmixedgraph():
    """Pytest fixture for augmented Confounder HCM in Figure 2 (d)."""
    Qa = Variable("Q_a")
    Qya = Variable("Q_{y|a}")
    Qy = Variable("Q_y")
    return NxMixedGraph.from_edges(undirected=[(Qa, Qya)], directed=[(Qa, Qy), (Qya, Qy)])


@pytest.fixture
def confounder_interference_augmented_nxmixedgraph():
    """Pytest fixture for augmented Confounder Interference HCM in Figure 2 (h)."""
    Qa = Variable("Q_a")
    Qya = Variable("Q_{y|a}")
    Z = Variable("Z")
    Qy = Variable("Q_y")
    return NxMixedGraph.from_edges(
        undirected=[(Qa, Qya)], directed=[(Qa, Qy), (Qya, Qy), (Qa, Z), (Z, Qya)]
    )


@pytest.fixture
def instrument_augmented_nxmixedgraph():
    """Pytest fixture for augmented Instrument HCM in Figure A2."""
    Qaz = Variable("Q_{a|z}")
    Qa = Variable("Q_a")
    Y = Variable("Y")
    Qz = Variable("Q_z")
    return NxMixedGraph.from_edges(undirected=[(Qaz, Y)], directed=[(Qaz, Qa), (Qz, Qa), (Qa, Y)])


def test_augment_confounder(
    confounder_collapsed_nxmixedgraph: NxMixedGraph, confounder_augmented_nxmixedgraph: NxMixedGraph
):
    """Test that augmenting Figure 2 (c) fixture gives Figure 2 (d) fixture."""
    assert (
        augment_collapsed_model(
            confounder_collapsed_nxmixedgraph,
            Variable("Q_y"),
            (Variable("Q_a"), Variable("Q_{y|a}")),
        )
        == confounder_augmented_nxmixedgraph
    )


def test_augment_confounder_interference(
    confounder_interference_collapsed_nxmixedgraph: NxMixedGraph,
    confounder_interference_augmented_nxmixedgraph: NxMixedGraph,
):
    """Test that augmenting Figure 2 (g) fixture gives Figure 2 (h)."""
    assert (
        augment_collapsed_model(
            confounder_interference_collapsed_nxmixedgraph,
            Variable("Q_y"),
            (Variable("Q_a"), Variable("Q_{y|a}")),
        )
        == confounder_interference_augmented_nxmixedgraph
    )


def test_augment_instrument(
    instrument_collapsed_nxmixedgraph: NxMixedGraph, instrument_augmented_nxmixedgraph: NxMixedGraph
):
    """Test that augmenting Figure 2 (k) fixture gives Figure A2 fixture."""
    assert (
        augment_collapsed_model(
            instrument_collapsed_nxmixedgraph,
            Variable("Q_a"),
            (Variable("Q_z"), Variable("Q_{a|z}")),
        )
        == instrument_augmented_nxmixedgraph
    )


# For Algorithm 3


@pytest.fixture
def instrument_marginalized_nxmixedgraph():
    """Pytest fixture for augmented Instrument HCM in Figure 2 (l)."""
    Qaz = Variable("Q_{a|z}")
    Qa = Variable("Q_a")
    Y = Variable("Y")
    return NxMixedGraph.from_edges(undirected=[(Qaz, Y)], directed=[(Qaz, Qa), (Qa, Y)])


def test_marginalized_instrument(
    instrument_augmented_nxmixedgraph: NxMixedGraph,
    instrument_marginalized_nxmixedgraph: NxMixedGraph,
):
    """Test that marginalizing the Figure A2 fixture gives the Figure 2 (l) fixture."""
    assert (
        marginalize_augmented_model(
            instrument_augmented_nxmixedgraph, Variable("Q_a"), [Variable("Q_z")]
        )
        == instrument_marginalized_nxmixedgraph
    )
