import pytest
import pygraphviz as pgv
from y0.graph import NxMixedGraph
from y0.dsl import Variable
from y0.hierarchical import (get_observed,
    HCM_from_lists,
    get_unobserved, 
    get_subunits,
    get_units,
    parents,
    node_string,
    create_Qvar,
    direct_unit_descendents,
    collapse_HCM)

@pytest.fixture
def confounder_HCM_pygraphviz():
    HCM = pgv.AGraph(directed=True)
    HCM.add_node("A", style="filled", color="lightgrey")
    HCM.add_node("Y", style="filled", color="lightgrey")
    HCM.add_edge("U", "A")
    HCM.add_edge("A", "Y")
    HCM.add_edge("U", "Y")
    HCM.add_subgraph(["A", "Y"], name="cluster_subunits", style="dashed", label="m")
    return HCM

@pytest.fixture
def confounder_interference_HCM_pygraphviz():
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
def instrument_HCM_pygraphviz():
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

def test_get_observed_confounder(confounder_HCM_pygraphviz: pgv.AGraph):
    assert get_observed(confounder_HCM_pygraphviz) == set(['A', 'Y'])

def test_get_observed_confounder_interference(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    assert get_observed(confounder_interference_HCM_pygraphviz) == set(['A', 'Y', 'Z'])

def test_get_observed_instrument(instrument_HCM_pygraphviz: pgv.AGraph):
    assert get_observed(instrument_HCM_pygraphviz) == set(['A', 'Y', 'Z'])

def test_get_unobserved_confounder(confounder_HCM_pygraphviz: pgv.AGraph):
    assert get_unobserved(confounder_HCM_pygraphviz) == set(['U'])

def test_get_unobserved_confounder_interference(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    assert get_unobserved(confounder_interference_HCM_pygraphviz) == set(['U'])

def test_get_unobserved_instrument(instrument_HCM_pygraphviz: pgv.AGraph):
    assert get_unobserved(instrument_HCM_pygraphviz) == set(['U'])

def test_get_units_confounder(confounder_HCM_pygraphviz: pgv.AGraph):
    assert get_units(confounder_HCM_pygraphviz) == set(['U'])

def test_get_units_confounder_interference(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    assert get_units(confounder_interference_HCM_pygraphviz) == set(['U', 'Z'])

def test_get_units_instrument(instrument_HCM_pygraphviz: pgv.AGraph):
    assert get_units(instrument_HCM_pygraphviz) == set(['U', 'Y'])

def test_get_subunits_confounder(confounder_HCM_pygraphviz: pgv.AGraph):
    assert get_subunits(confounder_HCM_pygraphviz) == set(['A', 'Y'])

def test_get_subunits_confounder_interference(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    assert get_subunits(confounder_interference_HCM_pygraphviz) == set(['A', 'Y'])

def test_get_subunits_instrument(instrument_HCM_pygraphviz: pgv.AGraph):
    assert get_subunits(instrument_HCM_pygraphviz) == set(['A', 'Z'])

class TestFromListsConfounder:
    @pytest.fixture(autouse=True)
    def HCM_fixt(self):
        obs_sub = ['A', 'Y']
        unobs_unit = ['U']
        edges = [('U','A'), ('A','Y'), ('U','Y')]
        self.HCM = HCM_from_lists(obs_subunits=obs_sub, unobs_units=unobs_unit, edges=edges)
    
    def test_observed_nodes(self, confounder_HCM_pygraphviz):
        assert get_observed(self.HCM) == get_observed(confounder_HCM_pygraphviz)

    def test_unobserved_nodes(self, confounder_HCM_pygraphviz):
        assert get_unobserved(self.HCM) == get_unobserved(confounder_HCM_pygraphviz)

    def test_units(self, confounder_HCM_pygraphviz):
        assert get_units(self.HCM) == get_units(confounder_HCM_pygraphviz)

    def test_subunits(self, confounder_HCM_pygraphviz):
        assert get_subunits(self.HCM) == get_subunits(confounder_HCM_pygraphviz)
         
    def test_edges(self, confounder_HCM_pygraphviz):
        assert set(self.HCM.edges()) == set(confounder_HCM_pygraphviz.edges())

class TestFromListsConfounderInterference:
    @pytest.fixture(autouse=True)
    def HCM_fixt(self):
        obs_sub = ['A', 'Y']
        obs_units = ['Z']
        unobs_units = ['U']
        edges = [('U','A'), ('A','Y'), ('U','Y'), ('A','Z'), ('Z', 'Y')]
        self.HCM = HCM_from_lists(obs_subunits=obs_sub, obs_units=obs_units, 
                                  unobs_units=unobs_units, edges=edges)
        
    def test_observed_nodes(self, confounder_interference_HCM_pygraphviz):
        assert get_observed(self.HCM) == get_observed(confounder_interference_HCM_pygraphviz)

    def test_unobserved_nodes(self, confounder_interference_HCM_pygraphviz):
        assert get_unobserved(self.HCM) == get_unobserved(confounder_interference_HCM_pygraphviz)

    def test_units(self, confounder_interference_HCM_pygraphviz):
        assert get_units(self.HCM) == get_units(confounder_interference_HCM_pygraphviz)

    def test_subunits(self, confounder_interference_HCM_pygraphviz):
        assert get_subunits(self.HCM) == get_subunits(confounder_interference_HCM_pygraphviz)

    def test_edges(self, confounder_interference_HCM_pygraphviz):
        assert set(self.HCM.edges()) == set(confounder_interference_HCM_pygraphviz.edges())
        
class TestFromListsInstrument:
    @pytest.fixture(autouse=True)
    def HCM_fixt(self):
        obs_sub = ['A','Z']
        obs_units = ['Y']
        unobs_units = ['U']
        edges = [('Z','A'), ('A','Y'), ('U','Y'), ('U','A')]
        self.HCM = HCM_from_lists(obs_subunits=obs_sub, obs_units=obs_units, 
                                  unobs_units=unobs_units, edges=edges)
        
    def test_observed_nodes(self, instrument_HCM_pygraphviz):
        assert get_observed(self.HCM) == get_observed(instrument_HCM_pygraphviz)

    def test_unobserved_nodes(self, instrument_HCM_pygraphviz):
        assert get_unobserved(self.HCM) == get_unobserved(instrument_HCM_pygraphviz)

    def test_units(self, instrument_HCM_pygraphviz):
        assert get_units(self.HCM) == get_units(instrument_HCM_pygraphviz)

    def test_subunits(self, instrument_HCM_pygraphviz):
        assert get_subunits(self.HCM) == get_subunits(instrument_HCM_pygraphviz)

    def test_edges(self, instrument_HCM_pygraphviz):
        assert set(self.HCM.edges()) == set(instrument_HCM_pygraphviz.edges())

@pytest.fixture
def confounder_collapsed_nxmixedgraph():
    Qa = Variable("Q_a")
    U = Variable("U")
    Qya = Variable("Q_{y|a}")
    return NxMixedGraph.from_edges(undirected=[(Qa, Qya)])

@pytest.fixture
def confounder_interference_collapsed_nxmixedgraph():
    Qa = Variable("Q_a")
    U = Variable("U")
    Qya = Variable("Q_{y|a}")
    Z = Variable("Z")
    return NxMixedGraph.from_edges(undirected=[(Qa, Qya)], directed=[(Qa,Z), (Z,Qya)])

@pytest.fixture
def instrument_collapsed_nxmixedgraph():
    Qaz = Variable("Q_{a|z}")
    U = Variable("U")
    Y = Variable("Y")
    Qz = Variable("Q_z")
    return NxMixedGraph.from_edges(undirected=[(Qaz, Y)], directed=[(Qaz,Y), (Qz,Y)])

@pytest.fixture
def direct_unit_descendents_fixt():
    obs_sub = ['1','2','3','4','6','7']
    obs_units = ['5','8']
    edges = [('2','3'), ('3','4'), ('4','5'), ('2','8'), ('1','8'), ('6','7'), ('8','5')]
    return HCM_from_lists(obs_subunits=obs_sub, obs_units=obs_units, edges=edges)

def test_parents_three(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    HCM = confounder_interference_HCM_pygraphviz
    assert parents(HCM, 'Y') == {'U', 'A', 'Z'}

def test_parents_empty(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    HCM = confounder_interference_HCM_pygraphviz
    assert parents(HCM, 'U') == set()

# def test_Qvar_with_parent(confounder_interference_HCM_pygraphviz: pgv.AGraph):
#     HCM = confounder_interference_HCM_pygraphviz
#     assert create_Qvar(HCM, 'Y') == Variable('Q_{y|a}')
def test_Qvar_with_parents():
    HCM = HCM_from_lists(obs_subunits=['A','Y', 'Z'], 
                     edges=[('A','Y'), ('Z','Y')])
    assert create_Qvar(HCM, 'Y') in (Variable('Q_{y|a,z}'), Variable('Q_{y|z,a}'))

def test_Qvar_no_parents(confounder_interference_HCM_pygraphviz: pgv.AGraph):
    HCM = confounder_interference_HCM_pygraphviz
    assert create_Qvar(HCM, 'A') == Variable('Q_a')

def test_direct_unit_descends_multi(direct_unit_descendents_fixt):
    assert direct_unit_descendents(direct_unit_descendents_fixt, '2') == {'5','8'}

def test_direct_unit_descends_no_unit_to_unit(direct_unit_descendents_fixt):
    assert direct_unit_descendents(direct_unit_descendents_fixt, '1') == {'8'}

def test_direct_unit_descends_multisub_path(direct_unit_descendents_fixt):
    assert direct_unit_descendents(direct_unit_descendents_fixt, '3') == {'5'}

def test_direct_unit_descends_empty(direct_unit_descendents_fixt):
    assert direct_unit_descendents(direct_unit_descendents_fixt, '6') == set()

def test_collapse_confounder(confounder_HCM_pygraphviz, confounder_collapsed_nxmixedgraph):
    assert collapse_HCM(confounder_HCM_pygraphviz) == confounder_collapsed_nxmixedgraph

def test_collapse_confounder_interference(confounder_interference_HCM_pygraphviz, 
                                          confounder_interference_collapsed_nxmixedgraph):
    assert collapse_HCM(confounder_interference_HCM_pygraphviz) == confounder_interference_collapsed_nxmixedgraph

def test_collapse_instrument(instrument_HCM_pygraphviz, instrument_collapsed_nxmixedgraph):
    assert collapse_HCM(instrument_HCM_pygraphviz) == instrument_collapsed_nxmixedgraph