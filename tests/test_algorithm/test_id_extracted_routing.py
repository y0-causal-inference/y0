"""Contract tests for supports_query_lineN gate functions in id_extracted_bridge.

Each test class pins one gate function to its mathematical specification by
verifying:
  - At least one *positive* case where the gate must return True
  - At least one *negative* case where the gate must return False

These tests require no Dafny runtime — they exercise only the pure-Python gate
logic and the NxMixedGraph API.  They are designed to catch gate functions that
are too narrow (silently falling back to handwritten) or too broad (passing
queries the Dafny runtime cannot handle).

The critical regression case for Line 1: when treatments are non-empty but
X ∩ An(Y)_{G_{bar_x}} = ∅, the gate must return True.  The previous
implementation (`not identification.treatments`) would return False for this
case, causing the bridge to skip the Dafny runtime and fall back silently.
"""

from __future__ import annotations

from y0.algorithm.identify import Identification
from y0.algorithm.identify.id_extracted_bridge import (
    supports_query_line1,
    supports_query_line2,
    supports_query_line3,
    supports_query_line4,
    supports_query_line5,
    supports_query_line6,
    supports_query_line7,
)
from y0.dsl import M, X, Y, Z
from y0.examples import (
    line_1_example,
    line_2_example,
    line_3_example,
    line_5_example,
    line_6_example,
    line_7_example,
)
from y0.graph import NxMixedGraph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _id_in(example, identification_idx: int = 0) -> Identification:
    """Return the first id_in from an example's identification list."""
    return example.identifications[identification_idx]["id_in"][0]


# ---------------------------------------------------------------------------
# Shared fixture objects derived from the canonical line-N examples
# ---------------------------------------------------------------------------

# Line 1: Z→Y, outcomes={Y}, treatments=∅
_line1_id = _id_in(line_1_example)

# Line 2: Z→Y, Y→X, Z-X, outcomes={Y}, treatments={X}
# X is downstream of Y (Y→X), so X is NOT an ancestor of Y → V ≠ An(Y)_G
_line2_id = _id_in(line_2_example)

# Line 3: Z→X→Y, Z-X, outcomes={Y}, treatments={X}
# Z has no effect on Y when we intervene on X (Z→X is cut)
_line3_id = _id_in(line_3_example)

# Line 4: conservative 3-node frontdoor X→M→Y, X-Y
# The line_4_example has 4 nodes so the conservative gate rejects it;
# use the minimal 3-node pattern that the gate accepts.
_line4_id = Identification.from_parts(
    outcomes={Y},
    treatments={X},
    graph=NxMixedGraph.from_edges(directed=[(X, M), (M, Y)], undirected=[(X, Y)]),
)

# Line 5: X→Y, X-Y (bow-arc / hedge graph)
_line5_id = _id_in(line_5_example)

# Line 6: X→Y, X→Z, Z→Y, X-Z, outcomes={Y}, treatments={X,Z}
_line6_id = _id_in(line_6_example)

# Line 7: X→Y1, W1→X, W1-Y1, outcomes={Y1}, treatments={X,W1}
_line7_id = _id_in(line_7_example)

# ---------------------------------------------------------------------------
# Boundary case for Line 1 (the regression that motivated this test file)
# ---------------------------------------------------------------------------
# Graph: X→Z, Y (isolated).  treatments={X}, outcomes={Y}.
# ancestors_inclusive({Y}) = {Y} (Y has no predecessors).
# treatments ∩ {Y} = ∅  →  Line 1 applies.
# The previous gate implementation (`not treatments`) returned False here.
_line1_boundary_id = Identification.from_parts(
    outcomes={Y},
    treatments={X},
    graph=NxMixedGraph.from_edges(nodes=[Y], directed=[(X, Z)]),
)


# ===========================================================================
# Test classes
# ===========================================================================


class TestSupportsQueryLine1:
    """Gate for Line 1: X ∩ An(Y)_{G_{bar_x}} = ∅."""

    def test_positive_no_treatments(self) -> None:
        """Line 1 applies trivially when X = ∅."""
        assert supports_query_line1(_line1_id) is True

    def test_positive_nonempty_treatments_no_causal_path(self) -> None:
        """Line 1 applies when X ≠ ∅ but X ∩ An(Y)_{G_{bar_x}} = ∅.

        This is the regression case: treatments exist but none of them have a
        directed path to any outcome in the interventional graph.
        """
        assert supports_query_line1(_line1_boundary_id) is True

    def test_negative_treatments_with_causal_path(self) -> None:
        """Line 1 does not apply when some treatment can reach an outcome."""
        # In line_7_id: X→Y1 with W1→X, W1-Y1; removing in-edges to {X,W1}
        # still leaves X→Y1, so X ∈ An(Y1)_{G_{bar_{X,W1}}} → not ∅ → False.
        assert supports_query_line1(_line7_id) is False


class TestSupportsQueryLine2:
    """Gate for Line 2: V ≠ An(Y)_G (there are nodes not ancestral to Y)."""

    def test_positive(self) -> None:
        """Line 2 applies when some node is not an ancestor of any outcome."""
        # In line_2_id: Y→X, so X is downstream of Y and NOT an ancestor of Y.
        assert supports_query_line2(_line2_id) is True

    def test_negative_all_nodes_ancestral(self) -> None:
        """Line 2 does not apply when all nodes are already ancestors of Y."""
        # In line_1_id: Z→Y, An({Y}) = {Z, Y} = V.
        assert supports_query_line2(_line1_id) is False


class TestSupportsQueryLine3:
    """Gate for Line 3: some treatment has no effect on outcomes."""

    def test_positive(self) -> None:
        """Line 3 applies when get_no_effect_on_outcomes is non-empty."""
        # In line_3_id (Z→X→Y, Z-X): intervening on X cuts Z→X and Z-X,
        # so Z has no path to Y in G_{bar_X}.  Z is a no-effect node.
        assert supports_query_line3(_line3_id) is True

    def test_negative_all_treatments_affect_outcomes(self) -> None:
        """Line 3 does not apply when every treatment affects some outcome."""
        # In line_5_id (X→Y, X-Y): removing in-edges to X still leaves X→Y.
        # get_no_effect_on_outcomes = ∅.
        assert supports_query_line3(_line5_id) is False


class TestSupportsQueryLine4:
    """Gate for Line 4: conservative 3-node frontdoor pattern."""

    def test_positive_minimal_frontdoor(self) -> None:
        """Line 4 applies to the minimal X→M→Y, X-Y pattern."""
        assert supports_query_line4(_line4_id) is True

    def test_negative_no_treatments(self) -> None:
        """Line 4 does not apply when there are no treatments."""
        assert supports_query_line4(_line1_id) is False

    def test_negative_wrong_edge_structure(self) -> None:
        """Line 4 does not apply when treatment→mediator→outcome edges are absent."""
        # line_5_id has no mediator between X and Y.
        assert supports_query_line4(_line5_id) is False


class TestSupportsQueryLine5:
    """Gate for Line 5: full graph is one C-component (hedge/bow-arc case)."""

    def test_positive(self) -> None:
        """Line 5 applies when the full graph is a single district (hedge)."""
        # X→Y, X-Y: the bidirected edge makes {X, Y} one district.
        assert supports_query_line5(_line5_id) is True

    def test_negative_graph_has_multiple_districts(self) -> None:
        """Line 5 does not apply when the full graph has multiple C-components."""
        # line_6_id has districts {X,Z} and {Y} (not all one district).
        assert supports_query_line5(_line6_id) is False


class TestSupportsQueryLine6:
    """Gate for Line 6: district(G_without_X) is already a district of G."""

    def test_positive(self) -> None:
        """Line 6 applies when removing treatments leaves a known district."""
        # In line_6_id: removing {X,Z} → just {Y}.  {Y} is its own district in G.
        assert supports_query_line6(_line6_id) is True

    def test_negative_district_not_in_graph(self) -> None:
        """Line 6 does not apply when district(G_without_X) is not a district."""
        # In line_7_id: removing {X,W1} → just {Y1}.  But {Y1} is NOT a district
        # in G because W1-Y1 bidirected puts Y1 in the larger {W1,Y1} district.
        assert supports_query_line6(_line7_id) is False


class TestSupportsQueryLine7:
    """Gate for Line 7: district(G_without_X) is a proper subset of a G district."""

    def test_positive(self) -> None:
        """Line 7 applies when district(G_without_X) ⊂ some district of G."""
        # In line_7_id: removing {X,W1} → {Y1}.  {Y1} ⊂ {W1,Y1} in G.
        assert supports_query_line7(_line7_id) is True

    def test_negative_full_graph_is_one_district(self) -> None:
        """Line 7 does not apply when the graph is one big C-component (Line 5)."""
        # In line_5_id: the single bidirected X-Y makes the whole graph one district.
        assert supports_query_line7(_line5_id) is False
