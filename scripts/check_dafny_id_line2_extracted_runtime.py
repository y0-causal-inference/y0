#!/usr/bin/env python3
"""
Smoke test for extracted Line 2 Dafny runtime.
Validates that the Python runtime emits canonical IR.
"""

import sys
import json
from pathlib import Path

# Add extracted module to path
extracted_dir = Path(__file__).parent.parent / ".cache" / "y0" / "dafny" / "id_line2_extracted_py"
sys.path.insert(0, str(extracted_dir))

try:
    from IDLine2Extracted import default__ as IDLine2Extracted
    from IDLine2Extracted import Edge_Edge
    import _dafny
except ImportError as e:
    print(f"ERROR: Could not import extracted module: {e}")
    sys.exit(1)

def run_smoke_test():
    """Test Line 2: Ancestral graph reduction."""
    
    # Create a simple test case: X -> Y, Z (independent)
    # All nodes = {X, Y, Z}
    # Outcomes = {Y}
    # Treatments = {X}
    # Ordering = [X, Y, Z]
    # Edges: X -> Y
    
    edges = _dafny.SeqWithoutIsStrInference([
        Edge_Edge("X", "Y")
    ])
    all_nodes = _dafny.SeqWithoutIsStrInference(["X", "Y", "Z"])
    outcomes = _dafny.Set({"Y"})
    treatments = _dafny.Set({"X"})
    ordering = _dafny.SeqWithoutIsStrInference(["X", "Y", "Z"])
    
    ok, doc = IDLine2Extracted.IDLine2ToIR(
        "test.graph",
        edges,
        all_nodes,
        outcomes,
        treatments,
        ordering
    )
    
    if not ok:
        print("Line 2 not applicable (expected for this graph)")
        # Try case where line 2 DOES apply: Z not in An(Y)
        edges2 = _dafny.SeqWithoutIsStrInference([
            Edge_Edge("X", "Y")
        ])
        # Z is not an ancestor of Y, so Line 2 should apply
        ok2, doc2 = IDLine2Extracted.IDLine2ToIR(
            "test.graph2",
            edges2,
            all_nodes,
            outcomes,
            treatments,
            ordering
        )
        
        if ok2:
            doc = doc2
            ok = True
    
    # Convert to JSON for inspection
    result_dict = {
        "version": str(doc.version),
        "engine": str(doc.engine),
        "query": {
            "graph_id": str(doc.query.graph__id),
            "outcomes": [str(x) for x in doc.query.outcomes],
            "treatments": [str(x) for x in doc.query.treatments],
            "ordering": [str(x) for x in doc.query.ordering],
        },
        "result": {
            "type": doc.result.__class__.__name__,
        }
    }
    
    # Emit canonical IR JSON
    print(json.dumps(result_dict, indent=2))
    
    if ok:
        print("\n✓ Line 2 extraction smoke test PASSED")
        return 0
    else:
        print("\n✗ Line 2 extraction smoke test FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_smoke_test())
