# Code Implementation & Comparison and ID vs IDCD Analysis

---

## Summary 

**Findings**: Both Approach A (Making a call to identify_outcomes to construct the expression using apt-order) and Approach B (direct conditional probability definition) are **mathematically equivalent** across all the tested edge cases. Both correctly implement Line 23 of the IDCD algorithm. However, based on the comparison with the Tian & Pearl ID algorithm, using Approach A would be more beneficial. 

---

##
 
 ### Background & Findings

###
 Line 23 from Forré & Mooij (2019)

```

R_A[S] ← P̃(S | Pred^G_<(S) ∩ A, do(J ∪ V \ A))

```

**Purpose**: Compute the distribution for each strongly connected component in the consolidated district, conditioned on the
predecessors. 

**Context**:
- This is part of the IDCD algorithm's recursive case (Lines 21-26)
- S is an SCC in G[A] (induced subgraph)
- The result is a symbolic expression, as opposed to a numerical computation

### Main Question

How should we implement this line? Two approaches:
- **Approach A**: Call `identify_outcomes` with apt-order
- **Approach B**: Direct calculation using DSL operations

---

## The Two Approaches

### Approach A: Using `identify_outcomes` with apt-order

```python

from y0.algorith.identify import identify_outcomes

scc_distribution = identify_outcomes(
    graph=graph,
    outcomes=scc,
    treatments=intervention_set,
    conditions=predecessors if predecessors else None,
    strict=True,
    ordering=apt_order_a, # the key is this addition here
)
```
**How it works**:
- Leverages the Tian & Pearl ID Algorithm
- Substitutes apt-order for topological order
- Returns the symbolic expression using chain rule factorization
- Handles identification failures

**Design rationale**:
- Extends the acyclic ID algorith to a cyclic case
- Consistent with the format and design of Tian & Pearl algorithm

### Approach B: Direct Calculation

```python

def _calculate_scc_distribution(
    scc: frozenset[Variable],
    predecessors: set[Variable],
    intervention_set: set[Variable],
    original_distribution: Expression,
    graph: NxMixedGraph,
) -> Expression:

    """Direct construction using DSL operations."""
    # Get all variables in the distribution
    all_variables = original_distribution.get_variables()

    # Variables to keep: S ∪ Pred
    variables_to_keep = set(scc) | predecessors

    # Step 1: Marginalize to keep only S and predecessors
    variables_to_marginalize = all_variables - variables_to_keep
    if variables_to_marginalize:
        result = original_distribution.marginalize(variables_to_marginalize)
    else:
        result = original_distribution

    # Step 2: Condition on predecessors (if any)
    if predecessors:
        result = result.conditional(list(predecessors))  # P(S, Pred) / P(Pred)

    return result

```
**How it works**:
- Direct manipulation of symbolic distributions
- Marginalize to relevant variables
- Condition on predecessors using `.conditional()`
- Return compact conditional probability form

**Design rationale**:
- Direct implementation of mathematical definition
- Self-contained logic
- Explicit control over operations


---

## Test Results

5 edge cases were tested and shown below are 2 representative examples of them:

### Test 1: Single Node SCC with Multiple Predecessors

**Setup**:

- SCC: {Y} (single node)
- Predecessors: {W, X, Z}
- Apt-order: [W, X, Z, Y]

**Approach A** (chain rule):

```
P(Y | W, X, Z) / Sum[Y](P(Y | W, X, Z))
```

**Approach B** (direct):
```
P(W, X, Y, Z) / Sum[W, X, Z](P(W, X, Y, Z))
```

**Both compute**: `P(Y | W, X, Z)` ✅

---

### Test 2: Multi-Node Cycle with Single Predecessor


**Setup**:
- SCC: {W, X, Z} (forms cycle)
- Predecessors: {R}
- Apt-order: [R, W, X, Z, Y]
    
**Approach A** (chain rule):

```
P(W | R) × P(X | R, W) × P(Z | R, W, X) / Sum[W,X,Z](...)
```

**Approach B** (direct):
```
P(R, W, X, Z) / Sum[R](P(R, W, X, Z))
```

**Both compute**: `P(W, X, Z | R)` ✅


### Test 3: No Predecessors

**Setup**:
- SCC: {R} (first in apt-order)
- Predecessors: {} 

**Results**: Both handle empty predecessors correctly

---

## Final Decision

**Approach A was selected** based on:

1. **Code Quality**: As opposed to another function, this approach makes a call to another existing function. 
2.. **Clarity**: This approach makes the theoretical connection explicit.
3. **Mathematical Correctness**: Both approaches are mathematically equivalent to each other. 



**Testing**: All existing tests pass with Approach A implementation so far.


--

## ID vs IDCD Comparison with a DAG

