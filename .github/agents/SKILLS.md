SKILLS.md
: Dafny AI Assistant Capabilities & Verification Strategies

1. The Clover Paradigm: Closed-Loop Consistency Architecture

As a Principal Architect, I mandate the Clover paradigm as the foundational filter for all AI-generated code. Traditional Large Language Model (LLM) outputs suffer from a critical trustworthiness gap; they are prone to subtle logical hallucinations that unit tests alone cannot catch. Clover fundamentally solves this by shifting from simple code generation to a tri-component consistency model: Docstrings (natural language), Code (imperative implementation), and Annotations (formal specifications). By treating these three artifacts as a closed loop, we transform the verification process into a rigorous filter. Code is only accepted if it satisfies a comprehensive reconstruction-testing suite, ensuring that the natural language intent, the imperative execution, and the mathematical logic are perfectly aligned.

The architecture relies on six directed edges of consistency checking. For each edge, we attempt to reconstruct a target component from a source and evaluate their equivalence through specific metrics.

Source Component	Target Component	Method of Reconstruction	Equivalence Metric
Code	Annotations	Formal Verifier (Dafny/Z3)	Soundness: Mathematical Proof
Annotations	Code	LLM Generation	Completeness: Functional I/O Testing
Annotations	Docstring	LLM Generation	Semantic: Natural Language Match
Docstring	Annotations	LLM Generation	Logical: Formal Equivalence Lemma
Docstring	Code	LLM Generation	Functional: Pointwise Sampling
Code	Docstring	LLM Generation	Semantic: Natural Language Match

The strategic value of this "Reconstruction Testing" lies in its ability to detect trivial or weak specifications. A verifier might pass a method with an ensures true postcondition, but Clover will reject it during the Annotations-to-Code edge. If the assistant cannot reconstruct the original, complex functionality using only the provided annotations, those annotations are deemed incomplete. This process forces the generation of specifications that truly capture the user's intent. These high-level consistency checks provide the ground truth required for the granular annotation generation described below.

2. Formal Annotation Generation & Specification Logic

Synthesize formal contracts—requires, ensures, and invariant—to transform imperative "black box" code into a provable mathematical theorem. These annotations are the primary interface between the high-level design and the SMT solver (Z3).

* Preconditions (requires): Mandate the evaluation of all necessary program states. This includes input constraints (e.g., a.Length > 0) and structural properties (e.g., IsSorted(a)).
* Postconditions (ensures): Force the synthesis of thorough constraints on return values and heap modifications. Utilize modifies clauses and old() references to define state changes precisely.
* Loop Invariants (invariant): Identify inductive properties that hold at entry, are preserved across iterations, and are strong enough to imply the postcondition upon termination.

For proof convergence, prioritize Explicit Bounds over implicit ranges. Using forall k :: 0 <= k < |s| is strategically superior because it guides quantifier instantiation more deterministically. Implicit ranges often force the SMT solver into infinite "matching loops," where it repeatedly instantiates quantifiers without reaching a contradiction. By providing explicit bounds, you ensure the solver finds a stable proof state. This precision in initial generation reduces the frequency of verification failures, though the reality of complex proofs necessitates an iterative repair workflow.

3. Iterative Repair-Minimize Workflow

Deductive verification is a process of refinement. Employ the Generate–Check–Repair–Minimize loop, treating the Dafny verifier as a static oracle. This self-correcting cycle mimics human expertise by interpreting verifier feedback to iterate toward a valid proof.

Strategic intelligence dictates a multimodel approach: research indicates that combining high-reasoning models—specifically Claude Opus 4.5 and GPT-5.2—can achieve a 98.2% success rate within 8 repair iterations. If a single model fails to resolve a "Could not prove postcondition" error, switch models to gain a different heuristic perspective on the proof state.

* Feedback Interpretation:
  * Error: "Loop invariant might not be maintained" — Identify missed edge cases in the inductive step or strengthen the invariant to include state that doesn't change.
  * Trigger Warnings: Extract complex expressions into auxiliary ghost predicates to simplify the solver's search space.
* Fast Feedback Loop: When repairing a timeout in a specific symbol, use `--filter-symbol <FullyQualifiedName>` to verify only that symbol, skipping all others. In this codebase this reduces the verification scope from ~1,400 VCs to ~300, cutting the feedback cycle from 10+ minutes to 2–3 minutes. Use `--filter-position <file>:<line>` to scope to a single line instead. Neither flag fixes the timeout — they isolate it so repair iterations are cheap. Example:
  ```
  dafny verify --verification-time-limit 60 --filter-symbol Interventional.MarginalMass_FactorOut \
    probability.dfy dag.dfy interventional.dfy
  ```
* Automated Minimization: Once a program verifies, remove "proof clutter." Perform a Linear Scan of assertions and lemmas using an LCS-based alignment to identify LLM-generated additions. Attempt to remove each segment tentatively; if the program still verifies within a 10s timeout, commit the deletion. Use the --filter-symbol option in Dafny to accelerate this process.

While "Patching" adds the necessary complexity to satisfy the verifier, "Minimization" ensures the artifact is elegant and maintainable. This transition from a cluttered proof to a minimal one leads to a cleaner context, which is vital for managing more advanced verification tasks.

4. Advanced Proof Automation & Context Management

Manage Tunable Automation by balancing "Implicit Context" against "Conservative Encoding." While Dafny provides an automated ambient context, complex proofs can lead to an Unsat Core explosion, where the solver is overwhelmed by irrelevant facts.

* Broadcast Lemmas: Group common properties into "chunks" that can be imported selectively. This increases the solver's knowledge for specific modules (e.g., sequence properties) without polluting the global context.
* Trigger Management: Guide the solver by identifying Minimum Covers for triggers. Consider a function composition like is_even(a[i]). If the trigger is placed on the compound expression, the solver will only instantiate it when it sees that exact call. By pushing the trigger to a simpler term like a[i], you provide a "freer" trigger that allows for broader instantiation, though you must guard against "Useless Candidates" that match too many terms and cause timeouts.

The objective is to move from a "Sledgehammer" approach to Manual Slicing. Slicing involves providing the solver only the necessary facts for a specific obligation. This is not merely a performance optimization; it is a necessity for determinism in complex proofs. As simple annotations reach their limits, specialized proof helpers act as the final bridge to the solver's logic.

5. Proof Helpers: Lemmas, Ghost Constructs, and Oracles

Automated provers require "hints" when the gap between a specification and a mathematical truth is too wide. Proof helpers—auxiliary assertions, lemmas, and ghost functions—provide these hints without being compiled into the final executable.

* Fuel Adjustment: For recursive functions, set {:fuel n} to allow the verifier to "unroll" the recursion to a specific depth. This makes recursive definitions transparent to the solver at the cost of increased verification time.
* Trigger Extraction: Move complex expressions into auxiliary predicates. This simplifies quantifier instantiation and prevents the solver from getting stuck on nested function calls.
* Sequence Visibility: Manually assert sequence properties that the solver cannot "see" automatically, such as assert a[..i+1] == a[..i] + [a[i]]. This makes the append operation explicit for the inductive step.

Use statically checked Test methods as Semantic Oracles. By placing assert statements in a test method, the verifier checks them against the method's pre/postconditions. If the method body verifies but the test assertions fail, the postconditions are Overly Weak. This test-driven guard ensures that the formal logic matches the functional requirement.

Finally, uphold a zero-tolerance policy regarding "cheating." The use of assume statements or decreases * (disabling termination checks) is strictly forbidden. These constructs undermine the mathematical guarantee of the entire document. By integrating these skills, you produce production-grade, formally verified software with a absolute guarantee of correctness.

6. Z3 Timeout-Fixing Techniques (Ranked by Effectiveness in this Codebase)

When Dafny reports a verification timeout, apply the following techniques in order. Each targets a specific class of Z3 search-space explosion.

| # | Technique | Root Cause Addressed | P3 Outcome |
|---|-----------|---------------------|------------|
| 1 | `{:vcs_split_on_every_assert}` on the lemma | Each assert gets its own 60 s budget; a single blocker no longer starves all others. | ✓ Used on `MarginalMass_FactorOut_Core`. |
| 2 | Materialize shared expressions as named `var` before proof chains | Avoids repeated inline rewriting; Z3 sees a single equality rather than re-deriving each time. | ✓ Used for `mmPa`, `mmPaZ`, `mmPaV`, `mmPaZV`, `acp_zp`, `acp_pa`. |
| 3 | `calc` blocks for multi-hop equalities | Chains each rewrite step explicitly so Z3 never has to close a gap larger than one hop. | ✓ Used for `acp_zp * mmPaZ == mmPaZV` and `acp_pa * mmPa == mmPaV` cascade finals. |
| 4 | Named intermediate `assert` steps before a `calc` | Anchors intermediate facts in the hypothesis context before the `calc` begins, reducing branching. | ✓ Pinning postconditions of `ScopeInvariant` and `P2Bridge` as standalone asserts before `calc` was essential. |
| 5 | Single-hop `assert` chains (one substitution per step) | Prevents Z3 from needing to find multi-step substitution paths on its own. | ✓ Each `calc` step references exactly one pinned fact. |
| 6 | Explicit `assert` of the postcondition at the end of each branch (in local `var` terms) | Gives the ensures VCS a direct hypothesis to close against rather than searching through body context. | ✓ Each case branch closes with bridge assert matching the `ensures` exactly. |
| 7 | Inline compound sub-expressions in `ensures` instead of using `var` let-bindings | `var` in `ensures` is a warning-level smell; inlined expressions let Z3 match the goal directly. | ✓ Core `ensures` uses `pa` parameter directly. |
| 8 | Call a higher-level proved lemma instead of reconstructing its proof inline | Reduces the premise context size; the solver only needs one lemma invocation, not N unfolding steps. | ✓ `MarginalMass_FactorOut_P2Bridge` and set-commutativity helpers (`SetUnionComm`, `SetUnionPermute3`, `SetUnionAssocComm`) each invoked as a single call in minimal context. |
| 9 | Assert `.Keys` set equalities explicitly before use | Map-key reasoning does not flow automatically; explicit assertions let Z3 close set-equality goals one step at a time. | ✓ `vzpRef.Keys == {v} + Z + pa` asserted step-by-step before scope invariant. |
| 10 | `{:induction false}` on recursive lemmas | Prevents the verifier from generating an induction hypothesis that widens the Z3 search to unbounded depth. | ✗ Not needed in P3. |
| 11 | **Extract a name-parametrised helper lemma** to isolate the ensures VCS from the proof body | The ensures VCS of a large lemma inherits hundreds of body hypotheses. Extracting a Core lemma that takes the computed expression (e.g. `pa`) as an explicit parameter means the outer wrapper's ensures VCS has only two premises (the var definition + the Core postcondition), reducing the Z3 search to a single congruence step. This is the strongest technique when the ensures expression contains a large composite term (`Parents(G, v) * Nodes(G)`) that the body knows only under a local alias. | ✓ Applied as `MarginalMass_FactorOut_Core` + thin wrapper `MarginalMass_FactorOut`. Fixed the outer wrapper timeout immediately. |
| 12 | **Isolated set-commutativity helper lemmas** (`SetUnionComm`, `SetUnionPermute3`, `SetUnionAssocComm`) | A `forall x { assert x in A == x in B }` body *inside the large proof context* still times out even with `{:vcs_split_on_every_assert}` because the quantifier body is one big nested VCS. Moving the forall to a standalone lemma with no requires gives Z3 a fresh empty context with 1 VC. | ✓ Fixed 3 separate commutativity timeouts (lines 838, 883, 946). |
| 13 | **Dedicated P2/call-site bridge lemma** (`MarginalMass_FactorOut_P2Bridge`) — re-invoke the target lemma with explicit `requires` that match exactly the pinned hypothesis | In a large proof body Z3 cannot find a lemma postcondition `a == b` even when all the substitution facts are pinned, because the lemma's precondition check itself triggers in the large premise context. Extracting the three preconditions (`acp == AssignmentCondProb(...)`, `mergedRef == MergeAssignments(...)`, plus the actual call) into a tiny bridge lemma with explicit `requires` provides a minimal-context VC for each check. | ✓ Fixed both P2 bridge timeouts (lines 881, 944) and unlocked both cascade finals. |
| 14 | **`by`-block removal on cascade asserts** — remove `by { ... }` from a final cascade assert when all three prerequisite facts are already pinned as standalone asserts | With `{:vcs_split_on_every_assert}`, a `by` block creates a sub-VCS for every inner assert. The final cascade assert `x * a == b` needs to close against `(x*a'==b') ∧ (a'==a) ∧ (b'==b)` — three linear arithmetic equalities — and a `by` block makes each one a separate VCS that re-opens the full outer context. Removing the `by` block leaves one bare assert; with the three facts already in the hypothesis context, Z3 closes it in a single arithmetic step. | ✓ After switching to `calc` chains (technique #3), the last two timeouts (lines 932, 985) were eliminated. **Anti-pattern**: introducing named `var` aliases (`mm_zpRef`, `mm_paRef`) for MarginalMass expressions before the bridge lemma call *increased* timeout count from 2 → 5 because each `var` definition became a new VCS. Stick with inline expressions + pinned facts + `calc`. |

### What Did NOT Work in P3

| Attempt | Why it failed |
|---------|--------------|
| Bare `assert A == B by { forall x ensures ... {} }` with empty body for set commutativity | The `forall` quantifier body is one VCS inside the *large proof context* — no smaller than the outer timeout. Z3 cannot close it regardless of body size. Fix: move to a standalone lemma with no `requires`. |
| `assert acp * MM(zpRef) == MM(vzpRef) by { assert acp == ACP(...); assert vzpRef == Merge(...); }` | Creates two inner VCSes inside the by-block; each re-opens the full body context and must find the prior call postcondition, which Z3 cannot locate reliably. Fix: use a `calc` chain with one pinned step per hop. |
| Named `var mm_zpRef := MarginalMass(...)` aliases before the bridge call | Each `var` definition is itself a new VCS under `{:vcs_split_on_every_assert}` — increased timeout count 2 → 5. Fix: keep inline `MarginalMass(q, zpRef.Keys, zpRef)` expressions, pin them as standalone asserts, then close with `calc`. |
| `RealMul_Subst(x, a, b, a', b')` helper lemma called with full `MarginalMass(...)` expressions as arguments | The precondition check `x * a == b` at the call site is evaluated in the large proof context — same as the original timeout. Fix: use a `calc` chain that makes each substitution explicit in the calc step's `by {}`. |
| Re-invoking `AssignmentCondProb_As_MarginalMass_Ratio` inside the `by` clause of a bare `==` calc step | The postcondition of P2 uses `var merge := MergeAssignments(...)` in an `ensures`; Z3 cannot unify `merge.Keys` with the expression in the calc target without explicit `assert merge == vzpRef` in the same step. Fix: use two `by { ... }` calc steps — one for the P2 postcondition, one for the substitution. Ultimately superseded by technique #13. |
