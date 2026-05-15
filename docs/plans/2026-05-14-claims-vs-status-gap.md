I’ll quickly reconcile the current code state with the status-vs-claim table so
the gap analysis and plan are accurate as of now.

Ran terminal command: git status -sb

Ran terminal command: rg -n
"\{:\s\*axiom\}|assume|IDToIR\(|ExtractedFullUnavailableError|identify_handwritten|unsupported
shape|Line 4: conservative" src/dafny/identification.dfy
src/dafny/identification_executable_core.dfy src/dafny/id_full_extracted.dfy
src/y0/algorithm/identify/id_generated.py

Read
[](file:///Users/zuck016/Projects/CausalInference/y0-causal-inference/y0/docs/Midspiral-Formal-ID-Experiment.md#1-1),
lines 1 to 80

Yes, and this is a great moment to do it because the current gap is now very
concrete.

**Gap Discussion** Here is the real delta between current status and strong
claims:

1. Extraction presence vs extraction completeness

- Status: Per-line and consolidated extracted runtimes exist:
  id_full_extracted.dfy
- Gap: Consolidated runtime still contains conservative/special-case logic and
  an unsupported-shape exit: id_full_extracted.dfy, id_full_extracted.dfy

2. Generated engine vs fully extracted execution

- Status: Generated path prefers extracted runtime
- Gap: It still falls back to handwritten runtime: id_generated.py,
  id_generated.py

3. Axiom boundary clarity vs complete formal proof

- Status: You now have a clean split with an axiom-free executable core:
  identification_executable_core.dfy
- Gap: The theorem file remains heavily axiomatized: identification.dfy

4. Documentation accuracy gap (new)

- Status-vs-claim section still says executable entrypoint lives in
  identification.dfy
- Current state: it now lives in identification_executable_core.dfy
- Evidence: Midspiral-Formal-ID-Experiment.md,
  identification_executable_core.dfy

**Plan to Close the Status-vs-Claim Gap**

1. Phase A: Align claims with current architecture (1 day)

- Update status table and narrative to reflect the new split (executable core vs
  theorem layer)
- Add one “Trust Boundary” subsection to make the split explicit
- Target files: Midspiral-Formal-ID-Experiment.md,
  2026-05-09-id-autogen-phase-checklist.md
- Exit criterion: no stale mention of identification.dfy as executable
  entrypoint

2. Phase B: Quantify runtime completeness (2-3 days)

- Add a coverage dashboard for ID line-shapes handled by consolidated runtime
- Measure fallback frequency from generated to handwritten on oracle corpus
- Target files: test_id_generated_parity.py, id_cases.v1.json
- Exit criterion: published metric such as “X% of corpus handled without
  handwritten fallback”

3. Phase C: Reduce consolidated runtime gaps (1-2 weeks)

- Replace conservative Line 4 special-case with broader rule handling
- Reduce unsupported-shape exits with additional executable branches
- Target file: id_full_extracted.dfy
- Exit criterion: fallback rate drops release-over-release and parity remains
  green

4. Phase D: Harden generated-engine routing contract (3-5 days)

- Add explicit tests that fail if fallback occurs for designated
  “must-be-extracted” cases
- Keep fallback for safety, but make regressions visible
- Target files: id_generated.py, test_id_generated_parity.py
- Exit criterion: stable must-be-extracted suite with zero fallback

5. Phase E: Axiom burn-down for theorem layer (ongoing)

- Track axioms by category: structural, factorization, completeness
- Prioritize replacing the highest-value assumptions first
- Target file: identification.dfy
- Exit criterion: monotonic reduction in axiom count and unchanged theorem
  verification pass

**Suggested Milestones**

1. M1: Docs truthful and architecture-explicit
2. M2: Runtime completeness measured, not anecdotal
3. M3: Consolidated runtime handles core corpus without handwritten fallback
4. M4: Axiom count trendline decreasing per sprint

If you want, I can execute Phase A immediately now (docs alignment only) in one
commit, then start Phase B by adding the fallback/coverage metrics tests.
