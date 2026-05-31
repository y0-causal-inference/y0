# Fix Dafny timeout in Identification.IDImpl

**Suggested commit title:**
`fix(dafny): split Line 4 IDImpl proof to avoid verifier timeout`

## Commit message body

Diagnose and fix a Dafny verifier timeout in `Identification.IDImpl` caused by
the Line 4 decomposition branch in `identification.dfy`.

The timeout was isolated to the `IDLine4Check` call in the `|C(G \ X)| > 1`
case, where Dafny spent its time discharging quantifier-heavy preconditions for
the sequence returned by `SetOfSetsToSeq(CComponentsWithout(sm, X))`.

Add a small helper lemma, `IDLine4ComponentsReady`, that exposes the two facts
the recursive helpers need for every component in that sequence:

- each component is a subset of `SMNodes(sm)`
- each component is non-empty

Use that lemma immediately before the Line 4 recursive calls and assert the
elementwise facts on the local `comps` sequence before calling `IDLine4Check`
and `IDLine4Product`.

This keeps the control flow unchanged while reducing the cost of the proof
obligations enough for Dafny to verify within the default 30 second limit.

## Session summary

### Investigation

We reproduced the original failure with:

```bash
dafny verify src/dafny/*.dfy
```

and confirmed the reported timeout in `Identification.IDImpl`.

We then narrowed the problem with:

```bash
dafny verify src/dafny/identification.dfy --verification-time-limit:30
dafny verify src/dafny/identification.dfy --verification-time-limit:30 --isolate-assertions --progress:Batch --log-format:text
dafny verify src/dafny/identification.dfy --verification-time-limit:30 --isolate-assertions --filter-position=:331-423 --progress:Batch
dafny verify src/dafny/identification.dfy --verification-time-limit:30 --isolate-assertions --filter-position=:386 --log-format:text
```

That showed the expensive verification conditions were both at the call to
`IDLine4Check` in the Line 4 branch of `IDImpl`, specifically as
`function precondition satisfied` batches.

We also checked whether this was only a time-budget issue by increasing the
limit to 120 seconds. The same obligations still timed out, which confirmed a
proof-structure problem rather than a low timeout budget.

### Root cause

The verifier was repeatedly reconstructing quantified facts needed by the
preconditions of `IDLine4Check` and `IDLine4Product`:

- `forall i :: 0 <= i < |components| ==> components[i] <= SMNodes(sm)`
- `forall i :: 0 <= i < |components| ==> components[i] != {}`

Those facts were available only indirectly through:

- `CComponentsWithout(sm, X)`
- the definition and partition properties of `CComponents`
- the axiomatized set-to-sequence correspondence of `SetOfSetsToSeq`

That combination was enough to cause quantifier blow-up in the precondition VCs
at the Line 4 call site.

### Code change

In `src/dafny/identification.dfy` we added:

- `IDLine4ComponentsReady(sm, X)`

This lemma states directly that every element of
`SetOfSetsToSeq(CComponentsWithout(sm, X))` is a subset of `SMNodes(sm)` and is
non-empty.

We then updated the Line 4 branch in `IDImpl` to:

- invoke `IDLine4ComponentsReady(sm, X)`
- bind `comps := SetOfSetsToSeq(ccompsGX)`
- assert the two quantified facts over `comps`
- call `IDLine4Check` and `IDLine4Product` only after those assertions

This is a proof-splitting change only. It does not alter the algorithm,
recursion, or public interface.

### Verification

We verified the fix in increasing scope:

```bash
dafny verify src/dafny/identification.dfy --verification-time-limit:30 --isolate-assertions --filter-position=:383-390 --progress:Batch
```

Result: the Line 4 branch verified cleanly and the previous timeout at the
`IDLine4Check` call disappeared.

```bash
dafny verify src/dafny/identification.dfy --verification-time-limit:30
```

Result: `34 verified, 0 errors`

```bash
dafny verify src/dafny/*.dfy
```

Result: `98 verified, 0 errors`

## Files touched in this session

- `src/dafny/identification.dfy`
- `docs/plans/2026-05-09-dafny-idimpl-timeout-fix.md`

## Notes

This session focused on the `IDImpl` timeout only. Other unstaged workspace
changes already present in the repository were not modified as part of this fix.
