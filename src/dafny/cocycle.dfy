// Counterfactual Cocycle — Dafny Specification
// Formalises Dance & Bloem-Reddy "Counterfactual Cocycles" (JMLR 2025).
//
// A cocycle is a transport family  T : X × X × Y → Y  satisfying
//   (ID)  T(x, x,  y)              = y              for all x, y
//   (PI)  T(x'', x', T(x', x, y)) = T(x'', x, y)  for all x, x', x'', y
//
// X is the treatment space; Y is the outcome (response) space.
// The `(==)` type-parameter annotation tells Dafny that the type
// supports decidable equality, which is required for `==` in predicates.

module CausalCocycle {

  // ------------------------------------------------------------------
  // 1.  Core predicates
  // ------------------------------------------------------------------

  /// (ID)  Transporting from treatment x to itself leaves the outcome unchanged.
  predicate Identity<X(==), Y(==)>(T: (X, X, Y) -> Y) {
    forall x: X, y: Y :: T(x, x, y) == y
  }

  /// (PI)  The composed transport through any intermediate x' equals the
  ///       direct transport: the result is path-independent.
  predicate PathIndependence<X(==), Y(==)>(T: (X, X, Y) -> Y) {
    forall x: X, x': X, x'': X, y: Y ::
      T(x'', x', T(x', x, y)) == T(x'', x, y)
  }

  /// T is a cocycle iff it satisfies both (ID) and (PI).
  predicate IsCocycle<X(==), Y(==)>(T: (X, X, Y) -> Y) {
    Identity(T) && PathIndependence(T)
  }

  // ------------------------------------------------------------------
  // 2.  Consequences of the cocycle axioms
  // ------------------------------------------------------------------

  /// Invertibility: T(x, x', ·) and T(x', x, ·) are mutual inverses.
  ///
  /// Proof sketch:
  ///   T(x', x, T(x, x', y))
  ///     = T(x', x', y)   [by (PI) with x'' ← x', x' ← x, x ← x']
  ///     = y              [by (ID)]
  lemma Invertibility<X(==), Y(==)>(T: (X, X, Y) -> Y, x: X, x': X, y: Y)
    requires IsCocycle(T)
    ensures T(x', x, T(x, x', y)) == y
  {
    calc {
      T(x', x, T(x, x', y));
    ==  // (PI): T(x'', x', T(x', x, ·)) = T(x'', x, ·), here x'' ← x', x ↔ x'
      T(x', x', y);
    ==  // (ID)
      y;
    }
  }

  /// Symmetry of invertibility (the other direction).
  lemma InvertibilitySymm<X(==), Y(==)>(T: (X, X, Y) -> Y, x: X, x': X, y: Y)
    requires IsCocycle(T)
    ensures T(x, x', T(x', x, y)) == y
  {
    Invertibility(T, x', x, y);
  }

  /// Transitivity: T(x'', x, ·) is the composition of x → x' and x' → x''.
  /// This is exactly (PI), stated as a named lemma for clarity.
  lemma Transitivity<X(==), Y(==)>(T: (X, X, Y) -> Y, x: X, x': X, x'': X, y: Y)
    requires IsCocycle(T)
    ensures T(x'', x', T(x', x, y)) == T(x'', x, y)
  {}  // Dafny discharges this directly from PathIndependence.

  // ------------------------------------------------------------------
  // 3.  Coboundary construction:  T(x, x', y) := f_x( f_{x'}⁻¹(y) )
  //
  //     Theorem 3 in the paper shows every cocycle factors this way.
  //     Here we prove the converse: every such construction is a cocycle.
  // ------------------------------------------------------------------

  /// A family of bijections on Y, one per treatment level x,
  /// together with their pointwise inverses.
  predicate InverseFamily<X(==), Y(==)>(
    f:    X -> (Y -> Y),
    fInv: X -> (Y -> Y)
  ) {
    forall x: X, y: Y ::
      f(x)(fInv(x)(y)) == y &&   // f_x ∘ fInv_x = id
      fInv(x)(f(x)(y)) == y      // fInv_x ∘ f_x = id
  }

  /// The coboundary built from (f, fInv): T(x, x', y) = f_x( fInv_{x'}(y) ).
  function Coboundary<X(==), Y(==)>(
    f:    X -> (Y -> Y),
    fInv: X -> (Y -> Y)
  ): (X, X, Y) -> Y
  {
    (x, x', y) => f(x)(fInv(x')(y))
  }

  /// Theorem: every coboundary is a cocycle.
  ///
  /// (ID):  Coboundary(x, x, y) = f_x( fInv_x(y) ) = y
  ///
  /// (PI):  Coboundary(x'', x', Coboundary(x', x, y))
  ///          = f_{x''}( fInv_{x'}( f_{x'}( fInv_x(y) ) ) )
  ///          = f_{x''}( fInv_x(y) )    ← fInv_{x'} ∘ f_{x'} = id
  ///          = Coboundary(x'', x, y)
  lemma CoboundaryIsCocycle<X(==), Y(==)>(
    f:    X -> (Y -> Y),
    fInv: X -> (Y -> Y)
  )
    requires InverseFamily(f, fInv)
    ensures IsCocycle(Coboundary(f, fInv))
  {
    var T := Coboundary(f, fInv);

    // --- (ID) ---
    assert Identity(T) by {
      forall x: X, y: Y ensures T(x, x, y) == y {
        // T(x, x, y) = f(x)(fInv(x)(y)) = y  by InverseFamily
        assert f(x)(fInv(x)(y)) == y;
      }
    }

    // --- (PI) ---
    assert PathIndependence(T) by {
      forall x: X, x': X, x'': X, y: Y
          ensures T(x'', x', T(x', x, y)) == T(x'', x, y)
      {
        var u: Y := fInv(x)(y);             // u = fInv_x(y)
        assert fInv(x')(f(x')(u)) == u;     // fInv_{x'} ∘ f_{x'} = id (InverseFamily)
        // T(x', x, y)  = f(x')(u)
        // T(x'', x', f(x')(u)) = f(x'')(fInv(x')(f(x')(u))) = f(x'')(u)
        // T(x'', x, y) = f(x'')(u)
        assert T(x'', x', T(x', x, y)) == f(x'')(u);
        assert T(x'', x, y)             == f(x'')(u);
      }
    }
  }

  // ------------------------------------------------------------------
  // 4.  Trivial example: the constant cocycle T(x, x', y) = y
  // ------------------------------------------------------------------

  /// The constant transport (do nothing regardless of treatment change) is a cocycle.
  lemma ConstantCocycleIsValid<X(==), Y(==)>()
    ensures IsCocycle<X, Y>((x, x', y) => y)
  {}  // Both (ID) and (PI) hold trivially since T is the constant y.

}
