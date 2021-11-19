# -*- coding: utf-8 -*-

"""Test the probability DSL."""

import itertools as itt
import unittest
from typing import Optional

from y0.dsl import (
    A,
    B,
    C,
    CounterfactualVariable,
    D,
    Distribution,
    Element,
    Fraction,
    Intervention,
    One,
    P,
    Product,
    Q,
    R,
    S,
    Sum,
    T,
    Variable,
    W,
    X,
    Y,
    Z,
    Zero,
)
from y0.parser import parse_y0

V = Variable("V")


class TestDSL(unittest.TestCase):
    """Tests for the stringifying instances of the probability DSL."""

    def assert_exp(self, expression: Element, s: Optional[str] = None):
        """Test an element can be parsed, serialized, then again."""
        e = expression.to_y0()
        if s:
            self.assertEqual(s, e, msg=repr(e))
        reconstituted = parse_y0(e)
        self.assertEqual(
            expression,
            reconstituted,
            msg=f"\nExpected: {expression.to_y0()}\nActual:   {reconstituted.to_y0()}",
        )

    def assert_text(self, s: str, expression: Element):
        """Assert the expression when it is converted to a string."""
        self.assertIsInstance(s, str)
        self.assertIsInstance(hash(expression), int)  # can the expression be hashed?
        self.assertIsInstance(expression.to_text(), str)
        self.assertIsInstance(expression.to_latex(), str)
        self.assertIsInstance(expression._repr_latex_(), str)
        self.assertEqual(s, expression.to_text(), msg=f"Expression: {repr(expression)}")
        if not isinstance(expression, (Distribution, Intervention)):
            self.assert_exp(expression)

    def test_variable(self):
        """Test the variable DSL object."""
        self.assert_text("A", Variable("A"))
        self.assert_text("A", A)  # shorthand for testing purposes

    def test_stop_the_madness(self):
        """Test that a variable can not be named "P"."""
        with self.assertRaises(ValueError):
            _ = Variable("P")
        with self.assertRaises(ValueError):
            _ = Variable("Q")

    def test_intervention(self):
        """Test the invervention DSL object."""
        self.assert_text("W*", Intervention("W", star=True))
        self.assert_text("W", Intervention("W", star=False))
        self.assert_text("W", W)  # shorthand for testing purposes
        self.assert_text("W", -W)  # An intervention from variable W

        # inversions using the unary ~ operator
        self.assert_text("W", ~Intervention("W", star=True))
        self.assert_text("W*", ~Intervention("W", star=False))
        self.assert_text("W*", ~W)

    def test_counterfactual_variable(self):
        """Test the Counterfactual Variable DSL object."""
        # Normal instantiation
        self.assert_text("Y_{W}", CounterfactualVariable("Y", interventions=(-W,)))
        self.assert_text("Y_{W*}", CounterfactualVariable("Y", interventions=(~W,)))

        # Instantiation with list-based operand to matmul @ operator
        self.assert_text("Y_{W}", Variable("Y") @ [W])
        self.assert_text("Y_{W}", Y @ [W])
        self.assert_text("Y_{W*}", Variable("Y") @ [~W])
        self.assert_text("Y_{W*}", Y @ [~W])

        # Instantiation with two variables
        self.assert_text(
            "Y_{X, W*}",
            CounterfactualVariable(
                "Y", interventions=(Intervention("X", star=False), ~Intervention("W", star=False))
            ),
        )

        # Instantiation with matmul @ operator and single operand
        self.assert_text("Y_{W}", Y @ Intervention("W", star=False))
        self.assert_text("Y_{W*}", Y @ ~Intervention("W", star=False))

        # Instantiation with matmul @ operator and list operand
        self.assert_text("Y_{X, W*}", Y @ [X, ~W])

        # Instantiation with matmul @ operator (chained)
        self.assert_text("Y_{X, W*}", Y @ X @ ~W)

    def test_star_counterfactual(self):
        """Tests for generalized counterfactual variables."""
        for expr, expected in [
            # Single variable
            (P(Y @ X), "P(Y @ X)"),
            (P(Y @ -X), "P(Y @ X)"),
            (P(Y @ ~X), "P(Y @ ~X)"),
            (P(Y @ +X), "P(Y @ ~X)"),
            #
            (P(-Y @ X), "P(-Y @ X)"),
            (P(-Y @ -X), "P(-Y @ X)"),
            (P(-Y @ ~X), "P(-Y @ ~X)"),
            (P(-Y @ +X), "P(-Y @ ~X)"),
            #
            (P(~Y @ X), "P(+Y @ X)"),
            (P(~Y @ -X), "P(+Y @ X)"),
            (P(~Y @ ~X), "P(+Y @ ~X)"),
            (P(~Y @ +X), "P(+Y @ ~X)"),
            #
            (P(+Y @ X), "P(+Y @ X)"),
            (P(+Y @ -X), "P(+Y @ X)"),
            (P(+Y @ ~X), "P(+Y @ ~X)"),
            (P(+Y @ +X), "P(+Y @ ~X)"),
            #
            (P(Y @ X | ~X, ~Y), "P(Y @ X | ~X, ~Y)"),
            (P(Y @ -X | ~X, ~Y), "P(Y @ X | ~X, ~Y)"),
            (P(Y @ +X | ~X, ~Y), "P(Y @ ~X | ~X, ~Y)"),
            (P(Y @ ~X | ~X, ~Y), "P(Y @ ~X | ~X, ~Y)"),
            #
            (P(~(Y @ ~X) | X, Y), "P(+Y @ ~X | X, Y)"),
            (P(~Y @ ~X | X, Y), "P(+Y @ ~X | X, Y)"),  # should be same as above
        ]:
            with self.subTest(expr=expected):
                self.assert_exp(expr, expected)

    def test_counterfactual_errors(self):
        """Test that if two variables with the same name are given, an error is raised, regardless of star state."""
        for a, b in itt.product([True, False], [True, False]):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                Y @ Intervention("X", star=a) @ Intervention("X", star=b)

    def test_conditional_distribution(self):
        """Test the :class:`Distribution` DSL object."""
        # Normal instantiation
        self.assert_text("A | B", Distribution((A,), (B,)))

        # Instantiation with list-based operand to or | operator
        self.assert_text("A | B", Variable("A") | (B,))
        self.assert_text("A | B", A | (B,))

        # # Instantiation with two variables
        self.assert_text("A | B, C", A | [B, C])

        # Instantiation with or | operator and single operand
        self.assert_text("A | B", Variable("A") | B)
        self.assert_text("A | B", A | B)

        # Instantiation with or | operator (chained)
        self.assert_text("A | B, C", A | B | C)

        # Counterfactual uses work basically the same.
        #  Note: @ binds more tightly than |, but it's probably better to use parentheses
        self.assert_text("Y_{W} | B", (Y @ W) | B)
        self.assert_text("Y_{W} | B", Y @ W | B)
        self.assert_text("Y_{W} | B, C", Y @ W | B | C)
        self.assert_text("Y_{W, X*} | B, C", Y @ W @ ~X | B | C)
        self.assert_text("Y_{W, X*} | B_{N*}, C", Y @ W @ ~X | B @ Intervention("N", star=True) | C)

    def test_joint_distribution(self):
        """Test the JointProbability DSL object."""
        self.assert_text("A, B", Distribution((A, B)))
        self.assert_text("A, B", A & B)
        self.assert_text("A, B, C", Distribution((A, B, C)))
        self.assert_text("A, B, C", A & B & C)

    def test_probability(self):
        """Test generation of probabilities."""
        # Make sure there are children
        with self.assertRaises(ValueError):
            Distribution(tuple())

        self.assert_text("P(A)", P(A))
        self.assert_text("P(A)", P("A"))
        self.assert_text("P(A)", P(Distribution((A,))))

        # Test markov kernel with single parent (AKA has only one child variable)
        self.assert_text("P(A | B)", P(A | B))
        self.assert_text("P(A | B)", P(Distribution((A,), (B,))))
        self.assert_text("P(A | B)", P(A | [B]))

        # Test markov kernel with multiple parents
        self.assert_text("P(A | B, C)", P(A | B, C))
        self.assert_text("P(A | B, C)", P(Distribution((A,), (B,)) | C))
        self.assert_text("P(A | B, C)", P(A | [B, C]))
        self.assert_text("P(A | B, C)", P(A | B | C))
        self.assert_text("P(A | B, C)", P(A | B & C))

        # Test simple joint distributions
        self.assert_text("P(A, B)", P((A, B)))
        self.assert_text("P(A, B)", P([A, B]))
        self.assert_text("P(A, B)", P({A, B}))
        self.assert_text("P(A, B)", P(A, B))
        self.assert_text("P(A, B)", P(A & B))
        self.assert_text("P(A, B)", P("A", "B"))
        self.assert_text("P(A, B)", P(["A", "B"]))
        self.assert_text("P(A, B, C)", P(A & B & C))
        self.assert_text("P(A, B, C)", P("A", "B", "C"))
        self.assert_text("P(A, B, C)", P(["A", "B", "C"]))
        self.assert_text("P(A, B, C)", P((name for name in "ABC")))
        self.assert_text("P(A, B, C)", P(name for name in "ABC"))
        self.assert_text("P(A, B, C)", P((Variable(name) for name in "ABC")))
        self.assert_text("P(A, B, C)", P(Variable(name) for name in "ABC"))

        # Test mixed with single conditional
        self.assert_text("P(A, B | C)", P(A, B | C))
        self.assert_text("P(A, B | C)", P(Distribution((A, B), (C,))))
        self.assert_text("P(A, B | C)", P(Distribution((A, B), (C,))))
        self.assert_text("P(A, B | C)", P(Distribution((A, B)) | C))
        self.assert_text("P(A, B | C)", P(A & B | C))

        # Test mixed with multiple conditionals
        self.assert_text("P(A, B | C, D)", P(A, B | C, D))
        self.assert_text("P(A, B | C, D)", P(Distribution((A, B), (C, D))))
        self.assert_text("P(A, B | C, D)", P(Distribution((A, B)) | C | D))
        self.assert_text("P(A, B | C, D)", P(Distribution((A, B), (C,)) | D))
        self.assert_text("P(A, B | C, D)", P(A & B | C | D))
        self.assert_text("P(A, B | C, D)", P(A & B | (C, D)))
        self.assert_text("P(A, B | C, D)", P(A & B | Distribution((C, D))))
        self.assert_text("P(A, B | C, D)", P(A & B | C & D))

    def test_conditioning_errors(self):
        """Test erroring on conditionals."""
        for expression in [
            Distribution((B,), (C,)),
            Distribution((B, C), (D,)),
            Distribution((B, C), (D, W)),
        ]:
            with self.assertRaises(TypeError):
                _ = A | expression
            with self.assertRaises(TypeError):
                _ = X & Y | expression

    def test_sum(self):
        """Test the Sum DSL object."""
        # Sum with no variables
        self.assert_text(
            "[ sum_{} P(A | B) P(C | D) ]",
            Sum(P(A | B) * P(C | D)),
        )
        # Sum with one variable
        self.assert_text(
            "[ sum_{S} P(A | B) P(C | D) ]",
            Sum(P(A | B) * P(C | D), (S,)),
        )
        # Sum with two variables
        self.assert_text(
            "[ sum_{S, T} P(A | B) P(C | D) ]",
            Sum(P(A | B) * P(C | D), (S, T)),
        )

        # CRAZY sum syntax! pycharm doesn't like this usage of __class_getitem__ though so idk if we'll keep this
        self.assert_text(
            "[ sum_{S} P(A | B) P(C | D) ]",
            Sum[S](P(A | B) * P(C | D)),
        )
        self.assert_text(
            "[ sum_{S, T} P(A | B) P(C | D) ]",
            Sum[S, T](P(A | B) * P(C | D)),
        )

        # Sum with sum inside
        self.assert_text(
            "[ sum_{S, T} P(A | B) [ sum_{R} P(C | D) ] ]",
            Sum(P(A | B) * Sum(P(C | D), (R,)), (S, T)),
        )

    def test_q(self):
        """Test the Q DSL object."""
        self.assert_text("Q[A](X)", Q[A](X))  # type: ignore
        self.assert_text("Q[A, B](X)", Q[A, B](X))  # type: ignore
        self.assert_text("Q[A](X, Y)", Q[A](X, Y))  # type: ignore
        self.assert_text("Q[A, B](X, Y)", Q[A, B](X, Y))  # type: ignore

    def test_jeremy(self):
        """Test assorted complicated objects from Jeremy."""
        self.assert_text(
            "[ sum_{W} P(X, Y_{Z*, W}) P(D) P(Z_{D}) P(W_{X*}) ]",
            Sum(P(X, (Y @ ~Z @ W)) * P(D) * P(Z @ D) * P(W @ ~X), (W,)),
        )

        self.assert_text(
            "[ sum_{W} P(X, Y_{Z*, W}) P(W_{X*}) ]",
            Sum(P(X, Y @ ~Z @ W) * P(W @ ~X), (W,)),
        )

        self.assert_text(
            "frac_{[ sum_{W} P(X, Y_{Z, W}) P(W_{X*}) ]}{[ sum_{Y} [ sum_{W} P(X, Y_{Z, W}) P(W_{X*}) ] ]}",
            Fraction(
                Sum(P(X, Y @ Z @ W) * P(W @ ~X), (W,)),
                Sum(Sum(P(X, Y @ Z @ W) * P(W @ ~X), (W,)), (Y,)),
            ),
        )

        self.assert_text(
            "[ sum_{D} P(X, Y_{Z*, W}) P(D) P(Z_{D}) P(W_{X*}) ]",
            Sum(P(X, Y @ ~Z @ W) * P(D) * P(Z @ D) * P(W @ ~X), (D,)),
        )

        self.assert_text(
            "[ sum_{D, V, W, Z} [ sum_{} P(W | X) ] [ sum_{} [ sum_{V, W, X, Y, Z} P(D, V, W, X, Y, Z) ] ]"
            " [ sum_{} P(Z | D, V) ] [ sum_{} [ sum_{X} P(Y | D, V, W, X, Z) P(X) ] ]"
            " [ sum_{} [ sum_{D, W, X, Y, Z} P(D, V, W, X, Y, Z) ] ] ]",
            Sum[W, D, Z, V](
                Sum(P(W | X))
                * Sum(Sum[X, W, Z, Y, V](P(D, V, W, X, Y, Z)))
                * Sum(P(Z | D, V))
                * Sum(Sum[X](P(Y | D, V, W, X, Z) * P(X)))
                * Sum(Sum[X, W, D, Z, Y](P(X, W, D, Z, Y, V))),
            ),
        )

    def test_api(self):
        """Test the high-level API for getting variables."""
        for expression, variables in [
            (A @ B, {A @ B, -B}),
            (A @ ~B, {A @ ~B, ~B}),
            (P(A @ B), {A @ B, -B}),
            (P(A @ ~B), {A @ ~B, ~B}),
            (P(A @ ~B) * P(C), {A @ ~B, ~B, C}),
            (Sum[S](P(A @ ~B) * P(C)), {A @ ~B, ~B, C, S}),
            (Sum[S, T](P(A @ ~B) * P(C)), {A @ ~B, ~B, C, S, T}),
            (One() / P(A @ B), {A @ B, -B}),
            (P(B) / P(A @ ~B), {A @ ~B, B, ~B}),
            (P(Y | X) * P(X) / P(Y), {X, Y}),
            (Q[A, B](C, D), {A, B, C, D}),  # type: ignore
        ]:
            with self.subTest(expression=str(expression)):
                self.assertEqual(variables, expression.get_variables())


class TestCounterfactual(unittest.TestCase):
    """Tests for counterfactuals."""

    def test_event_semantics(self):
        """Check DSL semantics."""
        for expr, counterfactual_star, intervention_star in [
            (X @ X, None, False),
            (-X @ X, False, False),
            (+X @ X, True, False),
            (~X @ X, True, False),
        ]:
            with self.subTest(expr=expr.to_y0()):
                self.assertIsInstance(expr, CounterfactualVariable)
                self.assertEqual(counterfactual_star, expr.star)
                self.assertEqual(1, len(expr.interventions))
                self.assertEqual(intervention_star, expr.interventions[0].star)

    def test_event_failures(self):
        """Check for failure to determine tautology/inconsistent."""
        for expr in [
            # Opposite variable
            X @ Y,
            X @ +Y,
            X @ -Y,
            X @ ~Y,
            # Same variable
            X @ X,
            X @ +X,
            X @ -X,
            X @ ~X,
        ]:
            with self.subTest(expr=expr.to_y0()):
                self.assertIsInstance(expr, CounterfactualVariable)
                self.assertFalse(expr.is_event())
                with self.assertRaises(ValueError):
                    expr.has_tautology()
                with self.assertRaises(ValueError):
                    expr.is_inconsistent()

    def test_tautology(self):
        """Check for tautologies."""
        for expr, status in [
            # Different Variable
            (~X @ Y, False),
            (~X @ ~Y, False),
            # Same variable, self.star is False
            (-X @ X, False),
            (-X @ +X, False),
            (-X @ -X, True),
            (-X @ ~X, True),
            # Same variable, self.star is True
            (+X @ X, True),
            (+X @ +X, True),
            (+X @ -X, False),
            (+X @ ~X, False),
            # Same variable, self.star is True
            (~X @ X, False),
            (~X @ +X, False),
            (~X @ -X, True),
            (~X @ ~X, True),
        ]:
            with self.subTest(expr=expr.to_y0()):
                self.assertIsInstance(expr, CounterfactualVariable)
                self.assertEqual(status, expr.has_tautology())


class TestSafeConstructors(unittest.TestCase):
    """Test that the .safe() constructors work properly."""

    def test_do2_intervention(self):
        """Test the do-calculus level two interventions."""
        self.assertEqual(P[X](Y), P(Y @ X))
        self.assertEqual(P[X](Y, Z), P(Y @ X & Z @ X))
        self.assertEqual(P[X](Y | Z), P(Y @ X | Z @ X))

        # stack interventions mixed with $L_3$ notation
        self.assertEqual(P[X](Y @ Z), P(Y @ Z @ X))

        # mixed with $L_3$, where each variable can have different kinds of interventions
        self.assertEqual(P[X](Y @ Z | W), P(Y @ Z @ X | W @ X))

    def test_sum(self):
        """Test the :meth:`Sum.safe` constructor."""
        self.assertEqual(Sum(P(X, Y), (X,)), Sum.safe(P(X, Y), (X,)))
        self.assertEqual(Sum(P(X, Y), (X,)), Sum.safe(P(X, Y), [X]))
        self.assertEqual(Sum(P(X, Y), (X,)), Sum.safe(P(X, Y), {X}))
        self.assertEqual(Sum(P(X, Y), (X,)), Sum.safe(P(X, Y), X))

        self.assertEqual(Sum(P(X, Y, Z), (X, Y)), Sum.safe(P(X, Y, Z), (v for v in [X, Y])))

    def test_product(self):
        """Test the :meth:`Product.safe` constructor."""
        p = P(X, Y)
        self.assertEqual(Product((p,)), Product.safe(p))
        self.assertEqual(Product((p,)), Product.safe((p,)))
        self.assertEqual(Product((p,)), Product.safe([p]))
        self.assertEqual(Product((p,)), Product.safe({p}))

        self.assertEqual(Product((P(X), P(Y))), Product.safe(P(v) for v in [X, Y]))


zero = Zero()


class TestZero(unittest.TestCase):
    """Tests for zero."""

    exprs = [
        One(),
        Zero(),
        P(A),
        P(A) * P(B),
        P(A) / P(B),
        Sum.safe(P(A), [A]),
    ]

    def test_divide_failure(self):
        """Test failure is thron on division by zero."""
        for expr in self.exprs:
            with self.subTest(expr=expr), self.assertRaises(ZeroDivisionError):
                expr / zero

    def test_identity(self):
        """Test that zero divided by anything is zero."""
        for expr in self.exprs:
            if isinstance(expr, Zero):
                continue  # would raise divides by zero
            with self.subTest(expr=expr):
                self.assertEqual(zero, zero / expr)

    def test_multiply(self):
        """Test other operations."""
        for expr in self.exprs:
            with self.subTest(expr=expr.to_y0(), direction="right"):
                self.assertEqual(zero, zero * expr, msg=f"Got {zero * expr}")
            with self.subTest(expr=expr.to_y0(), direction="left"):
                self.assertEqual(zero, expr * zero, msg=f"Got {expr * zero}")
