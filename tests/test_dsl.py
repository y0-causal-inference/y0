# -*- coding: utf-8 -*-

"""Test the probability DSL."""

import itertools as itt
import unittest

from y0.dsl import (
    A, B, C, CounterfactualVariable, D, Distribution, Fraction, Intervention, One, P, Q, R, S, Sum, T, Variable, W, X,
    Y, Z,
)

V = Variable('V')


class TestDSL(unittest.TestCase):
    """Tests for the stringifying instances of the probability DSL."""

    def assert_text(self, s: str, expression):
        """Assert the expression when it is converted to a string."""
        self.assertIsInstance(s, str)
        self.assertIsInstance(hash(expression), int)  # can the expression be hashed?
        self.assertIsInstance(expression.to_text(), str)
        self.assertIsInstance(expression.to_latex(), str)
        self.assertIsInstance(expression._repr_latex_(), str)
        self.assertEqual(s, expression.to_text(), msg=f'Expression: {repr(expression)}')

    def test_variable(self):
        """Test the variable DSL object."""
        self.assert_text('A', Variable('A'))
        self.assert_text('A', A)  # shorthand for testing purposes

    def test_stop_the_madness(self):
        """Test that a variable can not be named "P"."""
        with self.assertRaises(ValueError):
            _ = Variable('P')

    def test_intervention(self):
        """Test the invervention DSL object."""
        self.assert_text('W*', Intervention('W', True))
        self.assert_text('W', Intervention('W', False))
        self.assert_text('W', Intervention('W'))  # False is the default
        self.assert_text('W', W)  # shorthand for testing purposes
        self.assert_text('W', -W)  # An intervention from variable W

        # inversions using the unary ~ operator
        self.assert_text('W', ~Intervention('W', True))
        self.assert_text('W*', ~Intervention('W', False))  # False is still the default
        self.assert_text('W*', ~Intervention('W'))
        self.assert_text('W*', ~W)

    def test_counterfactual_variable(self):
        """Test the Counterfactual Variable DSL object."""
        # Normal instantiation
        self.assert_text('Y_{W}', CounterfactualVariable('Y', (-W,)))
        self.assert_text('Y_{W*}', CounterfactualVariable('Y', (~W,)))

        # Instantiation with list-based operand to matmul @ operator
        self.assert_text('Y_{W}', Variable('Y') @ [W])
        self.assert_text('Y_{W}', Y @ [W])
        self.assert_text('Y_{W*}', Variable('Y') @ [~W])
        self.assert_text('Y_{W*}', Y @ [~W])

        # Instantiation with two variables
        self.assert_text('Y_{X,W*}', CounterfactualVariable('Y', (Intervention('X'), ~Intervention('W'))))

        # Instantiation with matmul @ operator and single operand
        self.assert_text('Y_{W}', Y @ Intervention('W'))
        self.assert_text('Y_{W*}', Y @ ~Intervention('W'))

        # Instantiation with matmul @ operator and list operand
        self.assert_text('Y_{X,W*}', Y @ [X, ~W])

        # Instantiation with matmul @ operator (chained)
        self.assert_text('Y_{X,W*}', Y @ X @ ~W)

    def test_counterfactual_errors(self):
        """Test that if two variables with the same name are given, an error is raised, regardless of star state."""
        for a, b in itt.product([True, False], [True, False]):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                Y @ Intervention('X', star=a) @ Intervention('X', star=b)

    def test_conditional_distribution(self):
        """Test the :class:`Distribution` DSL object."""
        # Normal instantiation
        self.assert_text('A|B', Distribution((A,), (B,)))

        # Instantiation with list-based operand to or | operator
        self.assert_text('A|B', Variable('A') | (B,))
        self.assert_text('A|B', A | (B,))

        # # Instantiation with two variables
        self.assert_text('A|B,C', A | [B, C])

        # Instantiation with or | operator and single operand
        self.assert_text('A|B', Variable('A') | B)
        self.assert_text('A|B', A | B)

        # Instantiation with or | operator (chained)
        self.assert_text('A|B,C', A | B | C)

        # Counterfactual uses work basically the same.
        #  Note: @ binds more tightly than |, but it's probably better to use parentheses
        self.assert_text('Y_{W}|B', (Y @ W) | B)
        self.assert_text('Y_{W}|B', Y @ W | B)
        self.assert_text('Y_{W}|B,C', Y @ W | B | C)
        self.assert_text('Y_{W,X*}|B,C', Y @ W @ ~X | B | C)
        self.assert_text('Y_{W,X*}|B_{N*},C', Y @ W @ ~X | B @ Intervention('N', True) | C)

    def test_joint_distribution(self):
        """Test the JointProbability DSL object."""
        self.assert_text('A,B', Distribution((A, B)))
        self.assert_text('A,B', A & B)
        self.assert_text('A,B,C', Distribution((A, B, C)))
        self.assert_text('A,B,C', A & B & C)

    def test_probability(self):
        """Test generation of probabilities."""
        # Make sure there are children
        with self.assertRaises(ValueError):
            Distribution(tuple())

        # Test markov kernels (AKA has only one child variable)
        self.assert_text('P(A|B)', P(Distribution((A,), (B,))))
        self.assert_text('P(A|B)', P(A | [B]))
        self.assert_text('P(A|B,C)', P(Distribution((A,), (B,)) | C))
        self.assert_text('P(A|B,C)', P(A | [B, C]))
        self.assert_text('P(A|B,C)', P(A | B | C))
        self.assert_text('P(A|B,C)', P(A | B & C))

        # Test simple joint distributions
        self.assert_text('P(A,B)', P((A, B)))
        self.assert_text('P(A,B)', P(A, B))
        self.assert_text('P(A,B)', P(A & B))
        self.assert_text('P(A,B,C)', P(A & B & C))

        # Test mixed with single conditional
        self.assert_text('P(A,B|C)', P(Distribution((A, B), (C,))))
        self.assert_text('P(A,B|C)', P(Distribution((A, B), (C,))))
        self.assert_text('P(A,B|C)', P(Distribution((A, B)) | C))
        self.assert_text('P(A,B|C)', P(A & B | C))

        # Test mixed with multiple conditionals
        self.assert_text('P(A,B|C,D)', P(Distribution((A, B), (C, D))))
        self.assert_text('P(A,B|C,D)', P(Distribution((A, B)) | C | D))
        self.assert_text('P(A,B|C,D)', P(Distribution((A, B), (C,)) | D))
        self.assert_text('P(A,B|C,D)', P(A & B | C | D))
        self.assert_text('P(A,B|C,D)', P(A & B | (C, D)))
        self.assert_text('P(A,B|C,D)', P(A & B | Distribution((C, D))))
        self.assert_text('P(A,B|C,D)', P(A & B | C & D))

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
            "[ sum_{} P(A|B) P(C|D) ]",
            Sum(P(A | B) * P(C | D)),
        )
        # Sum with one variable
        self.assert_text(
            "[ sum_{S} P(A|B) P(C|D) ]",
            Sum(P(A | B) * P(C | D), (S,)),
        )
        # Sum with two variables
        self.assert_text(
            "[ sum_{S,T} P(A|B) P(C|D) ]",
            Sum(P(A | B) * P(C | D), (S, T)),
        )

        # CRAZY sum syntax! pycharm doesn't like this usage of __class_getitem__ though so idk if we'll keep this
        self.assert_text(
            "[ sum_{S} P(A|B) P(C|D) ]",
            Sum[S](P(A | B) * P(C | D)),
        )
        self.assert_text(
            "[ sum_{S,T} P(A|B) P(C|D) ]",
            Sum[S, T](P(A | B) * P(C | D)),
        )

        # Sum with sum inside
        self.assert_text(
            "[ sum_{S,T} P(A|B) [ sum_{R} P(C|D) ] ]",
            Sum(P(A | B) * Sum(P(C | D), (R,)), (S, T)),
        )

    def test_q(self):
        """Test the Q DSL object."""
        self.assert_text("Q[A](X)", Q[A](X))
        self.assert_text("Q[A,B](X)", Q[A, B](X))
        self.assert_text("Q[A](X,Y)", Q[A](X, Y))
        self.assert_text("Q[A,B](X,Y)", Q[A, B](X, Y))

    def test_jeremy(self):
        """Test assorted complicated objects from Jeremy."""
        self.assert_text(
            '[ sum_{W} P(Y_{Z*,W},X) P(D) P(Z_{D}) P(W_{X*}) ]',
            Sum(P((Y @ ~Z @ W) & X) * P(D) * P(Z @ D) * P(W @ ~X), (W,)),
        )

        self.assert_text(
            '[ sum_{W} P(Y_{Z*,W},X) P(W_{X*}) ]',
            Sum(P(Y @ ~Z @ W & X) * P(W @ ~X), (W,)),
        )

        self.assert_text(
            'frac_{[ sum_{W} P(Y_{Z,W},X) P(W_{X*}) ]}{[ sum_{Y} [ sum_{W} P(Y_{Z,W},X) P(W_{X*}) ] ]}',
            Fraction(
                Sum(P(Y @ Z @ W & X) * P(W @ ~X), (W,)),
                Sum(Sum(P(Y @ Z @ W & X) * P(W @ ~X), (W,)), (Y,)),
            ),
        )

        self.assert_text(
            '[ sum_{D} P(Y_{Z*,W},X) P(D) P(Z_{D}) P(W_{X*}) ]',
            Sum(P(Y @ ~Z @ W & X) * P(D) * P(Z @ D) * P(W @ ~X), (D,)),
        )

        self.assert_text(
            '[ sum_{W,D,Z,V} [ sum_{} P(W|X) ] [ sum_{} [ sum_{X,W,Z,Y,V} P(X,W,D,Z,Y,V) ] ]'
            ' [ sum_{} P(Z|D,V) ] [ sum_{} [ sum_{X} P(Y|X,D,V,Z,W) P(X) ] ]'
            ' [ sum_{} [ sum_{X,W,D,Z,Y} P(X,W,D,Z,Y,V) ] ] ]',
            Sum[W, D, Z, V](
                Sum(P(W | X))
                * Sum(Sum[X, W, Z, Y, V](P(X, W, D, Z, Y, V)))
                * Sum(P(Z | [D, V]))
                * Sum(Sum[X](P(Y | [X, D, V, Z, W]) * P(X)))
                * Sum(Sum[X, W, D, Z, Y](P(X, W, D, Z, Y, V))),
            ),
        )

        '''
        [[sum_{D,Z,V} [sum_{} [sum_{X,W,Z,Y,V} P(X,W,D,Z,Y,V)]][sum_{}P(Z|D,V)][sum_{} [sum_{X} P(Y|X,D,V,Z,W)P(X|)]]
        [sum_{} [sum_{X,W,D,Z,Y} P(X,W,D,Z,Y,V)]]]]/[ sum_{Y}[sum_{D,Z,V} [sum_{} [sum_{X,W,Z,Y,V} P(X,W,D,Z,Y,V)]]
        [sum_{}P(Z|D,V)][sum_{} [sum_{X} P(Y|X,D,V,Z,W)P(X|)]][sum_{} [sum_{X,W,D,Z,Y} P(X,W,D,Z,Y,V)]]]]
        '''

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
            (Q[A, B](C, D), {A, B, C, D}),
        ]:
            with self.subTest(expression=str(expression)):
                self.assertEqual(variables, expression.get_variables())
