# -*- coding: utf-8 -*-

"""Tests for the identify algorithm."""

import itertools as itt
import unittest

from y0.algorithm.identify import Fail, idc, identify
from y0.algorithm.identify.id_std import (
    line_1,
    line_2,
    line_3,
    line_4,
    line_5,
    line_6,
    line_7,
)
from y0.dsl import Expression, P, Product, Sum, Variable, X, Y, Y1, Z, get_outcomes_and_treatments
from y0.examples import (
    figure_6a,
    line_1_example,
    line_2_example,
    line_3_example,
    line_4_example,
    line_5_example,
    line_6_example,
    line_7_example,
)
from y0.mutate import canonicalize

P_XY = P(X, Y)
P_XYZ = P(X, Y, Z)
M = Variable("M")


class TestIdentify(unittest.TestCase):
    """Test cases from https://github.com/COVID-19-Causal-Reasoning/Y0/blob/master/ID_whittemore.ipynb."""

    def assert_expr_equal(self, expected: Expression, actual: Expression) -> None:
        """Assert that two expressions are the same."""
        expected_outcomes, expected_treatments = get_outcomes_and_treatments(query=expected)
        actual_outcomes, actual_treatments = get_outcomes_and_treatments(query=actual)
        self.assertEqual(expected_treatments, actual_treatments)
        self.assertEqual(expected_outcomes, actual_outcomes)
        ordering = tuple(expected.get_variables())
        expected_canonical = canonicalize(expected, ordering)
        actual_canonical = canonicalize(actual, ordering)
        self.assertEqual(
            expected_canonical,
            actual_canonical,
            msg=f"\nExpected: {str(expected_canonical)}\nActual:   {str(actual_canonical)}",
        )

    def test_idc(self):
        r"""Test that the IDC algorithm works correctly."""
        for identification in figure_6a.identifications:
            id_in = identification["id_in"][0]
            id_out = identification["id_out"][0]
            self.assert_expr_equal(
                expected=id_out.estimand,
                actual=idc(id_in),
            )

    def test_line_1(self):
        r"""Test that line 1 of ID algorithm works correctly.

        If no action has been taken, the effect on :math:`\mathbf Y` is just the marginal of
        the observational distribution :math:`P(\mathbf v)` on :math:`\mathbf Y`.
        """
        for identification in line_1_example.identifications:
            self.assert_expr_equal(
                expected=identification["id_out"][0].estimand,
                actual=line_1(identification["id_in"][0]),
            )
            self.assert_expr_equal(
                identification["id_out"][0].estimand,
                identify(identification["id_in"][0]),
            )

    def test_line_2(self):
        r"""Test line 2 of the identification algorithm.

        2. If we are interested in the effect on :math:`\mathbf Y`, it is sufficient to restrict our
        attention on the parts of the model ancestral to :math:`\mathbf Y`.
        """
        for identification in line_2_example.identifications:
            id_out = identification["id_out"][0]
            self.assertEqual(
                id_out,
                line_2(identification["id_in"][0]),
            )
            self.assert_expr_equal(
                Sum.safe(expression=P(Y, Z), ranges=[Z]),
                identify(identification["id_in"][0]),
            )

    def test_line_3(self):
        r"""Test line 3 of the identification algorithm.

        3. Forces an action on any node where such an action would have no effect on :math:`\mathbf Y`—assuming
        we already acted on :math:`\mathbf X`. Since actions remove incoming arrows, we can view line 3 as
        simplifying the causal graph we consider by removing certain arcs from the graph, without
        affecting the overall answer.
        """
        for identification in line_3_example.identifications:
            self.assertEqual(
                identification["id_out"][0],
                line_3(identification["id_in"][0]),
            )
            self.assert_expr_equal(
                P(Y | (X, Z)),
                identify(identification["id_in"][0]),
            )

    def test_line_4(self):
        r"""Test line 4 of the identification algorithm.

        4. The key line of the algorithm, it decomposes the problem into a set of smaller problems
        using the key property of *c-component factorization* of causal models. If the entire graph
        is a single C-component already, further problem decomposition is impossible, and we must
        provide base cases. :math:`\mathbf{ID}` has three base cases.
        """
        for identification in line_4_example.identifications:
            actuals = line_4(identification["id_in"][0])
            expecteds = identification["id_out"]
            self.assertEqual(len(expecteds), len(actuals))
            match = []
            for expected, actual in itt.product(expecteds, actuals):
                if expected == actual:
                    match.append((expected, actual))
            self.assertEqual(len(expecteds), len(match), msg="Could not match identifications")

            # TODO @cthoyt
            self.assert_expr_equal(
                Sum.safe(
                    ranges=[M, Z],
                    expression=Product.safe(
                        [
                            P(M | (Z, X)),
                            P(Y | (M, Z, X)),
                            Sum(P(Z)),
                        ]
                    ),
                ),
                identify(identification["id_in"][0]),
            )

    def test_line_5(self):
        r"""Test line 5 of the identification algorithm.

        5. Fails because it finds two C-components, the graph :math:`G` itself, and a subgraph :math:`S` that
        does not contain any :math:`\mathbf X` nodes. But that is exactly one of the properties of C-forests
        that make up a hedge. In fact, it turns out that it is always possible to recover a hedge
        from these two c-components.
        """
        for identification in line_5_example.identifications:
            with self.assertRaises(Fail):
                line_5(identification["id_in"][0])
            with self.assertRaises(Fail):
                identify(identification["id_in"][0])

    def test_line_6(self):
        r"""Test line 6 of the identification algorithm.

        6. Asserts that if there are no bidirected arcs from X to the other nodes in the current
        subproblem under consideration, then we can replace acting on X by conditioning, and thus
        solve the subproblem.
        """
        for identification in line_6_example.identifications:
            id_out = identification["id_out"][0]

            self.assert_expr_equal(
                expected=id_out.estimand,
                actual=line_6(identification["id_in"][0]),
            )
            self.assert_expr_equal(
                P(Y | (X, Z)),
                line_6(identification["id_in"][0]),
            )

    def test_line_7(self):
        r"""Test line 2 of the identification algorithm.

        7. The most complex case where :math:`\mathbf X` is partitioned into two sets, :math:`\mathbf W` which
        contain bidirected arcs into other nodes in the subproblem, and :math:`\mathbf Z` which do not.
        In this situation, identifying :math:`P(\mathbf y|do(\mathbf x))` from :math:`P(v)` is equivalent to
        identifying :math:`P(\mathbf y|do(\mathbf w))` from :math:`P(\mathbf V|do(\mathbf z))`, since
        :math:`P(\mathbf y|do(\mathbf x)) = P(\mathbf y|do(\mathbf w), do(\mathbf z))`. But the term
        :math:`P(\mathbf V|do(\mathbf z))` is identifiable using the previous base case, so we can consider
        the subproblem of identifying :math:`P(\mathbf y|do(\mathbf w))`.
        """
        for identification in line_7_example.identifications:
            id_out = identification["id_out"][0]

            self.assertEqual(
                id_out,
                line_7(identification["id_in"][0]),
            )
            self.assert_expr_equal(
                Sum(P(Y1)),
                identify(identification["id_in"][0]),
            )
