from tests.test_algorithm import cases
from y0.algorithm.identify import identify_outcomes, identify
from y0.algorithm.tian_id import identify_district_variables
from y0.algorithm.identify.utils import Identification, Query, Unidentifiable
from y0.dsl import Z1, Z2, Fraction, P, Sum, X, Y
from y0.examples import napkin


class TestNapkinIdentification(cases.GraphTestCase):
    """Unit test to verify the fix for the napkin graph when calling identify_outcomes."""

    def test_napkin_identify_outcomes_returns_correct_estimand(self) -> None:
        """Testing that identify_outcomes returns correct estimand formula for the napkin graph."""
        # # Check what variables napkin actually has
        # print("Napkin graph nodes:", list(napkin.nodes()))
        # print("Napkin graph directed edges:", list(napkin.directed.edges()))
        # print("Napkin graph undirected edges:", list(napkin.undirected.edges()))

        result = identify_outcomes(napkin, treatments={X}, outcomes={Y})

        numerator = Sum[Z1](P(Y, X | Z1, Z2) * P(Z2))
        denominator = Sum[Z1](P(X | Z1, Z2) * P(Z2))
        expected = Fraction(numerator, denominator)

        # main check
        self.assertIsInstance(result, Fraction)

        # verify mathematical equivalence
        self.assert_expr_equal(expected, result)

    def test_napkin_id_std_returns_correct_estimand(self) -> None:
        """Testing that id_std returns correct estimand formula for the napkin graph."""
        # # Check what variables napkin actually has
        # print("Napkin graph nodes:", list(napkin.nodes()))
        # print("Napkin graph directed edges:", list(napkin.directed.edges()))
        # print("Napkin graph undirected edges:", list(napkin.undirected.edges()))

        identification = Identification.from_parts(outcomes={Y}, treatments={X}, graph=napkin)

        result = identify(identification)

        numerator = Sum[Z1](P(Y, X | Z1, Z2) * P(Z2))
        denominator = Sum[Z1](P(X | Z1, Z2) * P(Z2))
        expected = Fraction(numerator, denominator)

        print("Identified estimand:", result)

        # main check
        self.assertIsInstance(result, Fraction)

        # verify mathematical equivalence
        self.assert_expr_equal(expected, result)


def test_napkin_tian_identify_returns_correct_estimand(self) -> None:
        """Testing that id_std returns correct estimand formula for the napkin graph."""
        # # Check what variables napkin actually has
        # print("Napkin graph nodes:", list(napkin.nodes()))
        # print("Napkin graph directed edges:", list(napkin.directed.edges()))
        # print("Napkin graph undirected edges:", list(napkin.undirected.edges()))

        C = {Y}
        T = {X, Y, Z2}
        district_probability = P( Y @ Z1)
        result = identify_district_variables(
        numerator = Sum[Z1](P(Y, X | Z1, Z2) * P(Z2))
        denominator = Sum[Z1](P(X | Z1, Z2) * P(Z2))
        expected = Fraction(numerator, denominator)

        print("Identified estimand:", result)

        # main check
        self.assertIsInstance(result, Fraction)

        # verify mathematical equivalence
        self.assert_expr_equal(expected, result)

def test_napkin_counterfactual_transportability(self) -> None:
    """Testing that counterfactual transportability generates correct estimand for the napkin graph."""

    