# -*- coding: utf-8 -*-

"""Tests for estimating parameters of a linear SCM."""

import unittest

import pandas as pd

from y0.algorithm.estimation.linear_scm import get_single_door
from y0.dsl import Z1, Z2, X, Y, Z
from y0.examples import backdoor_example, frontdoor_example, napkin_example


class TestLinearSCM(unittest.TestCase):
    """Tests for estimating parameters of a  linear SCM."""

    def setUp(self):
        """Set up the test case."""
        self.backdoor_df = pd.read_csv("tests/test_algorithm/backdoor_example.csv")
        self.frontdoor_df = pd.read_csv("tests/test_algorithm/frontdoor_example.csv")
        self.napkin_df = pd.read_csv("tests/test_algorithm/napkin_example.csv")

    def test_get_single_door_backdoor(self):
        """Test that the single door criterion works as expected on the backdoor example."""
        actual_backdoor_parameters = get_single_door(backdoor_example.graph, self.backdoor_df)
        expected_backdoor_parameters = {
            (Z, X): 0.4659813040231536,
            (Z, Y): 0.5331826121659358,
            (X, Y): 0.9584191797996301,
        }
        for key, actual_value in actual_backdoor_parameters.items():
            self.assertAlmostEqual(expected_backdoor_parameters[key], actual_value)

    def test_get_single_door_frontdoor(self):
        """Test that the single door criterion works as expected on the frontdoor example."""
        actual_frontdoor_parameters = get_single_door(frontdoor_example.graph, self.frontdoor_df)
        expected_frontdoor_parameters = {(X, Z): 0.8074475606195012, (Z, Y): 0.30077413265953384}
        for key, actual_value in actual_frontdoor_parameters.items():
            self.assertAlmostEqual(expected_frontdoor_parameters[key], actual_value)

    def test_get_single_door_napkin(self):
        """Test that the single door criterion works as expected on the napkin example."""
        actual_napkin_parameters = get_single_door(napkin_example.graph, self.napkin_df)
        expected_napkin_parameters = {(Z2, Z1): 0.44658822516963126, (Z1, X): 0.3832318708801637}
        for key, actual_value in actual_napkin_parameters.items():
            self.assertAlmostEqual(expected_napkin_parameters[key], actual_value)
