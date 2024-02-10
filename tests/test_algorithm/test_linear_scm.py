# -*- coding: utf-8 -*-

"""Tests for estimating parameters of a linear SCM."""

import unittest

import pandas as pd

from y0.algorithm.estimation.linear_scm import get_single_door
from y0.dsl import X, Y, Z
from y0.examples import backdoor_example, frontdoor_example
from y0.graph import NxMixedGraph


class TestLinearSCM(unittest.TestCase):
    """Tests for estimating parameters of a  linear SCM."""

    def setUp(self):
        """Set up the test case."""
        self.backdoor_example = backdoor_example
        self.backdoor_df = pd.read_csv("tests/test_algorithm/backdoor_example.csv")

    def test_get_single_door(self):
        """Test that the single door criterion works as expected."""
        actual_backdoor_parameters = get_single_door(self.backdoor_example.graph, self.backdoor_df)
        expected_backdoor_parameters = {
            (Z, X): 0.4659813040231536,
            (Z, Y): 0.5331826121659358,
            (X, Y): 0.9584191797996301,
        }
        for key, actual_value in actual_backdoor_parameters.items():
            self.assertAlmostEqual(expected_backdoor_parameters[key], actual_value)
