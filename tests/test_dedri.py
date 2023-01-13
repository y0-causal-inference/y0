import unittest

from y0.dedri import Interval


class TestDeDri(unittest.TestCase):
    """Tests for the stringifying instances of the probability DSL."""

    def test_interval(self):
        """Test that the 13 allen interval relations work as expected"""
        a = Interval("a", 1, 10)
        b = Interval("b", 11, 20)
        c = Interval("c", 5, 15)
        d = Interval("d", 10, 11)
        e = Interval("e", 1, 20)
        f = Interval("f", 1, 10)

        self.assertTrue(a < b)
        self.assertFalse(a > b)
        self.assertTrue(b > a)
        self.assertFalse(b < a)
        self.assertTrue(a.meets(d))
        self.assertFalse(d.meets(a))
        self.assertTrue(d.is_met_by(a))
        self.assertFalse(a.is_met_by(d))
        self.assertTrue(a.overlaps_with(c))
        self.assertFalse(c.overlaps_with(a))
        self.assertTrue(c.is_overlapped_by(a))
        self.assertFalse(a.is_overlapped_by(c))
        self.assertTrue(a.starts(e))
        self.assertFalse(e.starts(a))
        self.assertTrue(e.is_started_by(a))
        self.assertFalse(a.is_started_by(e))
        self.assertTrue(c.during(e))
        self.assertFalse(e.during(c))
        self.assertTrue(e.contains(c))
        self.assertFalse(c.contains(e))
        self.assertTrue(b.finishes(e))
        self.assertFalse(e.finishes(b))
        self.assertTrue(e.is_finished_by(b))
        self.assertFalse(b.is_finished_by(e))
        self.assertTrue(a == f)
        self.assertFalse(a == b)
        self.assertFalse(a == c)
        self.assertFalse(a == d)
        self.assertFalse(a == e)

    def test_to_json(self):
        """Tests for serialization to JSON abstract syntax tree"""
        
