"""Tests for :mod:`y0`."""

import importlib.util
import unittest

__all__ = [
    "requires_pgmpy",
]

requires_pgmpy = unittest.skipUnless(importlib.util.find_spec("pgmpy"), reason="requires pgmpy")
