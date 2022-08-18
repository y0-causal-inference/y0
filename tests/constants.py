"""Constants for tests."""

import unittest

try:
    import ananke
except ImportError:
    ananke = None

__all__ = [
    "requires_ananke",
]

requires_ananke = unittest.skipIf(
    ananke is None, reason="need ananke to run falsification workflow"
)
