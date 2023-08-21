"""Testing constants for Y0."""

from pathlib import Path

__all__ = ["HERE", "DATA_DIRECTORY", "NAPKIN_OBSERVATIONAL_PATH"]

HERE = Path(__file__).parent.resolve()
DATA_DIRECTORY = HERE.joinpath("data")
NAPKIN_OBSERVATIONAL_PATH = DATA_DIRECTORY.joinpath("napkin", "observational.tsv")
