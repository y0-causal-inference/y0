from pathlib import Path

__all__ = ["HERE", "DATA_DIRECTORY", "NAPKIN_TEST_PATH"]

HERE = Path(__file__).parent.resolve()
DATA_DIRECTORY = HERE.joinpath("data")
NAPKIN_TEST_PATH = DATA_DIRECTORY.joinpath("napkin_test.tsv")
