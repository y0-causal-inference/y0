"""Utilities for y0."""

__all__ = [
    "InPaperAs",
]


class InPaperAs:
    """Annotate the text/LaTeX of a variable in the paper."""

    def __init__(self, value: str) -> None:
        """Initialize an InPaperAs object."""
        self.value = value

    def __repr__(self) -> str:
        return f'InPaperAs("{self.value}")'
