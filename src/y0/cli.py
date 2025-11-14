"""Command line interface for :mod:`y0`."""

import click

__all__ = [
    "main",
]


@click.group()
@click.version_option()
def main() -> None:
    """CLI for y0."""


if __name__ == "__main__":
    main()
