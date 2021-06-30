# -*- coding: utf-8 -*-

"""Constants for y0."""

from __future__ import annotations

from typing import Protocol, TypeVar

__all__ = [
    "NodeType",
    "NodeProtocol",
]


class NodeProtocol(Protocol):
    """Represents what can be a node in a mixed graph."""

    def __hash__(self) -> int:
        ...

    def __lt__(self, other) -> bool:
        ...


NodeType = TypeVar("NodeType", bound=NodeProtocol)
