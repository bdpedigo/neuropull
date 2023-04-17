"""Module for representing and manipulating data objects with associated metadata."""

from .adjacency import AdjacencyFrame, MultiAdjacencyFrame
from .network_frame import NetworkFrame

__all__ = ["AdjacencyFrame", "MultiAdjacencyFrame", "NetworkFrame"]
