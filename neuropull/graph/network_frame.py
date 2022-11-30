"""Interfaces for representing networks with Frames."""
from .base_frame import BaseFrame


class BaseNetworkFrame(BaseFrame):
    """Base class for representing a network with associated metadata."""

    def __init__(self, network, source_nodes=None, target_nodes=None) -> None:
        super().__init__(network, source_nodes, target_nodes)

    @property
    def source_nodes(self):
        """Return the source node metadata of the frame."""
        return self.row_objects

    @property
    def target_nodes(self):
        """Return the target node metadata of the frame."""
        return self.col_objects

    @property
    def nodes(self):
        """Return the node metadata of the frame."""
        raise NotImplementedError()
        # TODO something about checking whether source/target are the same thing?
        pass
