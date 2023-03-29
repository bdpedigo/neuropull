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
        if self._unipartite:
            return self.source_nodes
        else:
            msg = (
                "`nodes` attributed is only available for verified unipartite "
                "networks. Use `source_nodes` or `target_nodes` instead, or try "
                "`.lock_unipartite()` if the network has the same source and target "
                "nodes."
            )
            raise ValueError(msg)

    def lock_unipartite(self):
        if self.source_nodes.equals(self.target_nodes):
            self._unipartite = True
        else:
            msg = (
                "Can only set unipartite if network has the same source node and "
                "target node metadata."
            )
            raise ValueError(msg)
