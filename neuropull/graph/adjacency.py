"""Classes for representing networks as adjacency matrices with node metadata."""
import numpy as np
import pandas as pd
from scipy.sparse import csgraph, csr_array

from .matrix import DenseMatrix, SparseMatrix
from .network_frame_old import BaseNetworkFrame


def _largest_connected_component(adjacency):
    n_components, labels = csgraph.connected_components(
        adjacency, directed=True, connection="weak", return_labels=True
    )
    if n_components > 1:
        unique_labels, counts = np.unique(labels, return_counts=True)
        lcc_label_ind = np.argmax(counts)  # LCC is the component with the most nodes,
        # so it is the component label with the highest count in the label array

        lcc_label = unique_labels[lcc_label_ind]  # grab the component label for the LCC

        lcc_mask = labels == lcc_label  # create a boolean mask array for where the
        # component labels equal that of the largest connected component

        lcc = adjacency[lcc_mask][:, lcc_mask]  # mask the adjacency matrix to only LCC

        all_inds = np.arange(adjacency.shape[0])
        lcc_inds = all_inds[lcc_mask]
        return lcc, lcc_inds
    else:
        return adjacency, np.arange(adjacency.shape[0])


class AdjacencyFrame(BaseNetworkFrame):
    """Represent a network as an adjacency matrix with associated metadata.

    Parameters
    ----------
    BaseNetworkFrame : _type_
        _description_
    """

    def __init__(
        self,
        adjacency,
        nodes=None,
        source_nodes=None,
        target_nodes=None,
    ) -> None:
        if nodes is not None:
            source_nodes = nodes
            target_nodes = nodes

        if isinstance(adjacency, np.ndarray):
            adjacency = pd.DataFrame(adjacency)

        if isinstance(adjacency, pd.DataFrame):
            adjacency = DenseMatrix(adjacency)
        elif isinstance(adjacency, csr_array):
            if source_nodes is not None:
                source_index = source_nodes.index
            else:
                source_index = np.arange(adjacency.shape[0])
            if target_nodes is not None:
                target_index = target_nodes.index
            else:
                target_index = np.arange(adjacency.shape[1])
            adjacency = SparseMatrix(adjacency, source_index, target_index)

        if source_nodes is None:
            source_nodes = pd.DataFrame(index=adjacency.index)
        if target_nodes is None:
            target_nodes = pd.DataFrame(index=adjacency.columns)

        if source_nodes.equals(target_nodes):
            self._unipartite = True

        super().__init__(adjacency, source_nodes, target_nodes)

    @property
    def shape(self):
        """Return the shape of the matrix."""
        return self._data.shape

    def __repr__(self) -> str:
        """Return a string representation of the frame."""
        out = f"AdjacencyFrame with shape: {self.shape}\n"
        out += f"Source node features: {self.source_nodes.shape[1]}\n"
        out += f"Target node features: {self.target_nodes.shape[1]}\n"
        return out

    def largest_connected_component(self):
        """Return a new frame with only the largest connected component."""
        # TODO hacky, this should use reindex
        # but need to fix the sparse version first
        lcc, lcc_inds = _largest_connected_component(self.data)
        source_nodes = self.source_nodes.iloc[lcc_inds]
        target_nodes = self.target_nodes.iloc[lcc_inds]
        return AdjacencyFrame(lcc, source_nodes=source_nodes, target_nodes=target_nodes)


# class OldAdjacencyFrame(BaseGraphFrame):
#     def __init__(self, adjacency, nodes=None, name=None):
#         # TODO check that indexed the same way

#         super().__init__(adjacency, nodes=nodes, name=name)

#         self._adjacency = MatrixFrame(adjacency)

#         # TODO make sure this is robust
#         self.dtype = adjacency.values.dtype

#     @property
#     def adjacency(self):
#         return self._adjacency.adjacency

#     def _reindex_network(self, index):
#         adjacency = self._adjacency.reindex(columns=index, index=index, fill_value=0)
#         return adjacency

#     def largest_connected_component(self):
#         raise NotImplementedError()

#     def __add__(self, other):
#         raise NotImplementedError()

#     def __len__(self):
#         return len(self.index)


class MultiAdjacencyFrame:
    """Multiplex network as a collection of adjacencies with associated metadata."""

    def __init__(self, adjacencies, nodes=None):

        if isinstance(adjacencies, list):
            new_adjacencies = {}
            for i, adjacency in enumerate(adjacencies):
                new_adjacencies[i] = adjacency
            adjacencies = new_adjacencies

        layer_names = list(adjacencies.keys())

        index = adjacencies[layer_names[0]].index
        for graph in adjacencies.values():
            if not graph.index.equals(index):
                raise ValueError("Adjacency matrices must have the same index.")

        self.adjacencies = adjacencies

        if nodes is None:
            nodes = pd.DataFrame(index=adjacencies[layer_names[0]].index)

        self.nodes = nodes
        self.index = self.nodes.index

    @property
    def layer_names(self):
        """Return the names of the layers in the multiplex network."""
        return self.adjacencies.keys()

    @layer_names.setter
    def layer_names(self, layer_name_map):
        """Rename the layers in the multiplex network."""
        new_adjacencies = {}
        for layer_name in self.layer_names:
            new_adjacencies[layer_name_map[layer_name]] = self.adjacencies[layer_name]
        self.adjacencies = new_adjacencies

    @classmethod
    def from_union(cls, adjacencies):
        """Create a multiplex network from a collection of adjacency matrices."""
        if isinstance(adjacencies, dict):
            names = list(adjacencies.keys())
            adjacencies = list(adjacencies.values())
        else:
            names = [str(i) for i in range(len(adjacencies))]

        if isinstance(adjacencies[0], pd.DataFrame):
            adjacency_frames = []
            for adjacency in adjacencies:
                adjacency_frames.append(AdjacencyFrame(adjacency))
        elif isinstance(adjacencies[0], AdjacencyFrame):
            adjacency_frames = adjacencies
        else:
            raise TypeError(
                "Adjacencies must be a list of DataFrames or AdjacencyFrames."
            )

        all_nodes = [frame.nodes for frame in adjacency_frames]
        nodes = _concat_check_duplicates(all_nodes)
        union_index = nodes.index

        new_adjacencies = {}
        for name, frame in zip(names, adjacency_frames):
            new_adjacencies[name] = frame.reindex(union_index).adjacency

        union_frame = cls(new_adjacencies, nodes=nodes)

        return union_frame

    def to_adjacency_frames(self):
        """Return a list of AdjacencyFrames from the layers of the multiplex network."""
        frames = {}
        for name, adjacency in self.adjacencies.items():
            frames[name] = AdjacencyFrame(adjacency, nodes=self.nodes)
        return frames

    def to_adjacency_frame(self, agg=None):
        """Return a single AdjacencyFrame from the layers of the multiplex network."""
        if agg is None:
            agg = np.sum

        adjacencies = [frame.data for frame in self.to_adjacency_frames().values()]
        adjacency = agg(adjacencies, axis=0)

        return AdjacencyFrame(
            adjacency, source_nodes=self.nodes, target_nodes=self.nodes
        )


def _concat_check_duplicates(dataframes):
    data = pd.concat(dataframes, axis=0, join="outer")

    duplicate_data = data[data.index.duplicated(keep=False)]

    for index_id, index_data in duplicate_data.groupby(level=0):
        if not index_data.duplicated(keep=False).all():
            msg = (
                f"Index value {index_id} (and possibly others) is duplicated in node"
                " metadata, but has different values."
            )
            raise ValueError(msg)
    # If we get to this point, then there are no duplicate index values with differing
    # data. We can safely drop the duplicates and keep the first row for each.
    data = data[~data.index.duplicated(keep="first")]
    return data
