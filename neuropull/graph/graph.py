import pandas as pd

from .base import BaseGraphFrame, _PandasSubgraphAdjacency


class AdjacencyFrame(BaseGraphFrame):
    def __init__(self, adjacency, nodes=None, name=None):
        # TODO check that indexed the same way

        super().__init__(adjacency, nodes=nodes, name=name)

        self._adjacency = _PandasSubgraphAdjacency(adjacency)

        # TODO make sure this is robust
        self.dtype = adjacency.values.dtype

    @property
    def adjacency(self):
        return self._adjacency.adjacency

    def _reindex_network(self, index):
        adjacency = self._adjacency.reindex(columns=index, index=index, fill_value=0)
        return adjacency

    def largest_connected_component(self):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()

    def __len__(self):
        return len(self.index)


class MultiAdjacencyFrame:
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
        return self.adjacencies.keys()

    @layer_names.setter
    def layer_names(self, layer_name_map):
        new_adjacencies = {}
        for layer_name in self.layer_names:
            new_adjacencies[layer_name_map[layer_name]] = self.adjacencies[layer_name]
        self.adjacencies = new_adjacencies

    @classmethod
    def from_union(cls, adjacencies):

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
        frames = {}
        for name, adjacency in self.adjacencies.items():
            print(type(adjacency))
            frames[name] = AdjacencyFrame(adjacency, nodes=self.nodes)
        return frames


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
