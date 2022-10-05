from functools import reduce

import pandas as pd

from .base import BaseGraphFrame


class AdjacencyFrame(BaseGraphFrame):
    def __init__(self, adjacency, nodes=None):
        # TODO check that indexed the same way

        super().__init__(adjacency, nodes=nodes)

        self.adjacency = adjacency

        # TODO make sure this is robust
        self.dtype = adjacency.values.dtype

    def _reindex_network(self, index):
        adjacency = self.adjacency.reindex(columns=index, index=index, fill_value=0)
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
        adjacency_frames = []
        for adjacency in adjacencies:
            adjacency_frames.append(AdjacencyFrame(adjacency))
        # TODO this is not robust, probably broken
        union_frame = reduce(adjacencies[0].union, adjacencies[1:])
        return union_frame
