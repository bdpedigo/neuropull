import pandas as pd


class ArrayGraph:
    def __init__(self, adjacency, nodes=None):
        # TODO check that indexed the same way
        self.adjacency = adjacency
        if nodes is None:
            nodes = pd.DataFrame(index=adjacency.index)
        self.nodes = nodes

        self.index = self.nodes.index

        # TODO make sure this is robust
        self.dtype = adjacency.values.dtype

    # @property
    # def adjacency(self):
    #     return self.adjacency

    # @adjacency.setter
    # def adjacency(self, adjacency):
    #     self.adjacency = adjacency
    # # TODO check indexing
    # raise NotImplementedError()

    def _reindex_adjacency(self, index):
        adjacency = self.adjacency.reindex(columns=index, index=index, fill_value=0)
        return adjacency

    def reindex(self, index):
        nodes = self.nodes
        nodes = nodes.reindex(index)
        adjacency = self._reindex_adjacency(index)
        return ArrayGraph(adjacency, nodes)

    def reindex_like(self, index):
        raise NotImplementedError()

    def intersection(self, other):
        raise NotImplementedError()

    def union(self, other):
        self_index = self.index
        other_index = other.index
        union_index = self_index.union(other_index)
        return self.reindex(union_index)

    def query(self, query):
        raise NotImplementedError()

    def largest_connected_component(self):
        raise NotImplementedError()

    def __add__(self, other):
        raise NotImplementedError()

    def __sub__(self, other):
        raise NotImplementedError()

    def __len__(self):
        return len(self.index)
