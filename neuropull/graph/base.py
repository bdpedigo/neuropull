from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd

Index = Union[pd.Index, pd.MultiIndex, np.ndarray, list]


class BaseGraphFrame:
    def __init__(self, *args, nodes=None, name=None):
        if nodes is None:
            nodes = pd.DataFrame(index=args[0].index)
        self._nodes = nodes

        self.index = self.nodes.index

        self.name = name

    @abstractmethod
    def _reindex_network(self, index):
        pass

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        # Allow setting as long as the index matches the current one
        if nodes.index.equals(self.index):
            self._nodes = nodes
        else:
            raise ValueError("Nodes must have the same index as the current Frame.")

    def reindex(self, index):
        nodes = self.nodes
        nodes = nodes.reindex(index)
        network = self._reindex_network(index)
        return self.__class__(network, nodes)

    def reindex_like(self, other):
        other_index = other.index
        return self.reindex(other_index)

    def union(self, other):
        self_index = self.index
        other_index = other.index
        union_index = self_index.union(other_index)
        return self.reindex(union_index)

    def intersection(self, other):
        self_index = self.index
        other_index = other.index
        intersection_index = self_index.intersection(other_index)
        return self.reindex(intersection_index)

    def query(self, query):
        nodes = self.nodes
        nodes = nodes.query(query)
        index = nodes.index
        return self.reindex(index)

    def __len__(self):
        return len(self.index)
