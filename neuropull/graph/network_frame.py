#%%
import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from typing import Optional, Union

import numpy as np
import pandas as pd
from beartype import beartype
from scipy.sparse import csr_array
from graspologic.utils import largest_connected_component


AxisType = Union[
    Literal[0], Literal[1], Literal["index"], Literal["columns"], Literal['both']
]

EdgeAxisType = Union[Literal['source'], Literal['target'], Literal['both']]

ColumnsType = Union[list, str]

NetworkFrameReturn = Union["NetworkFrame", None]


class NetworkFrame:
    """Represent a network as a pair of dataframes, one for nodes and one for edges.

    Parameters
    ----------
    nodes : pd.DataFrame
        Table of node attributes, the node IDs are assumed to be the index
    edges : pd.DataFrame
        Table of edges, with source and target columns which correspond with the node
        Ids in the nodes dataframe.
    directed : bool, optional
        Whether the network should be treated as directed, by default True
    sources : pd.Index, optional
        Specification of source nodes if representing a subgraph, by default None
    targets : pd.Index, optional
        Specification of target nodes if representing a subgraph, by default None
    """

    @beartype
    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        directed: bool = True,
        sources: Optional[pd.Index] = None,
        targets: Optional[pd.Index] = None,
    ):

        # TODO checks ensuring that nodes and edges are valid.

        if not nodes.index.is_unique:
            raise ValueError("Node IDs must be unique.")

        referenced_node_ids = np.union1d(
            edges['source'].unique(), edges['target'].unique()
        )
        if not np.all(np.isin(referenced_node_ids, nodes.index)):
            raise ValueError("All nodes referenced in edges must be in nodes.")

        # should probably assume things like "source" and "target" columns
        # and that these elements are in the nodes dataframe
        # TODO are multigraphs allowed?
        # TODO assert that sources and targets and node index are all unique?
        self.nodes = nodes
        self.edges = edges
        if sources is None and targets is None:
            self.induced = True
            self._sources = None
            self._targets = None
        else:
            self.induced = False
            self._sources = sources
            self._targets = targets
        # TODO some checks on repeated edges if not directed
        self.directed = directed

    @property
    def sources(self):
        """Return the source node IDs of the network."""
        if self.induced:
            return self.nodes.index
        else:
            return self.nodes.index.intersection(self._sources, sort=False)
            # all_sources = self.edges["source"].unique()
            # # TODO verify that this retains the order
            # return self.nodes.index.intersection(all_sources, sort=False)

    @property
    def targets(self):
        """Return the target node IDs of the network."""
        if self.induced:
            return self.nodes.index
        else:
            return self.nodes.index.intersection(self._targets, sort=False)
            # all_targets = self.edges["target"].unique()
            # # TODO verify that this retains the order
            # return self.nodes.index.intersection(all_targets, sort=False)

    @property
    def source_nodes(self):
        """Return the source nodes of the network and their metadata."""
        return self.nodes.loc[self.sources]

    @property
    def target_nodes(self):
        """Return the target nodes of the network and their metadata."""
        return self.nodes.loc[self.targets]

    def __repr__(self) -> str:
        """Return a string representation of the NetworkFrame."""
        out = f"NetworkFrame(nodes={self.nodes.shape}, edges={self.edges.shape}, "
        out += f"induced={self.induced}, directed={self.directed})"
        return out

    def reindex_nodes(self, index: pd.Index) -> "NetworkFrame":
        """
        Reindex the nodes dataframe to a new index. Also removes edges as necessary.

        """
        nodes = self.nodes.reindex(index=index, axis=0)
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")
        return NetworkFrame(nodes, edges, directed=self.directed)

    # def reindex_edges(self, index: pd.Index, axis: AxisType = 0) -> "NetworkFrame":
    #     if axis != 'both':
    #         self.edges = self.edges.reindex(index=index, axis=axis)
    #     else:
    #         self.edges = self.edges.reindex(index=index, axis=0)
    #         self.edges = self.edges.reindex(index=index, axis=1)
    #     return self

    # def reindex_like(self, other: "NetworkFrame") -> "NetworkFrame":
    #     self.reindex_nodes(other.nodes.index, axis='both')
    #     self.reindex_edges(other.edges.index, axis='both')
    #     return self

    def query_nodes(self, query: str, inplace=False) -> Optional["NetworkFrame"]:
        """Query the nodes dataframe and remove edges that are no longer valid."""
        nodes = self.nodes.query(query)
        # get the edges that are connected to the nodes that are left after the query
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")
        if inplace:
            self.nodes = nodes
            self.edges = edges
            return None
        else:
            return NetworkFrame(nodes, edges, directed=self.directed)

    def query_edges(self, query: str, inplace=False) -> Optional["NetworkFrame"]:
        """Query the edges dataframe."""
        edges = self.edges.query(query)
        if inplace:
            self.edges = edges
            return None
        else:
            return NetworkFrame(self.nodes, edges, directed=self.directed)

    def remove_unused_nodes(self, inplace=False) -> Optional["NetworkFrame"]:
        """Remove nodes that are not connected to any edges."""
        index = self.nodes.index
        new_index = index.intersection(
            self.edges.source.append(self.edges.target).unique()
        )
        nodes = self.nodes.loc[new_index]
        if inplace:
            self.nodes = nodes
            return None
        else:
            return NetworkFrame(nodes, self.edges, directed=self.directed)

    def apply_node_features(
        self, columns: ColumnsType, axis: EdgeAxisType = 'both', inplace=False
    ) -> Optional["NetworkFrame"]:
        """Apply node features to the edges dataframe."""
        if not inplace:
            edges = self.edges.copy()
        else:
            edges = self.edges
        if isinstance(columns, str):
            columns = [columns]
        if axis in ['source', 'both']:
            for col in columns:
                edges[f'source_{col}'] = self.edges['source'].map(self.nodes[col])
        if axis in ['target', 'both']:
            for col in columns:
                edges[f'target_{col}'] = self.edges['target'].map(self.nodes[col])
        if inplace:
            self.edges = edges
            return None
        else:
            return NetworkFrame(self.nodes, edges, directed=self.directed)

    def to_adjacency(self, weight_col: str = 'weight', aggfunc='sum') -> pd.DataFrame:
        """Return the adjacency matrix of the network."""
        # TODO: wondering if the sparse method of doing this would actually be faster
        # here too...
        adj_df = self.edges.pivot_table(
            index='source',
            columns='target',
            values=weight_col,
            fill_value=0,
            aggfunc=aggfunc,
            sort=False,
        )
        adj_df = adj_df.reindex(
            index=self.sources,
            columns=self.targets,
            fill_value=0,
        )
        adj_df.index = adj_df.index.set_names('source')
        adj_df.columns = adj_df.columns.set_names('target')
        return adj_df

    def to_networkx(self):
        """Return a networkx graph of the network."""
        import networkx as nx

        if self.directed:
            create_using = nx.MultiDiGraph
        else:
            create_using = nx.MultiGraph

        g = nx.from_pandas_edgelist(
            self.edges,
            source='source',
            target='target',
            edge_attr=True,
            create_using=create_using,
        )
        nx.set_node_attributes(g, self.nodes.to_dict(orient='index'))
        return g

    def to_sparse_adjacency(
        self, weight_col: str = 'weight', aggfunc='sum'
    ) -> csr_array:
        """Return the adjacency matrix of the network as a sparse array."""
        edges = self.edges
        # TODO only necessary since there might be duplicate edges
        # might be more efficient to have a attributed checking this, e.g. set whether
        # this is a multigraph or not
        effective_edges = edges.groupby(['source', 'target'])[weight_col].agg(aggfunc)

        data = effective_edges.values
        source_indices = effective_edges.index.get_level_values('source')
        target_indices = effective_edges.index.get_level_values('target')

        source_indices = pd.Categorical(source_indices, categories=self.nodes.index)
        target_indices = pd.Categorical(target_indices, categories=self.nodes.index)

        adj = csr_array(
            (data, (source_indices.codes, target_indices.codes)),
            shape=(len(self.sources), len(self.targets)),
        )
        return adj

    def largest_connected_component(self, inplace=False, verbose=False):
        """Return the largest connected component of the network."""
        adjacency = self.to_sparse_adjacency()
        _, indices = largest_connected_component(adjacency, return_inds=True)
        if verbose:
            n_removed = len(self.nodes) - len(indices)
            print(f"Nodes removed when taking largest connected component: {n_removed}")
        nodes = self.nodes.iloc[indices]
        edges = self.edges.query("(source in @nodes.index) & (target in @nodes.index)")
        if inplace:
            self.nodes = nodes
            self.edges = edges
            return None
        else:
            return NetworkFrame(nodes, edges, directed=self.directed)

    def groupby_nodes(self, by=None, axis='both', **kwargs):
        """Group the frame by node data for the source or target (or both).

        Parameters
        ----------
        by : _type_, optional
            _description_, by default None
        axis : str, optional
            _description_, by default 'both'

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if axis == 0:
            source_nodes_groupby = self.source_nodes.groupby(by=by, **kwargs)
        elif axis == 1:
            target_nodes_groupby = self.target_nodes.groupby(by=by, **kwargs)
        elif axis == 'both':
            source_nodes_groupby = self.source_nodes.groupby(by=by, **kwargs)
            target_nodes_groupby = self.target_nodes.groupby(by=by, **kwargs)
        else:
            raise ValueError("Axis must be 0 or 1 or 'both'")

        return NodeGroupBy(self, source_nodes_groupby, target_nodes_groupby)

    @property
    def loc(self):
        """Return a LocIndexer for the frame."""
        return LocIndexer(self)


class NodeGroupBy:
    """A class for grouping a NetworkFrame by a set of labels."""

    def __init__(self, frame, source_groupby, target_groupby):
        self._frame = frame
        self._source_groupby = source_groupby
        self._target_groupby = target_groupby

        if source_groupby is None:
            self._axis = 1
        elif target_groupby is None:
            self._axis = 0
        else:
            self._axis = 'both'

        if self.has_source_groups:
            self.source_group_names = list(source_groupby.groups.keys())
        if self.has_target_groups:
            self.target_group_names = list(target_groupby.groups.keys())

    @property
    def has_source_groups(self):
        """Whether the frame has row groups."""
        return self._source_groupby is not None

    @property
    def has_target_groups(self):
        """Whether the frame has column groups."""
        return self._target_groupby is not None

    def __iter__(self):
        """Iterate over the groups."""
        if self._axis == 'both':
            for source_group, source_objects in self._source_groupby:
                for target_group, target_objects in self._target_groupby:
                    yield (source_group, target_group), self._frame.loc[
                        source_objects.index, target_objects.index
                    ]
        elif self._axis == 0:
            for source_group, source_objects in self._source_groupby:
                yield source_group, self._frame.loc[source_objects.index]
        elif self._axis == 1:
            for target_group, target_objects in self._target_groupby:
                yield target_group, self._frame.loc[:, target_objects.index]

    # def apply(self, func, *args, data=False, **kwargs):
    #     """Apply a function to each group."""
    #     if self._axis == 'both':
    #         answer = pd.DataFrame(
    #             index=self.source_group_names, columns=self.target_group_names
    #         )

    #     else:
    #         if self._axis == 0:
    #             answer = pd.Series(index=self.source_group_names)
    #         else:
    #             answer = pd.Series(index=self.target_group_names)
    #     for group, frame in self:
    #         if data:
    #             value = func(frame.data, *args, **kwargs)
    #         else:
    #             value = func(frame, *args, **kwargs)
    #         answer.at[group] = value
    #     return answer

    @property
    def source_groups(self):
        """Return the row groups."""
        if self._axis == 'both' or self._axis == 0:
            return self._source_groupby.groups
        else:
            raise ValueError('No source groups, groupby was on targets only')

    @property
    def target_groups(self):
        """Return the column groups."""
        if self._axis == 'both' or self._axis == 1:
            return self._target_groupby.groups
        else:
            raise ValueError('No target groups, groupby was on sources only')


class LocIndexer:
    """A class for indexing a NetworkFrame using .loc."""

    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, args):
        """Return a NetworkFrame with the given indices."""
        if isinstance(args, tuple):
            if len(args) != 2:
                raise ValueError("Must provide at most two indexes.")
            else:
                row_index, col_index = args
        else:
            raise NotImplementedError()

        if isinstance(row_index, int):
            row_index = [row_index]
        if isinstance(col_index, int):
            col_index = [col_index]

        if isinstance(row_index, slice):
            row_index = self._frame.nodes.index[row_index]
        if isinstance(col_index, slice):
            col_index = self._frame.nodes.index[col_index]

        row_index = pd.Index(row_index)
        col_index = pd.Index(col_index)

        source_nodes = self._frame.nodes.loc[row_index]
        target_nodes = self._frame.nodes.loc[col_index]

        edges = self._frame.edges.query(
            "source in @source_nodes.index and target in @target_nodes.index"
        )

        if row_index.equals(col_index):
            nodes = source_nodes
            return NetworkFrame(
                nodes,
                edges,
                directed=self._frame.directed,
            )
        else:
            nodes = pd.concat([source_nodes, target_nodes], copy=False, sort=False)
            nodes = nodes.loc[~nodes.index.duplicated(keep='first')]
            return NetworkFrame(
                nodes,
                edges,
                directed=self._frame.directed,
                sources=row_index,
                targets=col_index,
            )
