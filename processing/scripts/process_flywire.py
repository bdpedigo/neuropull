#%%
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from neuropull.graph import AdjacencyFrame, MultiAdjacencyFrame

data_dir = Path("neuropull/processing/raw_data/flywire/526")

#%%
# this is filtered to at least 5 synapses
edgelist = pd.read_csv(data_dir / "connections.csv.gz")
edgelist.head()
#%%
# Question: is this the soma location? Centroid?
coordinates = pd.read_csv(data_dir / "coordinates.csv.gz")
coordinates.head()
# %%
labels = pd.read_csv(data_dir / "labels.csv.gz")
labels.head()

#%%
squashed_labels = labels.groupby('root_id')['tag'].agg(
    lambda x: str(x.drop_duplicates().tolist())
)
squashed_labels

#%%
neurons = pd.read_csv(data_dir / "neurons.csv.gz")
neurons.set_index('root_id', inplace=True)
neurons.head()

#%%
neurons["flow"].unique()

#%%
neurons["side"].unique()

#%%
neuropil_synapse_table = pd.read_csv(data_dir / "neuropil_synapse_table.csv.gz")
neuropil_synapse_table.head()

# %%
print(neurons.shape)
print(labels.shape)
print(coordinates.shape)
#%%
neurons['labels'] = neurons.index.map(squashed_labels)

#%%
nodes = neurons

#%%

g = nx.from_pandas_edgelist(
    edgelist,
    source='pre_root_id',
    target='post_root_id',
    edge_attr=True,
    create_using=nx.MultiDiGraph,
    edge_key='nt_type',
)

nx.set_node_attributes(g, nodes.to_dict(orient='index'))

#%%
nodelist = list(g.nodes())

print(len(nodelist))
print(len(nodes))
print(len(np.intersect1d(nodes.index, nodelist)))
nodelist = np.intersect1d(nodes.index, nodelist)
nodes = nodes.loc[nodelist]

#%%
frames = {}
for nt_type in edgelist['nt_type'].unique():
    print(nt_type)
    nt_g = nx.subgraph_view(g, filter_edge=lambda u, v, k: k == nt_type)
    nt_g = nx.DiGraph(nt_g)
    nt_sparse_adj = nx.to_scipy_sparse_array(nt_g, nodelist=nodelist)
    nt_adj_frame = AdjacencyFrame(nt_sparse_adj, source_nodes=nodes, target_nodes=nodes)
    nt_adj_frame.lock_unipartite()
    frames[nt_type] = nt_adj_frame
#%%
multi_adj_frame = MultiAdjacencyFrame(frames, nodes=nodes)


#%%

channel_key = 'nt_type'


#%%

frames = {}
for nt_type, nt_edgelist in edgelist.groupby('nt_type'):
    print(nt_type)
    g = nx.from_pandas_edgelist(
        nt_edgelist,
        source='pre_root_id',
        target='post_root_id',
        edge_attr='syn_count',
        create_using=nx.DiGraph,
    )
    nodelist = list(g.nodes())
    sparse_adj = nx.to_scipy_sparse_array(g, nodelist=nodelist, weight='syn_count')
    nt_nodes = pd.DataFrame(index=nodelist)
    nt_nodes.index.name = 'root_id'
    nt_adj_frame = AdjacencyFrame(
        sparse_adj, source_nodes=nt_nodes, target_nodes=nt_nodes
    )
    nt_adj_frame.lock_unipartite()
    frames[nt_type] = nt_adj_frame

multi_adj_frame = MultiAdjacencyFrame.from_union(frames)

#%%
