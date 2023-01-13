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
sum_adj_frame = multi_adj_frame.to_adjacency_frame()

#%%
sum_adj_frame.data.count_nonzero() / (sum_adj_frame.data.shape[0] ** 2)

#%%
sum_adj_frame = sum_adj_frame.largest_connected_component()

#%%
out_path = Path("neuropull/data/flywire/526")

# adj_df = pd.DataFrame(
#     sum_adj_frame.data, index=sum_adj_frame.index, columns=sum_adj_frame.columns
# )
g = nx.from_scipy_sparse_array(sum_adj_frame.data, create_using=nx.DiGraph)
node_map = dict(zip(range(len(sum_adj_frame.index)), sum_adj_frame.index))
nx.relabel_nodes(g, node_map, copy=False)
nx.write_weighted_edgelist(g, out_path / "edgelist.csv.gz")

#%%
sum_adj_frame.nodes.to_csv(out_path / "nodes.csv.gz")

# #%%

# lse = LaplacianSpectralEmbed(n_components=4, form='R-DAD', concat=True)
# embedding = lse.fit_transform(sum_adj_frame.data)

# #%%

# rng = np.random.default_rng(8888)
# inds = rng.choice(embedding.shape[0], size=5000, replace=False)
# X = embedding[inds]
# y = sum_adj_frame.nodes.iloc[inds]['class'].fillna('unk').values

# pairplot(X, labels=y)
