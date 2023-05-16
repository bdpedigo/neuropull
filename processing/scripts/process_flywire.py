# %%
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from neuropull.graph import AdjacencyFrame, MultiAdjacencyFrame

data_dir = Path("neuropull/processing/raw_data/flywire/526")

# %%
# this is filtered to at least 5 synapses
edges = pd.read_csv(data_dir / "connections.csv.gz")
edges.rename(columns={"pre_root_id": "source", "post_root_id": "target"}, inplace=True)

# FOR CONSISTENCY WITH OTHER DATASETS
# REVISIT
edges = edges.pivot_table(
    index=["source", "target"], columns=["nt_type"], values="syn_count", aggfunc="sum"
)

edges.columns.name = None
edges.rename(columns=lambda x: x.lower() + "_weight", inplace=True)
edges["weight"] = edges.sum(axis=1)
edges = edges[["weight"] + list(edges.columns[:-1])]
edges.reset_index(inplace=True)

# %%
edges

# %%
# Question: is this the soma location? Centroid?
coordinates = pd.read_csv(data_dir / "coordinates.csv.gz")
coordinates.head()
# %%
labels = pd.read_csv(data_dir / "labels.csv.gz")
labels.head()

# %%
squashed_labels = labels.groupby("root_id")["tag"].agg(
    lambda x: str(x.drop_duplicates().tolist())
)
squashed_labels

# %%
neurons = pd.read_csv(data_dir / "neurons.csv.gz")
neurons.set_index("root_id", inplace=True)
neurons.index.name = "node_id"
neurons.head()

# %%
neurons["flow"].unique()

# %%
# neurons['io'] = neurons['flow']
# neurons.drop('flow', inplace=True, axis=1)

# %%
neurons["side"].unique()
neurons["side"] = neurons["side"].map(lambda x: np.nan if x == "na" else x)
neurons["side"].unique()

# %%
# neuropil_synapse_table = pd.read_csv(data_dir / "neuropil_synapse_table.csv.gz")
# neuropil_synapse_table.head()

# %%
print(neurons.shape)
print(labels.shape)
print(coordinates.shape)
# %%
neurons["labels"] = neurons.index.map(squashed_labels)

# %%
nodes = neurons

# %%

out_path = Path("neuropull/data/flywire")

edges.to_csv(out_path / "edgelist.csv", index=False)
edges.to_csv(out_path / "edgelist.csv.gz", index=False)

nodes.to_csv(out_path / "nodes.csv")
nodes.to_csv(out_path / "nodes.csv.gz")

# %%
central_nodes = nodes[nodes["class"] == "central"]
central_edges = edges.query(
    "(source in @central_nodes.index) and (target in @central_nodes.index)"
)

out_path = Path("neuropull/data/flywire_central")

central_edges.to_csv(out_path / "edgelist.csv", index=False)
central_edges.to_csv(out_path / "edgelist.csv.gz", index=False)

central_nodes.to_csv(out_path / "nodes.csv")
central_nodes.to_csv(out_path / "nodes.csv.gz")


# %%
################
quit()
# %%

g = nx.from_pandas_edgelist(
    edges,
    source="pre_root_id",
    target="post_root_id",
    edge_attr=True,
    create_using=nx.MultiDiGraph,
    edge_key="nt_type",
)

nx.set_node_attributes(g, nodes.to_dict(orient="index"))

# %%
count = 0
for i, j, w in g.edges(data=True):
    print(w)
    count += 1
    if count > 10:
        break

# %%
nodelist = list(g.nodes())

print(len(nodelist))
print(len(nodes))
print(len(np.intersect1d(nodes.index, nodelist)))
nodelist = np.intersect1d(nodes.index, nodelist)
nodes = nodes.loc[nodelist]

# %%
frames = {}
for nt_type in edges["nt_type"].unique():
    print(nt_type)
    nt_g = nx.subgraph_view(g, filter_edge=lambda u, v, k: k == nt_type)
    nt_g = nx.DiGraph(nt_g)
    nt_sparse_adj = nx.to_scipy_sparse_array(
        nt_g, nodelist=nodelist, weight="syn_count"
    )
    print(np.max(nt_sparse_adj))
    nt_adj_frame = AdjacencyFrame(nt_sparse_adj, source_nodes=nodes, target_nodes=nodes)
    nt_adj_frame.lock_unipartite()
    print(np.max(nt_adj_frame.data))
    frames[nt_type] = nt_adj_frame
# %%
multi_adj_frame = MultiAdjacencyFrame(frames, nodes=nodes)

# %%
sum_adj_frame = multi_adj_frame.to_adjacency_frame()
print(np.max(sum_adj_frame.data))

# %%
sum_adj_frame.data.count_nonzero() / (sum_adj_frame.data.shape[0] ** 2)

# %%
sum_adj_frame = sum_adj_frame.largest_connected_component()

# %%
out_path = Path("neuropull/data/flywire/526")

# adj_df = pd.DataFrame(
#     sum_adj_frame.data, index=sum_adj_frame.index, columns=sum_adj_frame.columns
# )
g = nx.from_scipy_sparse_array(sum_adj_frame.data, create_using=nx.DiGraph)
node_map = dict(zip(range(len(sum_adj_frame.index)), sum_adj_frame.index))
nx.relabel_nodes(g, node_map, copy=False)
nx.write_weighted_edgelist(g, out_path / "edgelist.csv.gz", delimiter=",")
# %%
count = 0
for i, j, w in g.edges(data=True):
    print(w)
    count += 1
    if count > 10:
        break

# %%
sum_adj_frame.nodes.to_csv(out_path / "nodes.csv.gz")
