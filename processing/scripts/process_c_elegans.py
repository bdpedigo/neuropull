#%% [markdown]
# # *C. elegans* connectomes
#%%
import datetime
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from neuropull.graph import AdjacencyFrame, MultiAdjacencyFrame

DATA_PATH = Path("neuropull/processing/raw_data")
OUT_PATH = Path("neuropull/data")

t0 = time.time()

# TODO: deal with nodes without a true pair
# TODO: at least add S/I/O/M categories


#%% [markdown]
# ## Load the raw adjacency matrices
#%%


def create_node_data(node_ids, exceptions=[]):
    node_rows = []

    for node_id in node_ids:
        is_sided = True
        if not ((node_id[-1] == "L") or (node_id[-1] == "R")):
            is_exception = False
            for exception in exceptions:
                if exception in node_id:
                    is_exception = True
            if not is_exception:
                is_sided = False

        if is_sided:
            left_pos = node_id.rfind("L")
            right_pos = node_id.rfind("R")
            is_right = bool(np.argmax((left_pos, right_pos)))
            side_indicator_loc = right_pos if is_right else left_pos
            node_pair = node_id[:side_indicator_loc] + node_id[side_indicator_loc + 1 :]
            hemisphere = "R" if is_right else "L"
        else:
            node_pair = node_id
            hemisphere = "C"

        node_rows.append(
            {"node_id": node_id, "pair": node_pair, "hemisphere": hemisphere}
        )

    nodes = pd.DataFrame(node_rows).set_index("node_id")

    return nodes


def load_adjacency(path):
    adj_df = pd.read_csv(path, index_col=0, header=0).fillna(0)
    node_ids = np.union1d(adj_df.index, adj_df.columns)
    adj_df = adj_df.reindex(index=node_ids, columns=node_ids).fillna(0)
    adj_df = pd.DataFrame(
        data=adj_df.values.astype(int), index=adj_df.index, columns=adj_df.columns
    )
    return adj_df


def from_pandas_adjacencies(adjacencies, weight_names):
    graphs = []
    for adj in adjacencies:
        graph = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)
        graphs.append(graph)

    graph = nx.DiGraph()
    for i, g in enumerate(graphs):
        graph.add_weighted_edges_from(g.edges(data="weight"), weight=weight_names[i])

    return graph


def write_edgelist(g, saveloc, weight_names):
    with open(saveloc, 'w') as f:
        # header
        lineout = 'source,target,'
        for weight_name in weight_names:
            lineout += f'{weight_name},'
        f.write(lineout[:-1] + '\n')

        # edges
        for u, v, data in g.edges(data=True):
            lineout = f'{u},{v},'
            for weight_name in weight_names:
                if weight_name in data:
                    lineout += f"{data[weight_name]},"
                else:
                    lineout += ','
            f.write(lineout[:-1] + '\n')


#%% [markdown]
# ## Filter data
# Make sure neurons are lateralized and fully connected

weight_names = ['chemical_weight', 'electrical_weight']


#%%

for sex in ["male", "herm"]:
    chem_name = f"{sex}_chem_adj.csv"

    ################
    raw_path = DATA_PATH / "c_elegans"
    chem_path = raw_path / chem_name
    elec_name = f"{sex}_elec_adj.csv"
    elec_path = raw_path / elec_name

    # load adjacency matrices
    chem_adj = load_adjacency(chem_path)
    elec_adj = load_adjacency(elec_path)

    # checking that the electrical connections are already symmetric
    assert (elec_adj.values.T == elec_adj.values.T).all()

    node_ids = chem_adj.index.union(elec_adj.index)

    # generate some node metadata programatically
    nodes = create_node_data(node_ids, exceptions=["vBWM", "dgl", "dBWM"])

    chem_graph = AdjacencyFrame(chem_adj)
    elec_graph = AdjacencyFrame(elec_adj)
    print(len(chem_graph))
    print(len(elec_graph))
    chem_graph = chem_graph.union(elec_graph)
    elec_graph = elec_graph.union(chem_graph)
    print(len(chem_graph))
    print(len(elec_graph))
    print()

    multigraph = MultiAdjacencyFrame(
        {"chemical": chem_graph, "electrical": elec_graph},
        nodes=nodes,
    )
    frames = multigraph.to_adjacency_frames()
    chem_graph = frames['chemical']
    elec_graph = frames['electrical']

    g = from_pandas_adjacencies(
        [chem_graph.adjacency, elec_graph.adjacency], weight_names
    )

    saveloc = OUT_PATH / f'c_elegans_{sex}_edgelist.csv'
    write_edgelist(g, saveloc, weight_names=weight_names)

    saveloc = OUT_PATH / f"c_elegans_{sex}_nodes.csv"
    chem_graph.nodes.to_csv(saveloc)

#%%
nodes_path = DATA_PATH / "c_elegans" / "nodes.csv"
nodes = pd.read_csv(nodes_path, index_col=0)
nodes

#%%
node_ids[~node_ids.isin(nodes.index)]

#%%
network_index = node_ids.str.lower().str.strip(" ")

should_be_herm_nodes = nodes[nodes['sex'] != 'male']
node_index = should_be_herm_nodes.index.str.lower().str.strip(' ')

missing_in_graph = node_index.difference(network_index)
missing_in_nodes = network_index.difference(node_index)

#%%
missing_in_graph.sort_values()


#%%
missing_in_nodes.sort_values()

#%%
multiframe = MultiAdjacencyFrame.from_union(
    {'chemical': chem_adj, "electrical": elec_adj}
)

#%% [markdown]
# ## End
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
