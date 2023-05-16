# %% [markdown]
# # *C. elegans* connectomes
# %%
import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("neuropull/processing/raw_data")
OUT_PATH = Path("neuropull/data")

t0 = time.time()

# TODO: deal with nodes without a true pair
# TODO: at least add S/I/O/M categories


# %% [markdown]
# ## Load the raw adjacency matrices
# %%


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
            side = "right" if is_right else "left"
        else:
            node_pair = node_id
            side = "center"

        node_rows.append({"node_id": node_id, "pair": node_pair, "side": side})

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


# def write_edgelist(g, saveloc, weight_names):
#     with open(saveloc, "w") as f:
#         # header
#         lineout = "source,target,"
#         for weight_name in weight_names:
#             lineout += f"{weight_name},"
#         f.write(lineout[:-1] + "\n")

#         # edges
#         for u, v, data in g.edges(data=True):
#             lineout = f"{u},{v},"
#             for weight_name in weight_names:
#                 if weight_name in data:
#                     lineout += f"{data[weight_name]},"
#                 else:
#                     lineout += ","
#             f.write(lineout[:-1] + "\n")


# %% [markdown]
# ## Filter data
# Make sure neurons are lateralized and fully connected

weight_names = ["chemical_weight", "electrical_weight"]


def to_edgelist(adjacency, weight_name="weight", directed=True):
    adj_values = adjacency.values.copy()
    if not directed:
        indices = np.triu_indices_from(adj_values, k=1)
        adj_values[indices] = 0

    source_ilocs, target_ilocs = np.nonzero(adj_values)
    weights = adjacency.values[source_ilocs, target_ilocs]
    source_locs = adjacency.index[source_ilocs]
    target_locs = adjacency.index[target_ilocs]
    edgelist = pd.DataFrame(
        {"source": source_locs, "target": target_locs, weight_name: weights}
    )
    return edgelist


# %%

for sex in ["male", "hermaphrodite"]:
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

    # convert to edge tables
    chem_edgelist = to_edgelist(chem_adj, weight_name="chemical_weight").set_index(
        ["source", "target"]
    )
    elec_edgelist = to_edgelist(elec_adj, weight_name="electrical_weight").set_index(
        ["source", "target"]
    )
    edgelist = (
        chem_edgelist.join(elec_edgelist, how="outer").astype("Int64").reset_index()
    )
    edgelist["weight"] = edgelist["chemical_weight"].fillna(0) + edgelist[
        "electrical_weight"
    ].fillna(0)
    edgelist = edgelist[
        ["source", "target", "weight", "chemical_weight", "electrical_weight"]
    ]

    node_ids = chem_adj.index.union(elec_adj.index)

    # generate some node metadata programatically
    nodes = create_node_data(node_ids, exceptions=["vBWM", "dgl", "dBWM"])

    nodes.index.name = "node_id"

    saveloc = OUT_PATH / f"c_elegans_{sex}"

    edgelist.to_csv(saveloc / "edgelist.csv", index=False)
    edgelist.to_csv(saveloc / "edgelist.csv.gz", index=False)
    nodes.to_csv(saveloc / "nodes.csv")
    nodes.to_csv(saveloc / "nodes.csv.gz")

# %% [markdown]
# ## End
# %%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
