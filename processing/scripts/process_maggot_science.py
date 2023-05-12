#%%

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("neuropull/processing/raw_data")
OUT_PATH = Path("neuropull/data")

data_dir = DATA_PATH / 'maggot' / 'science'

graph_types = ['aa', 'ad', 'all_all', 'da', 'dd']

edgelists = []
for graph_type in graph_types:
    adj = pd.read_csv(
        data_dir / 'Supplementary-Data-S1' / f'{graph_type}_connectivity_matrix.csv',
        index_col=0,
    )
    adj.columns = adj.columns.astype(int)
    adj.index = adj.index.astype(int)
    adj = adj.sort_index(axis=0).sort_index(axis=1)
    adj = pd.DataFrame(adj.values.astype(int), index=adj.index, columns=adj.index)
    assert (adj.index == adj.columns).all()
    source_ilocs, target_ilocs = np.nonzero(adj.values)
    weights = adj.values[source_ilocs, target_ilocs]
    source_locs = adj.index[source_ilocs]
    target_locs = adj.index[target_ilocs]
    edgelist = pd.DataFrame(
        {'source': source_locs, 'target': target_locs, f'{graph_type}_weight': weights}
    )
    edgelist.set_index(['source', 'target'], inplace=True)
    edgelists.append(edgelist)
#%%
edgelist = pd.concat(edgelists, axis=1).fillna(0).astype(int)
edgelist.rename(columns={'all_all_weight': 'weight'}, inplace=True)
edgelist = edgelist.reindex(
    columns=['weight', 'aa_weight', 'ad_weight', 'da_weight', 'dd_weight']
)
edgelist.to_csv(OUT_PATH / 'maggot' / 'science' / 'edgelist.csv')

#%%
paired_node_info = pd.read_csv(data_dir / 'Supplementary_Data_S2.csv')
paired_node_info['left_id'].replace('no pair', np.nan, inplace=True)
paired_node_info['right_id'].replace('no pair', np.nan, inplace=True)
paired_node_info['pair'] = np.arange(len(paired_node_info))
paired_node_info

#%%
nodes = paired_node_info.melt(
    id_vars=['celltype', 'additional_annotations', 'level_7_cluster', 'pair'],
    value_vars=['left_id', 'right_id'],
    var_name='side',
    value_name='id',
).copy()
nodes = nodes.dropna(axis=0, subset=['id'])
nodes['id'] = nodes['id'].astype(int)
nodes['side'] = nodes['side'].map({'left_id': 'left', 'right_id': 'right'})
counts = nodes.groupby('id').size()
dup_nodes = counts[counts > 1].index
nodes.drop_duplicates(subset=['id'], inplace=True)
nodes.set_index('id', inplace=True)
nodes.loc[dup_nodes, 'side'] = 'center'
nodes.rename(columns={'celltype': 'cell_type'}, inplace=True)

nodes['has_valid_pair'] = False

pair_counts = nodes['pair'].value_counts()
valid_pairs = pair_counts[pair_counts == 2].index
has_singleton_pair = nodes[nodes['pair'].isin(valid_pairs)].index
nodes.loc[has_singleton_pair, 'has_valid_pair'] = True

#%%
inputs = pd.read_csv(
    data_dir / 'Supplementary-Data-S1' / 'inputs.csv', index_col=0, dtype=int
)
outputs = pd.read_csv(
    data_dir / 'Supplementary-Data-S1' / 'outputs.csv', index_col=0, dtype=int
)


#%%
for io in ['axon_input', 'dendrite_input']:
    nodes[io] = nodes.index.map(inputs[io]).astype("Int64")

for io in ['axon_output', 'dendrite_output']:
    nodes[io] = nodes.index.map(outputs[io]).astype("Int64")

#%%
nodes.to_csv(OUT_PATH / 'maggot' / 'science' / 'nodes.csv')
