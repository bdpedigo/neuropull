import requests
import json
import pandas as pd
import networkx as nx

base_url = "http://nemanode.org/api/dataset-json?datasetId={}"

datasets = [
    "SEM_L1_3",
    "TEM_L1_5",
    "SEM_L1_4",
    "SEM_L1_2",
    "SEM_L2_2",
    "TEM_L3",
    "TEM_adult",
    "SEM_adult",
]


def read_graph(name):
    url = base_url.format(name)
    r = requests.get(url)
    json_graph = r.content
    json_graph = json.loads(json_graph)
    edgelist = pd.DataFrame(json_graph)
    edgelist["weight"] = edgelist["syn"].map(sum)
    edgelist["edge_type"] = edgelist["typ"].map({0: "chem", 2: "elec"})
    graph = nx.from_pandas_edgelist(
        edgelist,
        source="pre",
        target="post",
        edge_attr=True,
        create_using=nx.MultiDiGraph,
    )
    return graph


def load_witvilet_2020():
    graphs = [read_graph(d) for d in datasets]
    return graphs
