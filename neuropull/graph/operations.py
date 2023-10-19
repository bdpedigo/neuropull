import pandas as pd

from .network_frame import NetworkFrame


def concat(networks):
    """
    Concatenate a list of NetworkFrames.

    Note that this currently requires nodes and edges (source/target) to be unique.
    """
    # TODO handle directed + undirected, etc.
    nodes = pd.concat([n.nodes for n in networks], axis=0, verify_integrity=True)
    old_index = networks[0].edges.index
    edges = pd.concat(
        [n.edges.set_index(["source", "target"], drop=False) for n in networks],
        axis=0,
        verify_integrity=True,
    ).set_index(old_index, drop=False)
    return NetworkFrame(nodes, edges, directed=networks[0].directed)
