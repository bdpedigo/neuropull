# NetworkFrame
A lightweight package for manipulating networks with node attributes via Pandas.

## Motivation
Many network implementations exist in Python, with their own advantages/disadvantages.
I often work with attributed networks (e.g., where the nodes have features), and I find
myself using Pandas to do common operations on a dataframe of node features. For example,
selecting a subset of nodes, reordering nodes according to some feature, etc. I have yet
to see a Python package capable of combining the power/ease-of-use of Pandas with
network (or matrix) data.

## Design principles
- Pandas-esque
  - If you're familiar with using Pandas, then using a NetworkFrame should feel natural
- Lightweight
  - There are many reasonable ways of storing a network in Python, which already have lots of support. We don't want to reinvent the wheel, so these objects are designed to be lightweight wrappers around current formats.
- Extensible
  - If your favorite network format is unavailable, it should be fairly easy to extend the base classes to implement your format.

## Support
- Adjacency matrices
  - Numpy/Pandas
  - Scipy sparse arrays
- NetworkX (maybe someday)
- Edgelists (maybe someday)
- Multiplex network versions of the above

## Features
### Indexing and sorting
- [ ] `loc`
  - Select nodes according to an index
- [ ] `iloc`
  - Select nodes according to a positional index
- [x] `reindex`
  - Reorder a network according to some index, handling new nodes appropriately
- [ ] `set_index`
  - Reindex the network according to a unique feature of the nodes
- [ ] `union`
  - Take the union of nodes with another NetworkFrame
- [ ] `intersection`
  - Take the intersection of nodes with another NetworkFrame
- [x] `sort_values`
  - Sort nodes according to values in metadata
- [ ] `sort_values_on`
  - Sort according to implicit values from a grouping

### Selecting and grouping
- [x] `groupby`
  - Group nodes into subgraphs according to some feature
  - [ ] Add an `induced` option to groupby (basically only get the diagonal blocks)
- [x] `query`
  - Subselect networks according to node feature values
- [ ] `pair`
  - Something for working with paired data...

### Computing and mapping
- [ ] `map`...
- [x] `groupby.apply()` divide data into groups and then apply a function to each, collating the results

### Mathematical operations
- [x] `__add__` (`+` operator)
  - Add the adjacency matrices/edges of two networks
