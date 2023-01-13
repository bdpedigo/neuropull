# Metadata

Draft of codifying a standard method for naming fields.

## Network metadata

`publications` :

`name` :

`age` : the age of the organism when its connectome was mapped
- Time may be specified via ____ codes

`sex` : the sex of the organism.
- Can be one of `{"hermaphrodite", "female", "male"}`

## Node metadata

`node_id` : the unique descriptor used to define each node, i.e. a skeleton ID from CATMAID

`name` : a textual description of the node, distinct from `node_id`

`side` : the side a neuron is on in the organism
- Can be one of `{"left", "right", "center", "unknown"}`

`io` : short for input/output, defines the very high level role of a neuron in the sensory-motor transformation
- Can be one of `{"input", "output", "intrinsic"}`



## Edge metadata
