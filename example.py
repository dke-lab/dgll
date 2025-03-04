import pickle

"""
NOTE: VERY IMPORTANT
--------------------
Do not use "import torch"
Instead, use "from dgll import backend as F"
In your code, use "F.tensor" or any PyTorch functions using "F."
PyTorch is already included as a backend in the dgll library to ensure consistency and the possibily of multiple backends.
"""
from dgll import backend as F

# Load the graph
# You can download cora.graph and products.graph from the server

# Cora
g = pickle.load(open("dgll/dataset/cora.graph", "rb"))

# ogbn-products
# g = pickle.load(open("products.graph", "rb"))

# A set of node IDs
nodes = F.tensor([0, 2, 5, 6, 9, 23])

# Get the neighbors of the nodes
neighbors = g.get_neighbors(nodes)

# Get Induced Subgraph
induced_subgraph = g.get_induced_subgraph(nodes)

# Get the labels of the nodes
labels = g.get_labels(nodes)

# Get the features of the nodes
features = g.get_features(nodes)

# Get the nodes that are in the training subset
train_nodes = g.get_train_nodes()

# Get the nodes that are in the validation subset
validation_nodes = g.get_validation_nodes()

# Get the nodes that are in the test subset
test_nodes = g.get_test_nodes()

# Print the results
print(neighbors)
print(induced_subgraph)
print(labels)
print(features)
print(train_nodes)
print(validation_nodes)
print(test_nodes)


# Output
# [[809, 1217, 1218], [], [2, 114, 516, 541], [88, 91, 200, 226, 728], [2, 89, 121, 182, 208], [505, 582]]
# tensor([[0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0]], dtype=torch.int32)
# tensor([1, 4, 4, 6, 4, 4])
# tensor([[0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0]], dtype=torch.int32)
# tensor([   0,    9,   10,  ..., 2701, 2704, 2706])
# tensor([   2,    7,    8,  ..., 2697, 2702, 2707])
# tensor([   1,    3,    4,  ..., 2700, 2703, 2705])
