import pickle

"""
NOTE 1 VERY IMPORTANT: BACKEND
------------------------------
Do not use "import torch"
Instead, use "from dgll import backend as F"
In your code, use "F.tensor" or any PyTorch functions using "F."
PyTorch is already included as a backend in the dgll library to ensure consistency and the possibily of multiple backends.

NOTE 2 VERY IMPORTANT: DOCUMENTATION
------------------------------------
- Please pay close attention to source code formatting, adhere to coding standards.
- Avoid spaghetti code, ensure clean code.
- Use Object Oriented Design.
- Pay special attention to the documentation inside code.
  Make sure all the user_api is properly documented.
  Our final documentation will be generated from source code.
- For sample documentation, please refer to dgll/data/dgraph.py.

NOTE 3 VERY IMPORTANT: MODULARITY
---------------------------------
- This library is modular. Please maintain proper modularity.
- Make sure any new source file you add is in its proper place, accessible to other modules, properly named, formatted, and documented.
"""
from dgll import backend as F

# Load the graph
# You can download cora.graph and products.graph from the server
g = pickle.load(open("cora.graph", "rb"))

# A set of node IDs
nodes = F.tensor([0, 5, 6, 9, 23])

# Get the neighbors of the nodes
neighbors = g.get_neighbors(nodes)

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
print(labels)
print(features)
print(train_nodes)
print(validation_nodes)
print(test_nodes)


# Output
# [[809, 1217, 1218], [2, 114, 516, 541], [88, 91, 200, 226, 728], [2, 89, 121, 182, 208], [505, 582]]
# tensor([1, 4, 6, 4, 4])
# tensor([[0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0]], dtype=torch.int32)
# tensor([   0,    9,   10,  ..., 2701, 2704, 2706])
# tensor([   2,    7,    8,  ..., 2697, 2702, 2707])
# tensor([   1,    3,    4,  ..., 2700, 2703, 2705])
