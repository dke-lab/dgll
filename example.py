import pickle

"""
NOTE 1: BACKEND
    - Do not use "import torch" directly.
    - Instead, use "from dgll import backend as F".
    - In your code, refer to PyTorch functions using "F." (e.g., "F.tensor").
    - PyTorch is integrated as a backend in the dgll library to ensure consistency and support for multiple backends.

NOTE 2: DOCUMENTATION
    - Pay close attention to source code formatting and adhere to coding standards.
    - Avoid spaghetti code; ensure your code is clean and maintainable.
    - Use Object-Oriented Design principles.
    - Ensure that all User Api functions and classes are properly documented.
    - Final documentation will be generated from the source code.
    - For sample documentation, refer to dgll/data/dgraph.py.

NOTE 3: MODULARITY
    - This library is designed to be modular. Maintain proper modularity in your code.
    - Ensure any new source files are placed in their appropriate locations, are accessible to other modules, and are properly named, formatted, and documented.

NOTE 4: OPEN SOURCE GUIDELINES
    - Strictly follow Open Source Guidelines.
    - The code will undergo a plagiarism check.
    - Do not use DGL or any other third-party libraries.
    - You may use PyTorch for data loading, neural network operations, etc.
    - If you must borrow code, make sure it is marked, properly cited, and attributed to the original author.

NOTE 5: HOW TO USE THE REPOSITORY
    - Create a virtual environment.
    - Clone the repository.
    - Exclude virtual environment files/directories using ".gitignore" to avoid them being pushed to GitHub.
    - Add the names and versions of any new modules installed using pip to the "requirements.txt" file.
    - Please do not commit your changes directly to the main code base. Instead, issue a Pull Request to merge your changes. This allows us to review and discuss the changes before integrating them into the main codebase.
    - Every merge request will undergo a thorough review, ensuring code quality, proper documentation, formatting, and functionality before merging. Please consider this process carefully before issuing a merge request, as timely review is crucial.
    - If you encounter any issues with code owned by others, please create an issue and contact the responsible person for resolution.
    - For feature requests, please use the Discussion Board.
    - Please ensure you provide a proper summary before committing your changes.

NOTE 5: IDENTIFICATION AND ATTRIBUTION
    - Please add your name, date, and version at the top of each source file you own. Additionally, ensure to include this information for any changes you make in code owned by others.
    - Also, include your name in the comments or documentation for any changes you make to code owned by others for identification and attribution purposes.
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
