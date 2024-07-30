from .. import backend as F

"""
This `DGraph` class represents a graph data structure. Here's a summary of what each method does:

- `__init__`: Initializes the graph with nodes, edges, labels, features, and masks for training, testing, and validation subsets.
- `get_neighbors`: Returns the neighbors of a list of nodes.
- `get_labels`: Returns the labels associated with a list of nodes.
- `get_features`: Returns the features associated with a list of nodes.
- `get_train_nodes`: Returns the nodes in the training subset.
- `get_validation_nodes`: Returns the nodes in the validation subset.
- `get_test_nodes`: Returns the nodes in the testing subset.

Note that the `get_neighbors` method assumes that the `edges` attribute is a list where each element is a list of neighbor indices for the corresponding node.
"""


class DGraph(object):
    def __init__(
        self,
        nodes=None,
        edges=None,
        labels=None,
        features=None,
        train_mask=None,
        test_mask=None,
        validation_mask=None,
    ):
        """
        Initializes a DGraph object with the given parameters.

        Args:
            nodes (Optional[List[int]]): A list of node indices.
            edges (Optional[List[List[int]]]): A list of lists representing the edges in the graph.
            labels (Optional[List[int]]): A list of labels associated with each node.
            features (Optional[List[List[float]]]): A list of features associated with each node.
            train_mask (Optional[List[bool]]): A list of booleans indicating whether a node is in the training subset.
            test_mask (Optional[List[bool]]): A list of booleans indicating whether a node is in the testing subset.
            validation_mask (Optional[List[bool]]): A list of booleans indicating whether a node is in the validation subset.
        """
        self.nodes = nodes
        self.edges = edges
        self.labels = labels
        self.features = features
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.validation_mask = validation_mask

    def get_neighbors(self, nodes):
        """
        Get the neighbors of the given nodes.

        Parameters:
            nodes (tensor[int]): A tensor of node indices.

        Returns:
            List[List[int]]: A list of lists containing the neighbors of each node.
        """
        result = []
        for node in nodes:
            result.append(self.edges[node.item()])
        return result

    def get_induced_subgraph(self, nodes):
        """
        Get the induced subgraph of the graph induced by the given nodes.

        Parameters:
            nodes (torch.Tensor): A 1-dimensional tensor of node indices.

        Returns:
            torch.Tensor: A 2-dimensional tensor representing the adjacency matrix of the induced subgraph.
        """
        result = F.zeros(nodes.size(0), nodes.size(0), dtype=F.int32)
        mapping = {j.item(): i for i, j in enumerate(nodes)}
        for node in nodes:
            result[
                [mapping[node.item()]],
                [mapping[edge] for edge in self.edges[node.item()] if edge in mapping],
            ] = 1
        return result

    def get_labels(self, nodes):
        """
        Get the labels associated with the given nodes.

        Parameters:
            nodes (tensor[int]): A tensor of node indices.

        Returns:
            tensor[int]: A tensor of labels corresponding to the given node indices.
        """
        return self.labels[nodes]

    def get_features(self, nodes):
        """
        Get the features associated with the given nodes.

        Parameters:
            nodes (tensor[int]): A tensor of node indices.

        Returns:
            tensor[float]: A tensor of features corresponding to the given node indices.
        """
        return self.features[nodes]

    def get_train_nodes(self):
        """
        Get the indices of nodes that are in the training subset.

        Returns:
            tensor[int]: A tensor containing the indices of nodes in the training subset.
        """
        return self.nodes[self.train_mask]

    def get_validation_nodes(self):
        """
        Get the indices of nodes that are in the validation subset.

        Returns:
            tensor[int]: A tensor containing the indices of nodes in the validation subset.
        """
        return self.nodes[self.validation_mask]

    def get_test_nodes(self):
        """
        Get the indices of nodes that are in the test subset.

        Returns:
            tensor[int]: A tensor containing the indices of nodes in the test subset.
        """
        return self.nodes[self.test_mask]
