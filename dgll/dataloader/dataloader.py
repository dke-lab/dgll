import torch
from torch.utils.data import Dataset, DataLoader


class GraphDataset(Dataset):
    def __init__(self, dgraph, subset='train'):
        """
        Initializes the GraphDataset with the given DGraph object and subset.

        Args:
            dgraph (DGraph): The DGraph object containing the graph data.
            subset (str): The subset of data to use ('train', 'validation', or 'test').
        """
        self.dgraph = dgraph

        if subset == 'train':
            self.nodes = dgraph.get_train_nodes()
        elif subset == 'validation':
            self.nodes = dgraph.get_validation_nodes()
        elif subset == 'test':
            self.nodes = dgraph.get_test_nodes()
        else:
            raise ValueError("subset must be 'train', 'validation', or 'test'")

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        node = self.nodes[idx]
        label = self.dgraph.get_labels(node)
        features = self.dgraph.get_features(node)
        neighbors = self.dgraph.get_neighbors(node)

        return {
            'node': node,
            'label': label,
            'features': features,
            'neighbors': neighbors
        }


# # Example usage
# # Assuming dgraph is an instance of DGraph
#
# # Create datasets for each subset
# train_dataset = GraphDataset(dgraph, subset='train')
# validation_dataset = GraphDataset(dgraph, subset='validation')
# test_dataset = GraphDataset(dgraph, subset='test')
#
# # Create data loaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# # Example loop over the training data
# for batch in train_loader:
#     nodes = batch['node']
#     labels = batch['label']
#     features = batch['features']
#     neighbors = batch['neighbors']

    # Training step with nodes, labels, features, and neighbors
    # model.train_step(nodes, labels, features, neighbors)
