import os
import json
import numpy as np
import torch
from networkx.readwrite import json_graph
import networkx as nx


# Function to read the dataset files and prepare graph structures
def load_ppi_dataset(data_dir: str, split: str):
    """
    Loads the PPI dataset for the specified split (train/val/test).

    Args:
        data_dir (str): Directory where the PPI data files are stored.
        split (str): The split to load (either 'train', 'valid', or 'test').

    Returns:
        A list of tuples containing (edge_index, features, labels) for each graph.
    """
    # Load the graph structure
    graph_path = os.path.join(data_dir, f'{split}_graph.json')
    with open(graph_path, 'r') as f:
        G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

    # Load the features
    feats_path = os.path.join(data_dir, f'{split}_feats.npy')
    x = np.load(feats_path)
    x = torch.from_numpy(x).float()

    # Load the labels
    labels_path = os.path.join(data_dir, f'{split}_labels.npy')
    y = np.load(labels_path)
    y = torch.from_numpy(y).float()

    # Load the graph ID mapping
    graph_id_path = os.path.join(data_dir, f'{split}_graph_id.npy')
    idx = torch.from_numpy(np.load(graph_id_path)).long()
    idx = idx - idx.min()  # Normalize IDs to start from 0

    # Now, let's build the data for each graph in the dataset
    graph_list = []

    # Iterate over each unique graph in the dataset
    for i in range(int(idx.max()) + 1):
        mask = idx == i  # Select the nodes for the current graph

        # Extract the subgraph for the current graph
        G_s = G.subgraph(mask.nonzero(as_tuple=False).view(-1).tolist())
        edge_index = torch.tensor(list(G_s.edges)).t().contiguous()

        # Adjust edge indices to start from 0
        edge_index = edge_index - edge_index.min()

        # Remove self-loops (if any)
        edge_index = remove_self_loops(edge_index)

        # Add the tuple (edge_index, features, labels) to the graph list
        graph_list.append((edge_index, x[mask], y[mask]))

    return graph_list


def remove_self_loops(edge_index):
    """
    Removes self-loops from the edge index.
    """
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


# Test the data loader
if __name__ == "__main__":
    data_dir = 'path_to_your_data_directory'
    train_data = load_ppi_dataset(data_dir, 'train')
    valid_data = load_ppi_dataset(data_dir, 'valid')
    test_data = load_ppi_dataset(data_dir, 'test')

    print(f"Train graphs: {len(train_data)}")
    print(f"Validation graphs: {len(valid_data)}")
    print(f"Test graphs: {len(test_data)}")
