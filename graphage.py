# This code performs the following steps:

# Loads the graph data.
# Sets up training parameters.
# Creates data loaders for training, validation, and testing using your DataLoader class.
# Initializes the GraphSAGE model.
# Defines the optimizer and loss function.
# Runs the training loop for a specified number of epochs.
# Validates the model after each epoch.
# Tests the model after training is complete.

import pickle
from dgll import backend as F
from dgll.data.dgraph import Dgraph
from dgll.sampling.dgllsampler import DGLLNeighborSampler
from dgll.dataloader import DataLoader
from dgll.nn.Convolution.sageconv import *


# Load the graph
g = pickle.load(open("dgll/dataset/cora.graph", "rb"))

# Training settings
batch_size = 32
epochs = 20
learning_rate = 0.01
num_neighbors_list = [10, 10]
input_dim = g.feature_size()  # Assuming the graphpip object has a method to get feature size
hidden_dim = [64, 64]

# Create train, validation, and test dataloaders
train_nodes = g.get_train_nodes()
validation_nodes = g.get_validation_nodes()
test_nodes = g.get_test_nodes()

sampler = DGLLNeighborSampler(num_neighbors_list)
train_dataloader = DataLoader(g, train_nodes, sampler, batch_size=batch_size)
val_dataloader = DataLoader(g, validation_nodes, sampler, batch_size=batch_size)
test_dataloader = DataLoader(g, test_nodes, sampler, batch_size=batch_size)

# Initialize the model
model = GraphSage(input_dim, hidden_dim, num_neighbors_list)
optimizer = F.optim.Adam(model.parameters(), lr=learning_rate)
criterion = F.nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_nodes, output_nodes, subgraphs = batch
        node_features = g.get_features(input_nodes)
        labels = g.get_labels(output_nodes)

        optimizer.zero_grad()
        outputs = model(node_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_dataloader)}')

    # Validation
    model.eval()
    correct = 0
    total = 0
    with F.no_grad():
        for batch in val_dataloader:
            input_nodes, output_nodes, subgraphs = batch
            node_features = g.get_features(input_nodes)
            labels = g.get_labels(output_nodes)

            outputs = model(node_features)
            _, predicted = F.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total}%')

# Test the model
model.eval()
correct = 0
total = 0
with F.no_grad():
    for batch in test_dataloader:
        input_nodes, output_nodes, subgraphs = batch
        node_features = g.get_features(input_nodes)
        labels = g.get_labels(output_nodes)

        outputs = model(node_features)
        _, predicted = F.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total}%')
