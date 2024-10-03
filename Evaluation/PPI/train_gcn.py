import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from ppi_dataloader import load_ppi_dataset
from gcn_model import GCN

# Modify this path to point to the PPI directory
data_dir = 'data/PPI'

# Load dataset
train_data = load_ppi_dataset(data_dir, 'train')
valid_data = load_ppi_dataset(data_dir, 'valid')
test_data = load_ppi_dataset(data_dir, 'test')

# Define GCN model parameters (assuming the first graph for input dimensions)
input_dim = train_data[0][1].shape[1]  # Number of node features (x)
hidden_dim = 64  # Hidden dimension
output_dim = train_data[0][2].shape[1]  # Number of output classes (y)
num_layers = 3 # Try 1, 2, or 3 layers

# Instantiate GCN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim, hidden_dim, output_dim, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train(model, data_list, optimizer, criterion, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()

        for edge_index, features, labels in data_list:
            # Move data to the GPU (or CPU based on the device)
            edge_index = edge_index.to(device)
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass through the model
            output = model(edge_index, features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Time per Epoch: {epoch_time:.4f}s")


# Run training
train(model, train_data, optimizer, criterion)
