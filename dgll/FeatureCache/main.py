import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.data import RedditDataset
from dgl.nn import GCNConv

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, num_classes)

    def forward(self, g, feats):
        h = F.relu(self.conv1(g, feats))
        h = self.conv2(g, h)
        return h

# Load the Reddit dataset
dataset = RedditDataset()
g = dataset[0]
features = torch.Tensor(dataset.features)
labels = torch.LongTensor(dataset.labels)
train_mask = torch.BoolTensor(dataset.train_mask)
val_mask = torch.BoolTensor(dataset.val_mask)
test_mask = torch.BoolTensor(dataset.test_mask)

# Initialize the GCN model and optimizer
in_feats = features.shape[1]
hidden_feats = 16
num_classes = dataset.num_classes
model = GCN(in_feats, hidden_feats, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
cache_size = None
cached_nodes = None
cached_feats = None
for epoch in range(100):
    # Check if the cache needs to be initialized
    if cache_size is None:
        # Determine cache size through first mini batch training
        mini_batch_size = 256
        mini_batch_nodes = torch.nonzero(train_mask).squeeze()
        mini_batch_nodes = mini_batch_nodes[torch.randperm(mini_batch_nodes.shape[0])[:mini_batch_size]]
        mini_batch_feats = features[mini_batch_nodes].to("cuda")
        _ = model(g.to("cuda"), mini_batch_feats)
        cache_size = int(0.1 * mini_batch_size)

        # Cache features of high degree nodes
        deg = g.in_degrees().numpy()
        high_deg_nodes = torch.nonzero(train_mask & (deg > 10)).squeeze()
        high_deg_feats = features[high_deg_nodes].to("cuda")
        cached_nodes = high_deg_nodes.cpu()
        cached_feats = high_deg_feats.cpu()
        print(f"Caching {cache_size} features from {len(high_deg_nodes)} high-degree nodes")

    # Forward
    if cached_nodes is not None:
        # Check if all required features are in the cache
        required_nodes = torch.nonzero(train_mask).squeeze()
        required_feats = features[required_nodes].to("cuda")
        diff_nodes = torch.setdiff1d(required_nodes, cached_nodes)
        if len(diff_nodes) > cache_size:
            # Update cache with new high degree nodes
            new_nodes = diff_nodes[torch.randperm(diff_nodes.shape[0])[:cache_size]]
            new_feats = features[new_nodes].to("cuda")
            cached_nodes = torch.cat([cached_nodes, new_nodes.cpu()])
            cached_feats = torch.cat([cached_feats, new_feats.cpu()])
            print(f"Updating cache with {cache_size} features from {len(new_nodes)} high-degree nodes")
        # Get features from cache
        cached_idx = torch.searchsorted(cached_nodes, required_nodes)
        cached_idx[cached_idx == len(cached_nodes)] = 0
        cached_feats = cached_feats[cached_idx].to("cuda")
        feats = torch.where(cached_idx == len(cached_nodes), required_feats, cached_feats)

    logits = model(g.to("cuda"), feats)
    pred = logits.argmax(1)
    # Compute loss
    loss_fcn = nn.CrossEntropyLoss()
    loss = loss_fcn(logits[train_mask], labels[train_mask])

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluate
    with torch.no_grad():
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean().item()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean().item()
        print(f"Epoch [{epoch + 1}/100] Loss: {loss.item():.4f} Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

# Test
with torch.no_grad():
    test_logits = model(g.to("cuda"), features.to("cuda"))
    test_pred = test_logits.argmax(1)
    test_acc = (test_pred[test_mask] == labels[test_mask]).float().mean().item()
    print(f"Test Acc: {test_acc:.4f}")
