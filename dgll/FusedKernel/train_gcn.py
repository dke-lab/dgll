#train_gcn.py
import torch
from torch_geometric.datasets import PPI
from torch_geometric.utils import to_undirected
import gcn_extension
import torchmetrics.functional

class GCNFusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, row_ptr, col_idx, values, X, W, num_neighbors, actual_F):
        output = gcn_extension.gcn_fused_forward(row_ptr, col_idx, values, X, W, num_neighbors, actual_F)
        ctx.save_for_backward(row_ptr, col_idx, values, X, W, num_neighbors, output)
        ctx.actual_F = actual_F
        return output

    @staticmethod
    def backward(ctx, grad_output):
        row_ptr, col_idx, values, X, W, num_neighbors, output = ctx.saved_tensors
        actual_F = ctx.actual_F
        grad_X, grad_W = gcn_extension.gcn_fused_backward(grad_output, row_ptr, col_idx, values, X, W, num_neighbors, actual_F)
        # Return gradients for all inputs; None for those not requiring grad
        return None, None, None, grad_X, grad_W, None, None

class GCNLayer(torch.nn.Module):
    def __init__(self, in_features_padded, actual_in_features, out_features):
        super(GCNLayer, self).__init__()
        scale = 1.0 / torch.sqrt(torch.tensor(actual_in_features, dtype=torch.float32))
        self.W = torch.nn.Parameter(
            torch.randn(in_features_padded, out_features, dtype=torch.float32, device='cuda') * scale
        )
        self.actual_F = actual_in_features

    def forward(self, row_ptr, col_idx, values, X, num_neighbors):
        # print(f"row_ptr: {row_ptr.shape}, {row_ptr.dtype}, {row_ptr.device}")
        # print(f"col_idx: {col_idx.shape}, {col_idx.dtype}, {col_idx.device}")
        # print(f"values: {values.shape}, {values.dtype}, {values.device}")
        # print(f"X: {X.shape}, {X.dtype}, {X.device}")
        # print(f"W: {self.W.shape}, {self.W.dtype}, {self.W.device}")
        # print(f"actual_F: {self.actual_F}")
        # print("Calling gcn_fused kernel...")
        output = GCNFusedFunction.apply(row_ptr, col_idx, values, X, self.W, num_neighbors, self.actual_F)
        # print("Kernel call completed.")
        return output

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.input_dim_padded = ((input_dim + 3) // 4) * 4
        self.layer1 = GCNLayer(self.input_dim_padded, input_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, hidden_dim, output_dim)

    def forward(self, row_ptr, col_idx, values, X, num_neighbors):
        X_padded = torch.zeros(X.size(0), self.input_dim_padded, device='cuda', dtype=torch.float32)
        X_padded[:, :X.size(1)] = X
        h1 = self.layer1(row_ptr, col_idx, values, X_padded, num_neighbors)
        h2 = self.layer2(row_ptr, col_idx, values, h1, num_neighbors)
        return h2

# Load dataset
dataset = PPI(root='./data/PPI', split='train')
data = dataset[0]

# Prepare adjacency matrix
edge_index = to_undirected(data.edge_index)
row, col = edge_index
N = data.num_nodes
A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (N, N))
D = torch.sparse.sum(A, dim=1).to_dense()
D_inv_sqrt = 1.0 / torch.sqrt(D)
D_inv_sqrt[D == 0] = 0
A_hat = torch.sparse_coo_tensor(edge_index, D_inv_sqrt[row] * D_inv_sqrt[col], (N, N)).coalesce()

# Convert to CSR format
row_ptr = torch.cat([torch.tensor([0], dtype=torch.int32), 
                     torch.cumsum(torch.sparse.sum(A_hat, dim=1).to_dense(), dim=0).to(torch.int32)]).cuda()
col_idx = A_hat.indices()[1].to(torch.int32).cuda()
values = A_hat.values().cuda()
num_neighbors = torch.diff(row_ptr).int().cuda()

X = data.x.cuda()
Y = data.y.cuda()

# Initialize model and optimizer
gcn = GCN(input_dim=dataset.num_features, hidden_dim=64, output_dim=dataset.num_classes)
optimizer = torch.optim.Adam(list(gcn.parameters()), lr=0.01)

# # Debugging prints
# print(f"row_ptr: {row_ptr.shape}, {row_ptr.dtype}, {row_ptr.device}")
# print(f"col_idx: {col_idx.shape}, {col_idx.dtype}, {col_idx.device}")
# print(f"values: {values.shape}, {values.dtype}, {values.device}")
# print(f"X: {X.shape}, {X.dtype}, {X.device}")
# print(f"W1: {gcn.layer1.W.shape}, {gcn.layer1.W.dtype}, {gcn.layer1.W.device}")

# Training loop
for epoch in range(100):
    h = gcn(row_ptr, col_idx, values, X, num_neighbors)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(h, Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    h = gcn(row_ptr, col_idx, values, X, num_neighbors)
    pred = (h > 0).float()
    f1 = torch.mean(torch.tensor([torchmetrics.functional.f1_score(pred[i], Y[i], task='binary') 
                                  for i in range(N) if Y[i].sum() > 0]))
    print(f"Training Micro-F1 Score: {f1.item():.4f}")