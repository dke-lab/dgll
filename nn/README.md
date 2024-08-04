# Graph neural networks library 

### Introduction
This package contains implementation for popular Graph neural network models such as GCN, GraphSage, and GAT etc.
It also contains Global Pooling methods such as sum, mean, and max
In future we will add more GNNs models and will improve its performance both in a single
and distributed environment for GNNs training and inferences.

#### Requirements
```
Python 3.6 >
PyTorch 1.4 >
numpy==1.14.1
scipy==1.0.0
networkx==2.1
tensorflow-gpu==1.6.0
```

### How to utilize a code? (Tutorial)

To utlilize Implemented GNNs we have provided examples in the repository.
```
install DGLL package then add
from utils.utils import load_data, load_khop, accuracy
from gnn.Convolution import gcnConv
create model and do training as:

# Model and optimizer
class GCN(torch.nn.Module):
    def __init__(self, in_features, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.nhid = nhid
        self.nclass = nclass
        self.dropout = dropout
        self.gcn1 = gcnConv(in_features, nhid)
        self.gcn2 = gcnConv(nhid, nclass)

    def forward(self, x, adj):
        h1 = F.relu(self.gcn1(x, adj))
        h1_d = F.dropout(h1, self.dropout, training=self.training)
        logits = self.gcn2(h1_d, adj)
        output = F.log_softmax(logits, dim=1)
        return output

model = GCN(in_features=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
```



