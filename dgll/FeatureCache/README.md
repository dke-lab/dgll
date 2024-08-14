**Feature Cache Module for GNN Training
**
**Prerequisite:**

Python 3

PyTorch (v >= 1.3)

**Files:**

gcn.py file contains our optimized model with feature cache engine for gcn model

gs.py file contains our optimized model with feature cache engine for GraphSage model

dgl_gcn.py file contains pure dgl framework for gcn model

dgl_gs.py file contains pure dgl framework for GraphSage model

**Steps:**

Install python and required libraries.

Run gcn.py to check the training time for GCN model with our feature cache engine

Run gs.py to check the training time for GraphSage model with our feature cache engine

Run dgl_gcn.py to check the training time for pure DGL framework with GCN model

Run dgl_gs.py to check the training time for pure DGL framework with GraphSage model


**Example Command to run the script:**

$ python3 gcn.py --dataset xxx/datasetfolder

$ python3 dgl_gcn.py --dataset xxx/datasetfolder
