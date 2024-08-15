# Graph Embedding (GE) Module
### Introduction
This module provides the services and implementation for various graph embedding models.


## Usage and Tutorial
#### input graph
```
# import module
import ge


# Set Path to Data
data_dir = "Your Path to Data"
dataset = "File/Dataset Name"


# Load a Graph
inputGraph = ge.loadGraph(data_dir, dataset)
```

#### Configurable Parameter for Graph Embedding
```
embedDim = 2 # embedding size
numbOfWalksPerVertex = 2 # walks per vertex
walkLength = 4 # walk lenght
lr =0.025 # learning rate
windowSize = 3 # window size
```

#### Choose One of the Following Graph Embedding Models
```
# DeepWalk
rw = ge.DeepWalk(inputGraph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr = lr)
              
 ```
```
# Node2Vec
rw = ge.Node2vec(inputGraph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
               windowSize=windowSize, lr=lr, p = 0.5, q = 0.8)
```
```
# Struc2Vec
rw = ge.Struc2Vec(inputGraph, walkLength=walkLength, embedDim=embedDim, numbOfWalksPerVertex=numbOfWalksPerVertex, \
              windowSize=windowSize, lr = lr)
```
              
#### Skip Gram model
```
modelSkipGram = ge.SkipGramModel(rw.totalNodes, rw.embedDim)
```
#### Want Node Embedding or Edge Embedding
```
# Learning Node Embedding
model = rw.learnNodeEmbedding(modelSkipGram)
```


```
# Learning Edge Embedding
model = rw.learnEdgeEmbedding(modelSkipGram)
```

#### Save Embedding to Disk
```
ge.saveEmbedding(data_dir, dataset, rw)
```
#### Generate  Embedding for a Specific Node or Edge
```
node1 = 35
node2 = 40

# Get Embedding for a node
emb = rw.getNodeEmbedding(node1)
print("Node Embedding", emb)

# Get Embedding for an edge
emb = rw.getEdgeEmbedding(node1, node2)
print("Edge Embedding", emb)
```

### Utilize embedding for training classification models/classifiers

```
# import modules
import ge
import numpy as np
import scipy.sparse as sp
```

#### Load Embedding
```
features, labels = loadEmbedding(your_data_set_name.embedding, dtype=np.dtype(str))
```

#### Prepare Train test data
```
X_train, X_test, y_train, y_test = ge.prepareTrainTestData(features, labels, test_size=0.33)
```
#### Train Classifiers
```
y_pred = ge.classify(X_train, y_train, X_test, classifier = "MLP")
```
#### Get Evaluation Metrics
```
print(ge.evaluation_metrics(y_test, y_pred))
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* [Data and Knowlege Engineering Lab (DKE)](http://dke.khu.ac.kr/)
<p align="right">(<a href="#top">back to top</a>)</p>
