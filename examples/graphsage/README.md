Inductive Representation Learning on Large Graphs (GraphSAGE)
============

- Paper link: [http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)
- Author's code repo: [https://github.com/williamleif/graphsage-simple](https://github.com/williamleif/graphsage-simple)

Requirements
------------

```bash
pip install requests torchmetrics==0.11.4 ogb
```

How to run
-------

### Full graph training

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train_full.py --dataset cora --gpu 0    # full graph
```

#### Results:
* cora: ~0.8330
* citeseer: ~0.7110
* pubmed: ~0.7830


### Minibatch training for node classification

Train w/ mini-batch sampling in mixed mode (CPU+GPU) for node classification on "ogbn-products"

```bash
python3 node_classification.py
```

#### Results:
Test Accuracy: 0.7632
