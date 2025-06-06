
### DGLL Sampling

The neighbor sampler plays a key role in creating a subgraph based on the input data. It returns an induced subgraph that directly uses the provided seed nodes as the starting point for sampling.

### Example Usage with `DGLLNeighborSampler`



```python
>>> from dgll.sampling.dgllsampler import DGLLNeighborSampler
>>> from dgll import backend as F

>>> # Load a pre-saved graph from a file
>>> g = pickle.load(open("dgll/dataset/cora.graph", "rb"))

>>> # Define the fan-out for sampling neighbors (2-hop and 3-hop neighbors)
>>> fanout = [2, 3]

>>> # Specify seed nodes for subgraph sampling
>>> seed_nodes = F.tensor([5, 4, 2, 9, 6, 7])

>>> # Initialize the neighbor sampler with the specified fan-out
>>> nsampler = DGLLNeighborSampler(fanout)

>>> # Sample a subgraph based on the seed nodes
>>> input_nodes, seed_nodes, subgs = nsampler.sample(g, seed_nodes)

>>> # Display the input nodes of the sampled subgraph
>>> input_nodes
# tensor([1161,    4,    3,   13,  111,   89,  204,  111,   13,  200,  236,  200,
#          236,  221,   46])

>>> # Display the seed nodes used for sampling
>>> seed_nodes
# tensor([5, 4, 2, 9, 6, 7])

>>> # Display the subgraphs generated by the sampler
>>> subgs
# [<dgll.sampling.base_sampler.Subgraph object at 0x000001A581213580>, 
#  <dgll.sampling.base_sampler.Subgraph object at 0x000001A5812134C0>]

>>> # Access and display the source data of the first subgraph
>>> subgs[0].src_data
# tensor([1161,    4,    3,   13,  111,   89,  204,  111,   13,  200,  236,  200,
#          236,  221,   46])

>>> # Access and display the destination data of the first subgraph
>>> subgs[0].dst_data
# tensor([516, 541, 541, 121, 121, 182, 182, 121, 121,  91,  91, 728, 728,  88,
#         549])

>>> # Access and display the source data of the second subgraph
>>> subgs[1].src_data
# tensor([516, 541,   2, 121, 435,   2, 182, 208, 121,  91, 728,  88, 549])

>>> # Access and display the destination data of the second subgraph
>>> subgs[1].dst_data
# tensor([2, 5, 5, 5, 4, 4, 4, 9, 9, 9, 6, 6, 6, 7])
```

### Explanation

- **Fan-out Definition:** The fan-out is set to `[2, 3]`, meaning we want to sample 2-hop and 3-hop neighbors.
- **Seed Nodes:** The specific nodes `[5, 4, 2, 9, 6, 7]` are used as the starting points for sampling.
- **Neighbor Sampling:** The `DGLLNeighborSampler` is used to sample a subgraph around the seed nodes.
- **Subgraph Details:** The sampled subgraph's source and destination data are printed for both the first and second layers of neighbors.

This example illustrates how to effectively use the `DGLLNeighborSampler` to sample subgraphs from a larger graph in the `dgll` library.



### Development of Custom Samplers

The `base sampler` modules provides the required contituents for graph sampling, i.e., related classes and methods. 

The `Base_sampler` class provides a foundation for implementing custom graph samplers. It also includes utilities for creating subgraphs and generating neighbors for given nodes.


To implement custom graph samplers, you need to subclass the `dgll.sampling.base_sampler` base class and override its abstract sample method. This method should accept the following arguments:

```
def sample(self, g, seed_nodes):
pass
```

- **g**: The original DGLLGraph (DGraph) from which to sample.
- **seed_nodes**: The seed nodes of the current mini-batch.

The function should return the mini-batch of samples.

The code below implements a neighbor sampler:


```
class MyCusomSampler(Base_sampler):
    def __init__(self, fanouts):
      super().__init__()
      self.fanouts = fanouts

    def sample(self, g, seed_nodes):
        output_nodes = seed_nodes
        subgs = []
        print(g, seed_nodes)
        for fanout in reversed(self.fanouts):
            # Sample a fixed number of neighbors of the current seed nodes.
            subg = self.sample_neighbours(g, seed_nodes, fanout)
            seed_nodes = subg.src_nodes()
            subgs.insert(0, subg)
            input_nodes = seed_nodes
        return input_nodes, output_nodes, subgs
```


To use this sampler
```
g = path to g
train_nodeIDs = give training nodes IDs
mysampler =  MyCustomSampler([5, 6])

mydataloader = your data loader with g, train_nodeIDS, and mysampler

for i, input_nodes, output_nodes, subgs in enumerate(mydataloader):
  trainMyGNN(input_nodes, output_nodes, subgs)
```



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* [Data and Knowlege Engineering Lab (DKE)](http://dke.khu.ac.kr/)
<p align="right">(<a href="#top">back to top</a>)</p>
