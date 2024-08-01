import pickle
from dgll import backend as F
from base_sampler import Base_sampler

class DGLLNeighborSampler(Base_sampler):
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
        return input_nodes, output_nodes, [subgs, subg.get_features(g, subgs)]

