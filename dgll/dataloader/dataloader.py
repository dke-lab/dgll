from dgll import backend as F
from dgllsampler import *

class DataLoader:
    def __init__(self,
        Dgraph,
        train_nodes,
        sampler,
        batch_size=1,
        device=None
    ):
        self.Dgraph = Dgraph
        self.sampler = sampler
        self.train_nodes = train_nodes
        self.batch_size = batch_size
        self.device = device

    def sample(self):
        for i in range(0, len(self.data), self.batch_size):
            seed_nodes = self.data[i:i+batch_size]
            yield self.sampler.sample(self.Dgraph, seed_nodes)

    def __iter__(self):
        return self.sample()
