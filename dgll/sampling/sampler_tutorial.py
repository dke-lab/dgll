from dgllsampler import *
g = pickle.load(open("../dataset/cora.graph", "rb"))
fanout = [2, 3]
seed_nodes = F.tensor([5, 6, 2])
nsampler = DGLLNeighborSampler(fanout)

print(nsampler.sample(g,seed_nodes))