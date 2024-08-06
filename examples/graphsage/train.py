from dgll.sampling.dgllsampler import DGLLNeighborSampler
from dgll.dataset import *
import pickle
from dgll import backend as F
g = pickle.load(open("../../dgll/dataset/cora.graph", "rb"))
fanout = [2, 3]
seed_nodes = F.tensor([5, 4, 2, 9, 6, 7])
nsampler = DGLLNeighborSampler(fanout)