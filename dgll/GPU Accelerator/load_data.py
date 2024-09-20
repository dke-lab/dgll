from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import PubmedGraphDataset
from dgl.data import RedditDataset
from dgl.data import CoraGraphDataset
from dgl.data import CiteseerGraphDataset
from dgl.data.utils import load_graphs

def load_reddit(root):
    data = RedditDataset(raw_dir=root, self_loop=True)
    g = data[0]
    # mlog(f'finish loading Reddit, time elapsed: {time.time() - tic:.2f}s')
    return g



def load_pubmed(root):
    data = PubmedGraphDataset(raw_dir=root, reverse_edge=True)
    g = data[0]
    # mlog(f'finish loading Reddit, time elapsed: {time.time() - tic:.2f}s')

    return g

def load_citeseer(root):
    data = CiteseerGraphDataset(raw_dir=root, reverse_edge=True)
    g = data[0]
    # mlog(f'finish loading Reddit, time elapsed: {time.time() - tic:.2f}s')

    return g

def load_citeseer(root):
    data = CiteseerGraphDataset(raw_dir=root, reverse_edge=True)
    g = data[0]
    # mlog(f'finish loading Reddit, time elapsed: {time.time() - tic:.2f}s')

    return g

def load_arxiv(path):
    g= DglNodePropPredDataset('ogbn-arxiv')
    # mlog(f'finish loading Reddit, time elapsed: {time.time() - tic:.2f}s')

    return g

def load_cora(path):
    g= CoraGraphDataset()
    # mlog(f'finish loading Reddit, time elapsed: {time.time() - tic:.2f}s')

    return g

def load_products(path):
    g= DglNodePropPredDataset('ogbn-products')
    # mlog(f'finish loading Reddit, time elapsed: {time.time() - tic:.2f}s')

    return g

