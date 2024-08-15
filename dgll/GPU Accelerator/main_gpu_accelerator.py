import argparse

parser = argparse.ArgumentParser(description='Training GNN in Parallel on Multiple GPUs')

parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                    help='Dataset name')
parser.add_argument('--fanout', type=list, default=[4,4],
                    help='Specificy a list of the number of neighbors that a node in a graph is connected to in a specific layer of a graph neural network (GNN) model')
parser.add_argument('--epoch', type=int, default=4,
                    help='Specify number of epochs')
parser.add_argument('--num_gpus', type=int, default=1,
                    help='Specify number of GPUs')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Specify a batch size')
parser.add_argument('--buffer_size', type=int, default=4,
                    help='Specify a buffer size')
parser.add_argument('--GNN_Model', type=str, default='Custom_GNN_Model',
                    help='GCN_Model/GraphSAGE_Model/Specify your own GNN model object by providing just a class name')

args = parser.parse_args()