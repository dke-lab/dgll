import argparse
from CommGNN_train import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--Model', type=str, default='GCN',
                    help='Model name: GCN')
argparser.add_argument('--dataset', type=str, default='cora', help= "'products', 'reddit', 'arxiv'")
argparser.add_argument('--path', type=str, default='dataset/COG/',
                       help='Dataset path: provide path to a relable dataset')
argparser.add_argument('--batch_size', type=int, help='size of output node in a batch', default=5)
argparser.add_argument('--cached_nPercent', type=int, help='how much percent of features to be cached', default=100)
argparser.add_argument('--n_stops', type=int, help='number of steps to stop when F1 not increases', default=50)
argparser.add_argument('--fanouts', type=list, default=[10, 25], help='Number of sampled nodes per node')
argparser.add_argument('--n-epochs', type=int, default=1)
argparser.add_argument('--n-hidden', type=int, default=512)
argparser.add_argument('--n-layers', type=int, default=2)
argparser.add_argument('--n_trial', type=int, default=1,
                       help='Number of times to repeat experiments')
argparser.add_argument('--o_iters', type=int, default=1,
                       help='Number of iteration to run on a batch')
argparser.add_argument('--lr', type=float, default=0.01)
argparser.add_argument('--dropout', type=float, default=0.2)

argparser.add_argument('--gpu', type=int, default=0, help="GPU device ID. Use -1 for CPU training")
args = argparser.parse_args()
# initiate training
run_train(args)



