import argparse
from cog_train import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='arxiv')
argparser.add_argument('--path', type=str, default='dataset/relabel_dataset/',
                       help='Dataset path: provide path to a relable dataset')
argparser.add_argument('--batch_size', type=int, help='size of output node in a batch', default=512)
argparser.add_argument('--n_samp', type=int, default=512, help='Number of sampled nodes per node')
argparser.add_argument('--n-epochs', type=int, default=3)
argparser.add_argument('--n-hidden', type=int, default=512)
argparser.add_argument('--n-layers', type=int, default=2)
argparser.add_argument('--samp_growth_rate', type = float, default = 2,
                       help='Growth rate for node-wise sampling')
argparser.add_argument('--n_trial', type=int, default=5,
                       help='Number of times to repeat experiments')
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--dropout', type=float, default=0.2)
argparser.add_argument('--n-stops', type=int, default=200,
                       help='Stop after number of batches that f1 dont increase')
argparser.add_argument('--o_iters', type=int, default=2,
                    help='Number of iteration to run on a batch')
argparser.add_argument('--record-f1', action='store_false',
                       help='whether record the f1 score')
argparser.add_argument('--samp_type', type=str, default='hybrid', help='hybrid')
argparser.add_argument('--gpu', type=int, default=0, help="GPU device ID. Use -1 for CPU training")
args = argparser.parse_args()
# initiate training
run_train(args)



