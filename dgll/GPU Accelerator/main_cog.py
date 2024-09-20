import argparse
from cog import *




parser = argparse.ArgumentParser(description='Generate COG: cora/pubmed/citeseer/arxiv/reddit/products Datasets')

'''
   Input Arguments
'''

parser.add_argument('--dataset', type=str, default='arxiv',
                    help='Dataset name: cora/pubmed/citeseer/arxiv/reddit/products')
parser.add_argument('--output_path_dataset', type=str, default='dataset/relabel_dataset/',
                    help='Dataset name: cora/pubmed/citeseer/arxiv/reddit/products')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')

args = parser.parse_args()

# initiate cog generation
run_cog(args)