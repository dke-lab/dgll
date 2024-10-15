
**GPU Accelerator Module for GNN Training**


**Requirements**

- Python 3.7 or higher
- PyTorch 1.9 or higher




```
##### Configurable parameters for GNN training
Following are the user-defined configurable parameters in main.py
```
    parser = argparse.ArgumentParser(
        description='Training GCN/GraphSAGE on cora/citeseer/pubmed/proteins/arxiv/reddit/prdoucts Datasets')

    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset name: cora/citeseer/pubmed/proteins/arxiv/reddit/products')
    parser.add_argument('--samp_type', type=str, default='fastgcn',
                        help='node/fastgcn/fastgcnflat/fastgcnwrs/fastflatwrs/ladies/ladieswrs/ladiesflat/ladieswrs/ladiesflatwrs')
    # parser.add_argument('--samp_type', type=str, default='ladies', help='Sampling type: node/ladies/fastgcn')

    parser.add_argument('--Model', type=str, default='GCN',
                        help='Model name: GCN/GraphSAGE')

    parser.add_argument('--n_samp', type=int, default=5,
                        help='Number of sampled nodes per layer or per node (keep in mind LDS and Node samplers)')
    parser.add_argument('--nhid', type=int, default=256,
                        help='Hidden state dimension')
    parser.add_argument('--n_epochs', type=int, default=300,
                        help='Number of Epoch')
    parser.add_argument('--n_stops', type=int, default=200,
                        help='Stop after number of batches that f1 dont increase')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Specify number of GPUs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='size of output node in a batch')
    # parser.add_argument('--n_iters', type=int, default=1,
    #                     help='Number of iteration to run on a batch')
    parser.add_argument('--n_trial', type=int, default=1,
                        help='Number of times to repeat experiments')
    parser.add_argument('--record_f1', action='store_false',
                        help='whether record the f1 score')
    parser.add_argument('--samp_growth_rate', type=float, default=1,
                        help='Growth rate for layer-wise sampling')
    parser.add_argument('--batch_num', type=int, default=1,
                        help='Maximum Batch Number')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GNN layers')

    args = parser.parse_args()
```
##### Run the code
From the main directory excute the following command:
```
python main.py
```





<!-- ACKNOWLEDGMENTS -->
**Acknowledgments**
* [Data and Knowlege Engineering Lab (DKE)](http://dke.khu.ac.kr/)
<p align="right">(<a href="#top">back to top</a>)</p>
