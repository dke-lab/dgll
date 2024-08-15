
# Training GNN in Parallel on Multiple GPUs

This repository contains a script for training Graph Neural Networks (GNNs) in parallel on multiple GPUs. The script allows users to specify various training parameters, such as the dataset, number of GPUs, GNN model, and other hyperparameters through command-line arguments.

## Requirements

- Python 3.7 or higher
- PyTorch 1.9 or higher

### Arguments
Set the following arguments in main_gpu_accelerator.py, and run the script.

- `--dataset` (str): Specify the dataset name. Default is `'ogbn-arxiv'`.

- `--fanout` (list): Specify a list of the number of neighbors that a node in a graph is connected to in each layer of the GNN model. Default is `[4,4]`.

- `--epoch` (int): Specify the number of epochs for training. Default is `4`.

- `--num_gpus` (int): Specify the number of GPUs to use for parallel training. Default is `1`.

- `--batch_size` (int): Specify the batch size for training. Default is `1024`.

- `--buffer_size` (int): Specify the buffer size. Default is `4`.

- `--GNN_Model` (str): Specify the GNN model to use. Options are `GCN_Model`, `GraphSAGE_Model`, or provide a custom GNN model class name. Default is `'Custom_GNN_Model'`.


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* [Data and Knowlege Engineering Lab (DKE)](http://dke.khu.ac.kr/)
<p align="right">(<a href="#top">back to top</a>)</p>
