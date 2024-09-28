import os
os.environ['dgllBACKEND'] = 'pytorch'
import dgll
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgll.nn import SAGEConv
import tqdm
import sklearn.metrics
import time
from dgll.nn import GraphConv
import torch.multiprocessing as mp
# from torch.multiprocessing import Lock, Condition, Process, Queue
# from torch.multiprocessing import Process as Thread
import concurrent.futures
import threading
from queue import Queue
import asyncio
from contextlib import nullcontext
from buffer_queues import sample_generator, sample_consumer
import dgll
from utils import matrix_row_normalize, estWRS_weights, normalize_lap
import scipy.sparse as sp
import numpy as np
import torch

dataset = dgllNodePropPredDataset('ogbn-arxiv')
fanout = [10, 25]
batch_size = 1023

graph, node_labels = dataset[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgll.add_reverse_edges(graph)
graph.ndata['label'] = node_labels[:, 0]

node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()

idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']    # Test node IDs, not used in the tutorial though.

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h

class FastGCNSampler(dgll.dataloading.Sampler):
    def __init__(self, fanouts, g):
        super().__init__()
        self.fanouts = fanouts
        adj_matrix = g.adj_external(scipy_fmt="csr")
        self.lap_matrix = normalize_lap(adj_matrix + sp.eye(adj_matrix.shape[0]))
        del adj_matrix
        self.layers = len(self.fanouts)
        self.num_nodes = g.num_nodes()


    def sample(self, g, batch_nodes):
        prev_nodes_list = batch_nodes
        prob_i = np.array(np.sum(self.lap_matrix.multiply(self.lap_matrix), axis=0))[0]
        prob = prob_i / np.sum(prob_i)

        subgs = []
        for l in range(self.layers):
            Q = self.lap_matrix[prev_nodes_list , :]
            s_num = np.min([np.sum(prob > 0), self.fanouts[l]])
            next_nodes_list = np.random.choice(self.num_nodes, s_num, p = prob, replace = False)
            next_nodes_list = np.unique(np.concatenate((next_nodes_list, batch_nodes)))
            adj = Q[: , next_nodes_list].multiply(1/prob[next_nodes_list]/s_num).tocsr()
            subgs += [dgll.create_block(('csc', (adj.indptr, adj.indices, [])))]
            prev_nodes_list = subgs[-1].srcnodes()
        subgs.reverse()
        subgs[0].srcdata['feat'] = g.ndata['feat'][prev_nodes_list]
        subgs[-1].dstdata['label'] = g.ndata['label'][batch_nodes]
        return prev_nodes_list.clone().detach(), batch_nodes, subgs


async def gradient_generator(model, gradient_buffer, con):
            size = float(torch.distributed.get_world_size())
            con.acquire()
            if gradient_buffer.full():
                con.wait()
            parameters_list = list(model.parameters())
            param_avg = []
            for param in parameters_list:
                torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                param_avg.append(param.grad.data/size)
            gradient_buffer.put(param_avg)
            con.notify()
            con.release()


async def gradient_consumer(model, gradient_buffer, con, opt):
            con.acquire()
            if gradient_buffer.empty():
                con.wait()
            param_avg = gradient_buffer.get()
            con.notify()
            con.release()
            for param, param_garad in zip(model.parameters(), param_avg):
                param.grad.data = param_garad
            opt.step()


def average_gradients(model):
    size = float(torch.distirubed.get_world_size())
    for param in model.parameters():
        torch.distirubed.all_reduce(param.grad.data, op=torch.distirubed.ReduceOp.SUM)
        param.grad.data /= size

def get_gradients(model):
    size = float(torch.distributed.get_world_size())
    return [param.grad.data/size for name, param in model.named_parameters()]



def run(proc_id, devices, producer, consumer,  BUFFER_SIZE = 4):
    # Initialize distributed training context.
    # divisor = 1024*1024*1024
    # print("GPU Stats in the beginning:", list(map(lambda x: x//divisor, torch.cuda.mem_get_info())) )
    BUFFER_SIZE = 4
    dev_id = devices[proc_id]
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12345')
    if torch.cuda.device_count() < 1:
        device = torch.device('cpu')
        torch.distributed.init_process_group(
            backend='gloo', init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device('cuda:' + str(dev_id))
        torch.distributed.init_process_group(
            backend='nccl', init_method=dist_init_method, world_size=len(devices), rank=proc_id)

    # Define training and validation dataloader, copied from the previous tutorial
    # but with one line of difference: use_ddp to enable distributed data parallel
    # data loading.
    sampler = FastGCNSampler(fanouts, graph)
    train_dataloader = dgll.dataloading.DataLoader(
        # The following arguments are specific to DataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        use_ddp=True,       # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=batch_size,    # Per-device batch size.
        # The effective batch size is this number times the number of GPUs.
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )
    valid_dataloader = dgll.dataloading.DataLoader(
        graph, valid_nids, sampler,
        device=device,
        use_ddp=False,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = Model(num_features, 128, num_classes).to(device)
    # Wrap the model with distributed data parallel module.
    if device == torch.device('cpu'):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # Define optimizer
    opt = torch.optim.Adam(model.parameters())


    condition = threading.Condition()
    gpu_queue = Queue(maxsize=BUFFER_SIZE)
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(sample_generator, gpu_queue, condition, train_dataloader, valid_dataloader, model, proc_id)
        executor.submit(sample_consumer, gpu_queue, condition, opt, model, BUFFER_SIZE)
    end = time.time()
    print("Time:", end - start)

graph.create_formats_()

if __name__ == '__main__':
    num_gpus = 1
    mp.spawn(run, args=(list(range(num_gpus)), sample_generator, sample_consumer,), nprocs=num_gpus)
