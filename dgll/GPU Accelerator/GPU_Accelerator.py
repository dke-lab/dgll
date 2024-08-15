import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch
import numpy as np
from GNN_models import GCN_Model, GraphSAGE_Model
from GNN_models import Custom_GNN_Model
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import sklearn.metrics
import time
import torch.multiprocessing as mp
import concurrent.futures
import threading
from queue import Queue
import asyncio
from contextlib import nullcontext

from GPU_Accelerator_arguments import args

dataset = DglNodePropPredDataset(args.dataset)
graph, node_labels = dataset[0]
# Add reverse edges
graph = dgl.add_reverse_edges(graph)
graph.ndata['label'] = node_labels[:, 0]
node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()

idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']

def sample_generator(gpu_queue, condition, train_dataloader, valid_dataloader, model, proc_id):
    d_stream = torch.cuda.Stream()
    best_accuracy = 0
    # generate items
    start = time.time()
    for epoch in range(args.epoch):
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                with torch.cuda.stream(d_stream):
                    with condition:
                        condition.acquire()
                        if gpu_queue.full():
                            condition.wait()
                        gpu_queue.put([mfgs, mfgs[0].srcdata['feat'], mfgs[-1].dstdata['label'], step])
                        condition.notify()
                        condition.release()
        if proc_id == 0:
            model.eval()
            predictions = []
            labels = []
            with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
                for input_nodes, output_nodes, mfgs in tq:
                    inputs = mfgs[0].srcdata['feat']
                    labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
    end = time.time()
    print(50*"*")
    print("Accuracy:", best_accuracy*100, "\n")
    print("Time:", end - start)
    with condition:
        condition.acquire()
        gpu_queue.put(None)
        condition.notify()
        condition.release()

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

async def gradient_generation_consumption(g_stream,model, gradient_buffer, con, opt):
    with torch.cuda.stream(g_stream):
        grad_gen_task = asyncio.create_task(gradient_generator(model, gradient_buffer, con))
    grad_cons_task = asyncio.create_task(gradient_consumer(model, gradient_buffer, con, opt))
    await grad_gen_task
    await grad_cons_task

def sample_consumer(gpu_queue, condition, opt, model, BUFFER_SIZE = 4):
    con = threading.Condition()
    gradient_buffer = Queue(maxsize= BUFFER_SIZE)
    c_stream = torch.cuda.Stream()
    m_context = model.no_sync
    with torch.cuda.stream(c_stream):
        g_stream = torch.cuda.Stream()
        model.train()
        while True:
            with condition:
                condition.acquire()
                if gpu_queue.empty():
                    condition.wait()
                input_mfg_feat_label = gpu_queue.get()
                condition.notify()
                condition.release()

            if input_mfg_feat_label == None:
                break
            with m_context():
                opt.zero_grad()
                predictions = model(input_mfg_feat_label[0], input_mfg_feat_label[1])
                loss = F.cross_entropy(predictions, input_mfg_feat_label[2])
                loss.backward()
                asyncio.run(gradient_generation_consumption(g_stream,model, gradient_buffer, con, opt))
                accuracy = sklearn.metrics.accuracy_score(input_mfg_feat_label[2].cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())
                print(f'>Accuracy', accuracy)




def run(proc_id, devices, args):

    # print("GPU Stats in the beginning:", list(map(lambda x: x//divisor, torch.cuda.mem_get_info())) )
    BUFFER_SIZE = args.buffer_size
    dev_id = devices[proc_id]
    # Initialize distributed training context
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

    model = globals().get(args.GNN_Model)(num_features, 128, num_classes).to(device)
    # Define training and validation dataloader
    sampler = dgl.dataloading.NeighborSampler(args.fanout)
    train_dataloader = dgl.dataloading.DataLoader(
        # The following arguments are specific to DataLoader.
        graph,              # The graph
        train_nids,         # The node IDs to iterate over in minibatches
        sampler,            # The neighbor layer_dependent_sampler
        device=device,      # Put the sampled MFGs on CPU or GPU
        use_ddp=True,       # Make it work with distributed data parallel
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=args.batch_size,    # Per-device batch size.
        # The effective batch size is this number times the number of GPUs.
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of layer_dependent_sampler processes
    )
    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler,
        device=device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

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
    mp.spawn(run, args=(list(range(args.num_gpus)), args,), nprocs=args.num_gpus)
