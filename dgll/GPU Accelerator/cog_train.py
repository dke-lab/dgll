import os
import json
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import time
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data.utils import load_graphs
import argparse
import mylog
import datetime, sys
import pickle as pkl
mlog = mylog.get_logger()
from CommGNNmodel import *
from utils import *
from utils import *
from tensors_in_memory import *
from pynvml import *

import torch.autograd.profiler as profiler
import gc
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        assert n_layers > 1
        # input layer
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layer
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(dglnn.GraphConv(n_hidden, n_classes))

        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        if blocks[0].is_block:
            # normal gcn
            h = x
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.dropout(h)
        else:
            # transductive mos
            subg, block = blocks
            h = x
            for layer in self.layers[:-1]:
                # print(50*"++")
                # print(h.shape)
                # print(subg)
                # print(50*"++")
                h = layer(subg, h)

                h = self.dropout(h)

            # slice out the input nodes for block
            internal_input_nids = block.ndata[dgl.NID]['_N'].to('cuda')
            h = self.layers[-1](block, h[internal_input_nids])
        return h




class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        if isinstance(blocks, dgl.DGLGraph):
            # mos inductive
            assert isinstance(blocks, dgl.DGLGraph)
            h = x
            for l, layer in enumerate(self.layers):
                h = layer(blocks, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
        else:
            assert isinstance(blocks, list)
            if blocks[0].is_block:
                # normal sage
                h = x
                for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                    h = layer(block, h)
                    if l != len(self.layers) - 1:
                        h = self.activation(h)
                        h = self.dropout(h)
            else:
                # mos transductive
                subg, block = blocks
                h = x
                for layer in self.layers[:-1]:
                    h = layer(subg, h)
                    h = self.activation(h)
                    h = self.dropout(h)

                # slice out the input nodes for block
                internal_input_nids = block.ndata[dgl.NID]['_N'].to('cuda')
                h = self.layers[-1](block, h[internal_input_nids])

        return h






def evaluate(model, graph, communities, samp_num_list, batch_size, train_vlaid_test_comm_ranges, train_vlaid_test_node_ranges, node_features, node_labels,cached_nodeIDs, cachedData, device):
    model.eval()
    predictions = []
    labels = []
    dataloader = CommBNeighborSampler(graph, communities, samp_num_list, batch_size, train_vlaid_test_comm_ranges, train_vlaid_test_node_ranges, node_features, node_labels,cached_nodeIDs, cachedData, device)

    with tqdm.tqdm(dataloader) as tq:
        for step, (input_nodes, seeds, subg_feat, subg_labels, subgs) in enumerate(tq):

            # subg_feat, subg_labels = load_subteatures(node_features, node_labels, [input_nodes[0], input_nodes[-1]], seeds = [seeds[0], seeds[-1]], device=device)
            # feature copy from CPU to GPU takes place here
            # print(subgs, subg_feat, subg_labels)
            subgs = [subg.int().to(device) for subg in subgs]
            # print(50*"++++")
            # print("feat", subg_feat.shape)
            # print("subgs", subgs)
            # print(50*"++++")
            labels.append(subg_labels.cpu().numpy())
            with torch.no_grad():
                pred = model(subgs, subg_feat)
                predictions.append(pred.argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        valid_f1 = sklearn.metrics.f1_score(labels, predictions,  average="micro")
        # accuracy = sklearn.metrics.accuracy_score(labels, predictions)

        # if best_accuracy < accuracy:
        #     best_accuracy = accuracy

    # if isinstance(nids, list):
    #     accs = [compute_acc(pred[nid], labels[nid], multilabel) for nid in nids]
    # else:
    #     assert isinstance(nids, torch.Tensor)
    #     accs = compute_acc(pred[nids], labels[nids], multilabel)
    return valid_f1

def run_train(args):
    write_file = "w"
    original_stdout = sys.stdout
    result_pkl = dict()
    result_pkl["args"] = args
    best_model_idx = str(datetime.datetime.now()).replace(' ', '_').replace(':', '.')
    path = args.path + args.dataset + '/'

    dataset, _ = load_graphs(path + "graph.bin")


    graph = dataset[0]
    filename = "main_{}_{}_{}".format(args.dataset, args.n_layers, best_model_idx)
    # node_features = graph.ndata["feat"]
    # del graph.ndata["feat"]
    node_features = graph.ndata.pop('feat')
    in_feats = node_features.shape[1]
    node_labels = graph.ndata.pop('label')
    # node_labels = graph.ndata['label']
    # del graph.ndata['label']
    print("graph node", graph.num_nodes())
    n_classes = (node_labels.max() + 1).item()
    torch.cuda.empty_cache()




    lcb = load_community_book(path, args.dataset)


    train_proportion = lcb['train_valid_test_ratio'][0] / 100
    val_proportion = lcb['train_valid_test_ratio'][1] / 100
    test_proportion = lcb['train_valid_test_ratio'][2] / 100
    num_nodes = graph.num_nodes()

    # Calculate the sizes for each set
    train_size = int(train_proportion * num_nodes)
    val_size = int(val_proportion * num_nodes)
    test_size = num_nodes - train_size - val_size

    # Split the node IDs
    node_ids = np.arange(num_nodes)
    train_nids = node_ids[:train_size]
    val_nids = node_ids[train_size:train_size + val_size]
    test_nids = node_ids[train_size + val_size:]

    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_nids] = True
    val_mask[val_nids] = True
    test_mask[test_nids] = True
    samp_num_list = args.fanouts
    if args.gpu >= 0:
        device_id = 0
        device = torch.device('cuda:%d' % device_id)
    else:
        device = torch.device('cpu')
    multilabel = False
    if multilabel:
        loss_fcn = nn.BCEWithLogitsLoss()
    else:
        loss_fcn = nn.CrossEntropyLoss()
    cached_nodeIDs = load_top_nodes_cache(path, args.dataset)
    # Calculate the number of elements
    num_elements = int(args.cached_nPercent / 100 * cached_nodeIDs.numel())

    # Slice the tensor to take the first n% elements
    cached_nodeIDs = cached_nodeIDs[:num_elements]
    # memory_in_bytes = cached_nodeIDs.element_size() * cached_nodeIDs.numel()
    #
    # # Convert to megabytes (MB) or gigabytes (GB) if necessary
    # memory_in_MB = memory_in_bytes / (1024 ** 2)
    # memory_in_GB = memory_in_bytes / (1024 ** 3)
    #
    # print(f"Total memory: {memory_in_bytes} bytes")
    # print(f"Total memory: {memory_in_MB:.2f} MB")
    # print(f"Total memory: {memory_in_GB:.2f} GB")
    cachedData = CacheNodes(node_features, cached_nodeIDs, device= device)


    train_vlaid_test_id = 0
    train_vlaid_test_node_ranges = torch.tensor(lcb['train_vlaid_test_node_ranges'])[train_vlaid_test_id]
    train_vlaid_test_comm_ranges = list_to_tensor(lcb['train_vlaid_test_comm_ranges'][train_vlaid_test_id])
    communities = get_community_map_pt(lcb)

    train_dataloader = CommBNeighborSampler(graph, communities, samp_num_list, args.batch_size, train_vlaid_test_comm_ranges, train_vlaid_test_node_ranges, node_features, node_labels,cached_nodeIDs, cachedData, device)

    total_time_all = []
    batch_time_all = []
    test_f1_all   = []
    epoch_time_all = []
    epoch_time_all_cum_sum = []
    epoch_num = []
    valid_loss_all = []
    sample_batch_time_all = []
    valid_f1_all = []
    best_test = -1
    cnt = 0
    valid_acc, best_eval = -1, -1
    # create main directory: "Results/args.dataset"
    dir_name = '{}/{}'.format('Results', args.dataset)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    dir_name = '{}/{}/{}'.format('Results', args.dataset, 'model')
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    for niter in range(args.n_trial):
        times = []
        epoch_time = []
        valid_f1_single_iter = []
        sample_batch_time_single_iter = []
        valid_loss_single_iter = []
        single_epoch_time = []
        single_epoch_time_cum_sum = []
        if args.Model == "GCN":
            model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
        else:
            model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
        model = model.to(device)
        # Define optimizer
        opt =torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.n_epochs):
            model.train()
            temp_epoch_time = []
            train_losses = []

            for _iter in range(args.o_iters):

                with tqdm.tqdm(train_dataloader, leave=False) as tq:


                    # input_nodes,seeds, batch_node_feats, batch_node_labels, [subg, block]
                    single_batch_time = time.time()
                    for step, (input_nodes, seeds, subg_feat, subg_labels, subgs) in enumerate(tq):

                        tic = time.time()
                        single_batch_time = tic - single_batch_time

                        batch_pred = model(subgs, subg_feat)

                        loss = loss_fcn(batch_pred, subg_labels)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        tec = time.time()
                        times += [tec - tic]
                        temp_epoch_time += [tec - tic]
                        train_losses += [loss.detach().tolist()]
                        sample_batch_time_single_iter.append(single_batch_time)
                        single_batch_time = time.time()

                        # dump_tensors()
            single_epoch_time += [np.sum(temp_epoch_time)]
            single_epoch_time_cum_sum += [np.sum(times)]
            mlog('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, step, loss.item(), single_epoch_time[-1]))
            torch.cuda.empty_cache()

            train_vlaid_test_id = 1
            train_vlaid_test_node_ranges = torch.tensor(lcb['train_vlaid_test_node_ranges'])[train_vlaid_test_id]
            train_vlaid_test_comm_ranges = list_to_tensor(lcb['train_vlaid_test_comm_ranges'][train_vlaid_test_id])
            valid_acc = evaluate(model, graph, communities, samp_num_list, args.batch_size, train_vlaid_test_comm_ranges, train_vlaid_test_node_ranges, node_features, node_labels,cached_nodeIDs, cachedData, device)
            torch.cuda.empty_cache()
            train_vlaid_test_id = 2
            train_vlaid_test_node_ranges = torch.tensor(lcb['train_vlaid_test_node_ranges'])[train_vlaid_test_id]
            train_vlaid_test_comm_ranges = list_to_tensor(lcb['train_vlaid_test_comm_ranges'][train_vlaid_test_id])
            valid_f1_single_iter.append(valid_acc)
            test_f1 = evaluate(model, graph, communities, samp_num_list, args.batch_size, train_vlaid_test_comm_ranges, train_vlaid_test_node_ranges, node_features, node_labels,cached_nodeIDs, cachedData, device)

            if valid_acc > best_eval + 1e-2:
                best_eval = valid_acc
                cnt = 0
            else:
                cnt += 1
            if cnt == args.n_stops:
                break
            if test_f1 > best_test:
                best_test = test_f1


                # torch.save(model, '{}/{}/{}/best_model_{}_{}.pt'.format('Results',args.dataset, 'model' , args.n_layers, best_model_idx))


            mlog('Eval Acc: {:.4f}, Best Acc: {:.4f}'.format(valid_acc, best_eval))

        batch_time_all += [times]
        total_time_all += [np.sum(times)]
        test_f1_all  += [test_f1]
        epoch_num += [epoch]
        epoch_time_all += [single_epoch_time]
        epoch_time_all_cum_sum += [single_epoch_time_cum_sum]
        valid_f1_all += [valid_f1_single_iter]
        valid_loss_all += [valid_loss_single_iter]
        sample_batch_time_all += [sample_batch_time_single_iter]
        mlog(f'BestVal Test F1-mic: {best_eval:.4f}')
    if args.record_f1:
        text_filename = filename + '.txt'
        result_pkl[args.Model] = save_result_data(args, text_filename, total_time_all, samp_num_list,
                                                      valid_f1_all, valid_loss_all, test_f1_all, epoch_num, epoch_time_all, write_file,
                                                      original_stdout, sample_batch_time_all, batch_time_all, epoch_time_all_cum_sum)
        print(args.Model,args.dataset, "\'s information recorded")
    # record .pkl
    if args.record_f1:
        # add director name
        with open('Results/{}/{}/{}.pkl'.format(args.dataset, 'result', filename),'wb') as f:
            pkl.dump(result_pkl, f)
        print("All information is recorded")

