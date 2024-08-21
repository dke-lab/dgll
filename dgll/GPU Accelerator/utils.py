import os
import numpy as np
import sys
import dgl
from dgl.data import PubmedGraphDataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset
from dgl.data import RedditDataset
import scipy.sparse as sp
import torch
from ogb.nodeproppred import DglNodePropPredDataset

def matrix_row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx



def load_reddit(root):
    data = RedditDataset(raw_dir=root, self_loop=True)
    g = data[0]
    return g



def load_Pubmed(root):
    data = PubmedGraphDataset(raw_dir=root, reverse_edge=True)
    g = data[0]
    return g


# add loss record
def record_result_new(args, text_filename, total_time_all, samp_num_list, valid_f1_all, valid_loss_all,
                      test_f1_all, epoch_num, epoch_time_all, write_file,
                      sample_method, original_stdout, batch_time_all):
    dir_name = 'Results/{}/{}'.format(args.dataset, 'result')
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    with open('{}/{}/{}/{}'.format("Results", args.dataset, 'result', text_filename), 'a') as f:
        sys.stdout = f # Change the standard output to the file we created.
        np.set_printoptions(precision=5)
        print(args)
        print("{}_repeat {} times".format(args.dataset, args.n_trial))
        print("batch_size: {}, base_sample_num: {}, layers: {}".format(args.batch_size,
                                                                       args.n_samp, args.n_layers))
        print("samp_num_each_layer", samp_num_list)
        print("-" * 20)

        print("Sampler method: ", sample_method)
        f1_mean, f1_mean_sd = np.average(test_f1_all), np.std(test_f1_all) / np.sqrt(args.n_trial)
        # epoch_mean, epoch_sd = np.mean(epoch_num), np.std(epoch_num) / np.sqrt(args.n_trial)
        # epoch_time_all = sum(epoch_time_all, [])
        # batch_time_all = sum(batch_time_all, [])

        epoch_mean, epoch_sd = np.mean(epoch_time_all), np.std(epoch_time_all) / np.sqrt(args.n_trial)
        batch_mean, batch_sd = np.mean(batch_time_all), np.std(batch_time_all) / np.sqrt(args.n_trial)
        if args.dataset == "proteins": # 'proteins'
            print("roc-auc.mean", "roc-auc.se")
        elif args.dataset in ["reddit", "cora", "citeseer", "pubmed"]: # ['cora', 'citeseer', 'pubmed', 'reddit']
            print("f1.mean", "f1.se")
        else: # 'arxiv', 'products'
            print("accuracy.mean", "accuracy.se")
        print(np.array([f1_mean, f1_mean_sd]))
        # print(np.array([f1_mean - 1.96 * f1_mean_sd, f1_mean + 1.96 * f1_mean_sd]))
        print("Epoch time: epoch_mean, epoch_mean_sd")
        print([epoch_mean, epoch_sd])
        print("Batch time: batch_mean, batch_mean_sd")
        print([batch_mean, batch_sd])
        print("training time: mean, mean's sd")
        print(np.array([np.mean(total_time_all), np.std(total_time_all) / np.sqrt(args.n_trial)]))
        print("\n")
        print("_" * 20)

    sys.stdout = original_stdout # Reset the standard output to its original value

    # record the data to .pkl

    result_dict = dict()
    result_dict["args"] = args
    if args.dataset == "proteins": # 'proteins'
        result_dict["test_roc-auc"] = test_f1_all
        result_dict["roc-auc mean, mean sd"] = [f1_mean, f1_mean_sd]
        result_dict["valid_roc-auc_all"] = valid_f1_all
    elif args.dataset in ["reddit", "cora", "citeseer", "pubmed"]: # ['cora', 'citeseer', 'pubmed', 'reddit']
        result_dict["test_f1"] = test_f1_all
        result_dict["f1 mean, mean sd"] = [f1_mean, f1_mean_sd]
        result_dict["valid_f1_all"] = valid_f1_all
    else: # 'arxiv', 'products'
        result_dict["test_accuracy"] = test_f1_all
        result_dict["accuracy mean, mean sd"] = [f1_mean, f1_mean_sd]
        result_dict["valid_accuracy_all"] = valid_f1_all

    result_dict["time"] = total_time_all
    result_dict["avg time, avg std"] = [np.mean(total_time_all), np.std(total_time_all) / np.sqrt(args.n_trial)]
    result_dict["epoch_time_all"] = epoch_time_all
    result_dict["epoch_num"] = epoch_num
    result_dict["epoch mean, meand= sd"] = [epoch_mean, epoch_sd]
    result_dict["layer_samp_num"] = samp_num_list
    return result_dict

def remove_elements(lst, n):
    if 2 * n > len(lst):
        n = int(n/2)
    if 2 * n > len(lst):
        n = int(n/2)
    if n < len(lst):
        return lst[n:-n]
    if n < 0:
        raise ValueError("n should be a non-negative integer.")
    if n > len(lst):
        raise ValueError("n is too large for the given list.")
    else:
        raise ValueError("n is too large for the given list.")


def load_data(args):
    if args.dataset == "pubmed":
        path = "dataset/" + args.dataset
        graph = load_Pubmed(path)
        node_labels = graph.ndata['label']
        train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=False).squeeze()
        valid_nids = torch.nonzero(graph.ndata['val_mask'], as_tuple=False).squeeze()
        test_nids = torch.nonzero(graph.ndata['test_mask'], as_tuple=False).squeeze()    # Test node IDs, not used in the tutorial though.

    elif args.dataset == "cora":
        dataset = CoraGraphDataset()
        graph = dataset[0]
        node_labels = graph.ndata['label']
        train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=False).squeeze()
        valid_nids = torch.nonzero(graph.ndata['val_mask'], as_tuple=False).squeeze()
        test_nids = torch.nonzero(graph.ndata['test_mask'], as_tuple=False).squeeze()    # Test node IDs, not used in the tutorial though.

    elif args.dataset == "citeseer":
        dataset = CiteseerGraphDataset()
        graph = dataset[0]
        node_labels = graph.ndata['label']
        train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=False).squeeze()
        valid_nids = torch.nonzero(graph.ndata['val_mask'], as_tuple=False).squeeze()
        test_nids = torch.nonzero(graph.ndata['test_mask'], as_tuple=False).squeeze()    # Test node IDs, not used in the tutorial though.

    elif args.dataset == "reddit":
        path = "dataset/" + args.dataset
        graph_name = 'reddit'
        graph = load_reddit(path)
        node_labels = graph.ndata['label']
        train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=False).squeeze()
        valid_nids = torch.nonzero(graph.ndata['val_mask'], as_tuple=False).squeeze()
        test_nids = torch.nonzero(graph.ndata['test_mask'], as_tuple=False).squeeze()    # Test node IDs, not used in the tutorial though.

    elif args.dataset == "arxiv":
        dataset = DglNodePropPredDataset('ogbn-arxiv')
        graph, node_labels = dataset[0]
        # Add reverse edges since ogbn-arxiv is unidirectional.
        graph = dgl.add_reverse_edges(graph)
        # print(graph.edges(), graph.edges()[0].shape)
        # print(graph.ndata['feat'], graph.ndata['feat'].shape)
        graph.ndata['label'] = node_labels[:, 0]
        idx_split = dataset.get_idx_split()
        train_nids = idx_split['train']
        valid_nids = idx_split['valid']
        test_nids = idx_split['test']    # Test node IDs, not used in the tutorial though.

    elif args.dataset == "products":
        dataset = DglNodePropPredDataset('ogbn-products')
        graph, node_labels = dataset[0]
        # Add reverse edges since ogbn-arxiv is unidirectional.
        graph = dgl.add_reverse_edges(graph)
        # print(graph.edges(), graph.edges()[0].shape)
        # print(graph.ndata['feat'], graph.ndata['feat'].shape)
        graph.ndata['label'] = node_labels[:, 0]
        idx_split = dataset.get_idx_split()
        train_nids = idx_split['train']
        valid_nids = idx_split['valid']
        test_nids = idx_split['test']    # Test node IDs, not used in the tutorial though.


    elif args.dataset == "proteins":

        dataset = DglNodePropPredDataset('ogbn-proteins')
        graph, node_labels = dataset[0]

        # print(graph.edges(), graph.edges()[0].shape)
        # print(graph.ndata['feat'], graph.ndata['feat'].shape)
        graph.ndata['label'] = node_labels[:, 0]
        graph.ndata['feat'] = torch.zeros(graph.num_nodes(), graph.edata['feat'].shape[1])
        # Add reverse edges since ogbn-arxiv is unidirectional.
        graph = dgl.add_reverse_edges(graph)
        idx_split = dataset.get_idx_split()
        train_nids = idx_split['train']
        valid_nids = idx_split['valid']
        test_nids = idx_split['test']    # Test node IDs, not used in the tutorial though.


    else:
        print(10*"WRONG DATA SET SELECTED: Select from cora/citeseer/pubmed/proteins/arxiv/reddit/prdoucts")
    return graph, node_labels, train_nids, valid_nids, test_nids


def estWRS_weights(p, m):
    n = len(p)
    wrs_index = np.random.choice(n, m, False, p)

    weights = np.zeros(m)
    p_sum = 0

    for i in range(m):

        alpha = n / (i + 1) / (n - i)
        weights[i] = (1-p_sum) / p[wrs_index[i]] * alpha
        weights[:i] = weights[:i] * (1 - alpha) + alpha
        p_sum += p[wrs_index[i]]

    return wrs_index, weights

def normalize_lap(adj):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj