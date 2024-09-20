
import datetime, sys
import torch
import dgl
import numpy as np
import igraph as ig
import leidenalg as la
import os
import json

import pandas as pd
import time

from load_data import *



def store_tensor_to_disk(filename, tensor):
    np.savetxt('{}'.format(filename), tensor,fmt='%4i', delimiter=',')
# store_tensor_to_disk("Results/" + 'relable_graph_feat', graph.ndata['feat'][list(nid_mapping.keys())])
# print("dataset:", args.dataset)




def load_group_book(path, graph_name):
    with open(os.path.join(path, graph_name, graph_name)) as f:
        # load_group_book
        return json.load(f)

def relabel_groups(groups, path, graph_name, train_valid_test_ratio):
    """relable the group and store the group metadata to the disk"""
    relabel_groups = []
    con_id = 0
    con_id_mapping = {}
    groups_id_map_list = []
    for i, grp in enumerate(groups):
        relabel_grp = []
        for node in grp:
            con_id_mapping[node] = con_id
            relabel_grp.append(con_id)
            con_id += 1
        relabel_groups.append(relabel_grp)
        groups_id_map_list.append([int(relabel_grp[0]), int(relabel_grp[-1]+1)])
    return relabel_groups, con_id_mapping



def DGL_g_relabel(graph, nid_mapping, path, graph_name, cog_time):
    src_node_list = graph.edges()[0]
    dst_node_list = graph.edges()[1]
    relabel_graph.ndata['feat'], relabel_graph.ndata['label'] = reoder_DGL_gfeat_glabel(graph, nid_mapping)
    time_start= time.time()
    if os.path.exists(os.path.join(path, graph_name)):
        dgl.save_graphs(os.path.join(path, graph_name, "graph.bin"), relabel_graph)
    else:
        # create folder named as grapht_name
        os.makedirs(os.path.join(path, graph_name))
        dgl.save_graphs(os.path.join(path, graph_name, "graph.bin"), relabel_graph)
    cog_time["write data to disk"] = time.time() - time_start
    return relabel_graph, cog_time


def merge_groups(groups, max_length=5):
    merged_list = []
    merge_groups = []
    for group in groups:
        merge_groups.extend(group)
        if len(merge_groups) >= max_length:
            # Append the merged group to the result
            merged_list.append(merge_groups)
            # Reset the current group
            merge_groups = []
    # Add any remaining elements to the result
    if merge_groups:
        merged_list.append(merge_groups)
    return merged_list

def convert_bytes_to_MB(byte_size):
    # Convert bytes to megabytes
    mb_size = byte_size / (1024 ** 2)
    return mb_size

def generate_groups(graph, max_comm_size):
    # generate groups/communities from igrahp object
    return list(la.find_partition(graph, max_comm_size  = max_comm_size, partition_type=la.ModularityVertexPartition ))



def convert_bytes_to_GB(byte_size):
    # Convert megabytes to gigabytes
    gb_size = byte_size / 1024 ** 3
    return gb_size

def get_GPU_memory_in_Bytes(device):
    free_memory, total_memory = torch.cuda.mem_get_info(device= device)
    return free_memory, total_memory

def from_iGraph_to_DGL(graph):
    src_ids, dst_ids = zip(*igraph_graph.get_edgelist())
    return dgl.graph((src_ids, dst_ids))

def get_tensor_size_in_Bytes(tens):
    return tens.nelement() * tens.element_size()


def run_cog(args):
    cog_time = dict()
    cog_time["Dataset"] = args.dataset
    time_start= time.time()
    if args.dataset == "pubmed":
        path = "home/dke/dist/dataset/" + args.dataset
        graph = load_pubmed(path)
        node_labels = graph.ndata['label']
        total_nodes = graph.number_of_nodes()
        train_valid_test_ratio = [66, 10, 24]


        # train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=False).squeeze()
        # valid_nids = torch.nonzero(graph.ndata['val_mask'], as_tuple=False).squeeze()
        # test_nids = torch.nonzero(graph.ndata['test_mask'], as_tuple=False).squeeze()    # Test node IDs, not used in the tutorial though.

    elif args.dataset == "citeseer":
        path = "home/dke/dist/dataset/" + args.dataset
        graph = load_citeseer(path)
        node_labels = graph.ndata['label']
        total_nodes = graph.number_of_nodes()
        train_valid_test_ratio = [66, 10, 24]
        # train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=False).squeeze()
        # valid_nids = torch.nonzero(graph.ndata['val_mask'], as_tuple=False).squeeze()
        # test_nids = torch.nonzero(graph.ndata['test_mask'], as_tuple=False).squeeze()    # Test node IDs, not used in the tutorial though.


    elif args.dataset == "reddit":
        path = "home/dke/dist/dataset/" + args.dataset
        graph = load_reddit(path)
        node_labels = graph.ndata['label']
        total_nodes = graph.number_of_nodes()
        train_valid_test_ratio = [66, 10, 24]
        # train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=False).squeeze()
        # valid_nids = torch.nonzero(graph.ndata['val_mask'], as_tuple=False).squeeze()
        # test_nids = torch.nonzero(graph.ndata['test_mask'], as_tuple=False).squeeze()    # Test node IDs, not used in the tutorial though.

    elif args.dataset == "arxiv":
        path = "home/dke/dist/dataset/" + args.dataset
        dataset = load_arxiv(path)
        graph, node_labels = dataset[0]
        # Add reverse edges since ogbn-arxiv is unidirectional.
        graph = dgl.add_reverse_edges(graph)
        graph.ndata['label'] = node_labels[:, 0]
        total_nodes = graph.number_of_nodes()
        train_valid_test_ratio = [54, 18, 28]
        # idx_split = dataset.get_idx_split()
        # total = graph.number_of_nodes()
        # train_valid_test_ratio = [len(idx_split['train'])*100/total,len(idx_split['valid'])*100/total,len(idx_split['test'])*100/total]



    elif args.dataset == "products":
        path = "home/dke/dist/dataset/" + args.dataset
        dataset = load_products(path)
        graph, node_labels = dataset[0]
        # Add reverse edges since ogbn-arxiv is unidirectional.
        graph = dgl.add_reverse_edges(graph)
        # print(graph.edges(), graph.edges()[0].shape)
        # print(graph.ndata['feat'], graph.ndata['feat'].shape)
        graph.ndata['label'] = node_labels[:, 0]
        total_nodes = graph.number_of_nodes()
        train_valid_test_ratio = [90, 2, 8]
        # idx_split = dataset.get_idx_split()
        # train_nids = idx_split['train']
        # valid_nids = idx_split['valid']
        # test_nids = idx_split['test']    # Test node IDs, not used in the tutorial though.

    elif args.dataset == "cora":
        path = "home/dke/dist/dataset/" + args.dataset
        dataset = load_cora(path)
        graph = dataset[0]
        node_labels = graph.ndata['label']
        total_nodes = graph.number_of_nodes()
        train_valid_test_ratio = [54, 18, 28]
        # train_mask = graph.ndata['train_mask']
        # test_mask = graph.ndata['test_mask']
        # valid_mask = graph.ndata['valid_mask']
        # train_nids = torch.nonzero(graph.ndata['train_mask'], as_tuple=False).squeeze()
        # valid_nids = torch.nonzero(graph.ndata['val_mask'], as_tuple=False).squeeze()
        # test_nids = torch.nonzero(graph.ndata['test_mask'], as_tuple=False).squeeze()    # Test node IDs, not used in the tutorial though.
    else:
        print("WRONG DATA SET SELECTED: Select from pubmed/reddit/arxiv/prdoucts")

    graph = dgl.add_self_loop(graph)


    node_features = graph.ndata['feat']
    num_features = node_features.shape[1]

    num_classes = (node_labels.max() + 1).item()
    path = args.output_path_dataset
    cog_time["data_load"] = time.time() - time_start
    # print("path", path)

    time_start= time.time()
    single_label_size = get_tensor_size_in_Bytes(graph.ndata['label'][0])
    single_feat_size = get_tensor_size_in_Bytes(graph.ndata['feat'][0])

    free_memory, total_memory = get_GPU_memory_in_Bytes('cuda:0')

    size_of_group = int(free_memory /(single_feat_size+ single_label_size + 1000))

    size_train_nids = int((train_valid_test_ratio[0]/100) * total_nodes)
    if size_of_group > size_train_nids:
        size_of_group = size_train_nids

    cog_time["Find maximum size of a community"] = time.time() - time_start

    time_start= time.time()
    # print("converstion to iGraph started")
    graph_ig = from_DGLg_to_iGraph(graph)
    # print("iGraph Conversion completed")

    # print("Gropu Identification started")
    groups = generate_groups(graph_ig, size_of_group)
    cog_time["Generate communities"] = time.time() - time_start
    # print("Group Identification completed")
    del graph_ig
    time_start= time.time()
    # print("Merge Grouping started")
    groups = merge_groups(groups, args.batch_size)
    cog_time["Merge communities"] = time.time() - time_start
    # print("Merge Grouping Ended")
    # rel_groups, con_id_mapping = relabel_groups(groups, path, graph_name)
    # print("Group relableing started")
    time_start= time.time()
    rel_groups, con_id_mapping = relabel_groups(groups, path, args.dataset, train_valid_test_ratio)
    cog_time["Relabel communities"] = time.time() - time_start
    # print("Group relableing finished")
    time_start= time.time()
    # print("DGL graph relabelling started")
    graph, cog_time = DGL_g_relabel(graph, con_id_mapping, path, args.dataset, cog_time)
    cog_time["Relabel graph"] = time.time() - time_start

    cog_time["Total time"] = cog_time["data_load"] + cog_time["Find maximum size of a community"] + cog_time["Generate communities"] + \
                             cog_time["Merge communities"] +  cog_time["Relabel communities"] + cog_time["Relabel graph"] + cog_time["write data to disk"]


    # print total time
    print("tota_time:", cog_time["Total time"])

    #save time information to the disk
    cog_time_df = pd.DataFrame(cog_time,  index = [0]).transpose()
    cog_time_df.to_csv('{}/{}.csv'.format(os.path.join(path, args.dataset), args.dataset))
    # print time and dataset information
    print("cog time: ", cog_time_df)
    print("Finished: ", args.dataset)