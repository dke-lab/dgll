import os
os.environ["DGLBACKEND"] = "pytorch"

import sklearn.metrics
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.utils import load_graphs
import mylog

mlog = mylog.get_logger()
from CommGNNModel import GCN, GraphSAGE
from utils import *
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def evaluate(model, node_features, node_labels, device, vaid_dataloader):
    model.eval()
    predictions = []
    labels = []
    with tqdm.tqdm(vaid_dataloader) as tq:
        for step, (input_nodes, seeds, subg_feat, subg_labels, subgs) in enumerate(tq):
            labels.append(subg_labels.cpu().numpy())
            with torch.no_grad():
                pred = model(subgs, subg_feat)
                predictions.append(pred.argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        valid_f1 = sklearn.metrics.f1_score(labels, predictions,  average="micro")
    return valid_f1

def run_train(args):

    result_pkl = dict()
    result_pkl["args"] = args
    path = args.path + args.dataset + '/'

    dataset, _ = load_graphs(path + "graph.bin")


    graph = dataset[0]
    node_features = graph.ndata["feat"]
    del graph.ndata["feat"]
    in_feats = node_features.shape[1]
    node_labels = graph.ndata['label']
    del graph.ndata['label']
    n_classes = (node_labels.max() + 1).item()



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

    cached_nodeIDs = cached_nodeIDs[:num_elements]

    cachedData = CacheNodes(node_features, cached_nodeIDs, device= device)


    train_vlaid_test_id = 0
    train_vlaid_test_node_ranges = torch.tensor(lcb['train_vlaid_test_node_ranges'])[train_vlaid_test_id]
    train_vlaid_test_comm_ranges = list_to_tensor(lcb['train_vlaid_test_comm_ranges'][train_vlaid_test_id])
    communities = get_community_map_pt(lcb)
    train_dataloader = CommBNeighborSampler(graph, communities, samp_num_list, args.batch_size, train_vlaid_test_comm_ranges, train_vlaid_test_node_ranges, node_features, node_labels,cached_nodeIDs, cachedData, device)
    train_vlaid_test_id = 1
    train_vlaid_test_node_ranges = torch.tensor(lcb['train_vlaid_test_node_ranges'])[train_vlaid_test_id]
    train_vlaid_test_comm_ranges = list_to_tensor(lcb['train_vlaid_test_comm_ranges'][train_vlaid_test_id])
    valid_dataloader = CommBNeighborSampler(graph, communities, samp_num_list, args.batch_size, train_vlaid_test_comm_ranges, train_vlaid_test_node_ranges, node_features, node_labels,cached_nodeIDs, cachedData, device)
    train_vlaid_test_id = 2
    train_vlaid_test_node_ranges = torch.tensor(lcb['train_vlaid_test_node_ranges'])[train_vlaid_test_id]
    train_vlaid_test_comm_ranges = list_to_tensor(lcb['train_vlaid_test_comm_ranges'][train_vlaid_test_id])
    test_dataloader = CommBNeighborSampler(graph, communities, samp_num_list, args.batch_size, train_vlaid_test_comm_ranges, train_vlaid_test_node_ranges, node_features, node_labels,cached_nodeIDs, cachedData, device)

    best_test = -1
    cnt = 0
    valid_acc, best_eval = -1, -1
    for niter in range(args.n_trial):
        if args.Model == "GCN":
            model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
        else:
            model = GraphSAGE(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
        model = model.to(device)
        # Define optimizer
        opt =torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.n_epochs):
            model.train()
            train_losses = []

            for _iter in range(args.o_iters):
                with tqdm.tqdm(train_dataloader, leave=False) as tq:
                    for step, (input_nodes, seeds, subg_feat, subg_labels, subgs) in enumerate(tq):
                        subgs = [subg.int().to(device) for subg in subgs]

                        batch_pred = model(subgs, subg_feat)

                        loss = loss_fcn(batch_pred, subg_labels)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        train_losses += [loss.detach().tolist()]



            mlog('Epoch {:05d} | Step {:05d} | Loss {:.4f}'.format(
                epoch, step, loss.item()))

            valid_acc = evaluate(model, node_features, node_labels, device, valid_dataloader)
            test_f1 = evaluate(model, node_features, node_labels, device, test_dataloader)

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

        mlog(f'BestVal Test F1-mic: {best_eval:.4f}')
    print("Finished!")





