import os
import shutil
from collections import ChainMap, deque
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from joblib import Parallel, delayed
import torch
import warnings
import sys
try:
    from utils import partition_dict, preprocess_nxgraph, get_vertices, create_alias_table, cost, cost_max, \
        compute_dtw_dist, convert_dtw_struc_dist, operator_hadamard, custom_formatwarning
    from biasedRandomWalk import BiasedWalker
    from randomWalkEmbedding import RandomWalkEmbedding
except ModuleNotFoundError:
    from .utils import partition_dict, preprocess_nxgraph, get_vertices, create_alias_table, cost, cost_max, \
        compute_dtw_dist, convert_dtw_struc_dist, operator_hadamard, custom_formatwarning
    from .biasedRandomWalk import BiasedWalker
    from .randomWalkEmbedding import RandomWalkEmbedding

class Struc2Vec(RandomWalkEmbedding):
    # Constructor
    def __init__(self, graph = None, walkLength = 0, embedDim = 0, numbOfWalksPerVertex = 0, \
                 windowSize = 0, lr= 0, verbose=0, stay_prob=0, opt1_reduce_len=True, opt2_reduce_sim_calc=True, \
                 opt3_num_layers=None, temp_path='./temp_struc2vec/', reuse=False):
        if graph is None:
            warnings.warn("Provide a graph: {}".format(graph))
            sys.exit()

        super(Struc2Vec, self).__init__(graph, walkLength, embedDim, numbOfWalksPerVertex,  windowSize, lr)
        # validate arguments
        if stay_prob == 0:
            self.stay_prob = 0.3
            warnings.warn("Set stay prob. to default: {}".format(self.stay_prob))
        else:
            self.stay_prob = stay_prob
        self.verbose = verbose
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)

        self.adj_list, self.nodeEncoder = self.graph_to_adjList(graph)
        self.idx = list(range(len(self.nodeEncoder.classes_)))

        self.opt1_reduce_len = opt1_reduce_len
        self.opt2_reduce_sim_calc = opt2_reduce_sim_calc
        self.opt3_num_layers = opt3_num_layers

        self.resue = reuse
        self.temp_path = temp_path

        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)
        if not reuse:
            shutil.rmtree(self.temp_path)
            os.mkdir(self.temp_path)

        workers = 1
        self.create_context_graph(self.opt3_num_layers, workers, verbose)
        self.prepare_biased_walk()
        self.walker = BiasedWalker(self.nodeEncoder.classes_, self.temp_path)
        self._embeddings = {}

    # Walks generation
    def RandomWalk(self, startNode, walkLength):
        startNode = int(self.nodeEncoder.transform([startNode]))
        walk = self.walker. simulate_walks(
            startNode, walkLength, self.stay_prob, 1, self.verbose)
        return list(self.nodeEncoder.transform(walk[0]))

    # Generate features for nodes
    def generateNodeFeatures(self, totalNodes, wvi, j):
        nodeFeatures = torch.zeros(totalNodes)
        nodeFeatures[wvi[j]] = 1
        return nodeFeatures

    # Training graph embedding model
    def learnEmbedding(self, model, wvi):
        for j in range(len(wvi)):
            for k in range(max(0,j-self.windowSize) , min(j+self.windowSize, len(wvi))):
                # generate features
                nodeFeatures = self.generateNodeFeatures(self.totalNodes, wvi, j)
                out = self.model.forward(nodeFeatures)
                loss = torch.log(torch.sum(torch.exp(out))) - out[wvi[k]]
                loss.backward()
                for param in self.model.parameters():
                    param.data.sub_(self.lr*param.grad)
                    param.grad.data.zero_()
        return self.model

    # Training node embedding model
    def learnNodeEmbedding(self, model):
        self.model = model
        for startNode in list(self.graph.nodes):
            for i in range(self.numbOfWalksPerVertex):
                walkStartNode = self.RandomWalk(startNode, self.walkLength)
                self.model = self.learnEmbedding(self.model, walkStartNode)
        return self.model

    # Get node embedding for a specific node, i.e., "node"
    def getNodeEmbedding(self, node):
        return self.model.W1[int(self.nodeEncoder.transform([node]))].data

    # Training graph embedding model
    def learnEdgeEmbedding(self, model):
        self.model = model
        for startNode in list(self.graph.nodes):
            for i in range(self.numbOfWalksPerVertex):
                walkStartNode = self.RandomWalk(startNode, self.walkLength)
                self.model = self.learnEmbedding(self.model, walkStartNode)
        return self.model

    # Get edge embedding for a specific edge having source node, i.e., "srcNode" and destination node, i.e., dstNode
    def getEdgeEmbedding(self, srcNode, dstNode):
        return operator_hadamard(self.getNodeEmbedding(srcNode), self.getNodeEmbedding(dstNode))

    # Create a context graph
    def create_context_graph(self, max_num_layers, workers=1, verbose=0,):

        pair_distances = self._compute_structural_distance(
            max_num_layers, workers, verbose,)
        layers_adj, layers_distances = self._get_layer_rep(pair_distances)
        pd.to_pickle(layers_adj, self.temp_path + 'layers_adj.pkl')

        layers_accept, layers_alias = self._get_transition_probs(
            layers_adj, layers_distances)
        pd.to_pickle(layers_alias, self.temp_path + 'layers_alias.pkl')
        pd.to_pickle(layers_accept, self.temp_path + 'layers_accept.pkl')

    def prepare_biased_walk(self,):

        sum_weights = {}
        sum_edges = {}
        average_weight = {}
        gamma = {}
        layer = 0
        while (os.path.exists(self.temp_path+'norm_weights_distance-layer-' + str(layer)+'.pkl')):
            probs = pd.read_pickle(
                self.temp_path+'norm_weights_distance-layer-' + str(layer)+'.pkl')
            for v, list_weights in probs.items():
                sum_weights.setdefault(layer, 0)
                sum_edges.setdefault(layer, 0)
                sum_weights[layer] += sum(list_weights)
                sum_edges[layer] += len(list_weights)

            average_weight[layer] = sum_weights[layer] / sum_edges[layer]

            gamma.setdefault(layer, {})

            for v, list_weights in probs.items():
                num_neighbours = 0
                for w in list_weights:
                    if (w > average_weight[layer]):
                        num_neighbours += 1
                gamma[layer][v] = num_neighbours

            layer += 1

        pd.to_pickle(average_weight, self.temp_path + 'average_weight')
        pd.to_pickle(gamma, self.temp_path + 'gamma.pkl')

    def train(self, window_size=5, workers=3, epochs=5):

        # pd.read_pickle(self.temp_path+'walks.pkl')
        sentences = self.sentences

        print("Learning representation...")
        model = Word2Vec(sentences, vector_size=self.embedDim, window=window_size, min_count=0, hs=1, sg=1, workers=workers,
                         epochs=epochs)
        print("Learning representation done!")
        self.w2v_model = model

        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

    def _compute_ordered_degreelist(self, max_num_layers):

        degreeList = {}
        vertices = self.idx  # self.g.nodes()
        for v in vertices:
            degreeList[v] = self._get_order_degreelist_node(v, max_num_layers)
        return degreeList

    def _get_order_degreelist_node(self, root, max_num_layers=None):
        if max_num_layers is None:
            max_num_layers = float('inf')

        ordered_degree_sequence_dict = {}
        visited = [False] * len(self.graph.nodes())
        queue = deque()
        level = 0
        queue.append(root)
        visited[root] = True

        while (len(queue) > 0 and level <= max_num_layers):

            count = len(queue)
            if self.opt1_reduce_len:
                degree_list = {}
            else:
                degree_list = []
            while (count > 0):

                top = queue.popleft()
                node = int(self.nodeEncoder.inverse_transform([top]))
                degree = len(self.graph[node])

                if self.opt1_reduce_len:
                    degree_list[degree] = degree_list.get(degree, 0) + 1
                else:
                    degree_list.append(degree)

                for nei in self.graph[node]:

                    nei_idx = int(self.nodeEncoder.transform([nei]))
                    if not visited[nei_idx]:
                        visited[nei_idx] = True
                        queue.append(nei_idx)
                count -= 1
            if self.opt1_reduce_len:
                orderd_degree_list = [(degree, freq)
                                      for degree, freq in degree_list.items()]
                orderd_degree_list.sort(key=lambda x: x[0])
            else:
                orderd_degree_list = sorted(degree_list)
            ordered_degree_sequence_dict[level] = orderd_degree_list
            level += 1

        return ordered_degree_sequence_dict

    def _compute_structural_distance(self, max_num_layers, workers=1, verbose=0,):

        if os.path.exists(self.temp_path+'structural_dist.pkl'):
            structural_dist = pd.read_pickle(
                self.temp_path+'structural_dist.pkl')
        else:
            if self.opt1_reduce_len:
                dist_func = cost_max
            else:
                dist_func = cost

            if os.path.exists(self.temp_path + 'degreelist.pkl'):
                degreeList = pd.read_pickle(self.temp_path + 'degreelist.pkl')
            else:
                degreeList = self._compute_ordered_degreelist(max_num_layers)
                pd.to_pickle(degreeList, self.temp_path + 'degreelist.pkl')

            if self.opt2_reduce_sim_calc:
                degrees = self._create_vectors()
                degreeListsSelected = {}
                vertices = {}
                n_nodes = len(self.idx)
                for v in self.idx:  # c:list of vertex
                    nbs = get_vertices(
                        v, len(self.graph[int(self.nodeEncoder.inverse_transform([v]))]), degrees, n_nodes)
                    vertices[v] = nbs  # store nbs
                    degreeListsSelected[v] = degreeList[v]  # store dist
                    for n in nbs:
                        # store dist of nbs
                        degreeListsSelected[n] = degreeList[n]
            else:
                vertices = {}
                for v in degreeList:
                    vertices[v] = [vd for vd in degreeList.keys() if vd > v]

            results = Parallel(n_jobs=workers, verbose=verbose,)(
                delayed(compute_dtw_dist)(part_list, degreeList, dist_func) for part_list in partition_dict(vertices, workers))
            dtw_dist = dict(ChainMap(*results))

            structural_dist = convert_dtw_struc_dist(dtw_dist)
            pd.to_pickle(structural_dist, self.temp_path +
                         'structural_dist.pkl')

        return structural_dist

    def _create_vectors(self):
        degrees = {}  # sotre v list of degree
        degrees_sorted = set()  # store degree
        G = self.graph
        for v in self.idx:
            degree = len(G[ int(self.nodeEncoder.inverse_transform([v])) ])
            degrees_sorted.add(degree)
            if (degree not in degrees):
                degrees[degree] = {}
                degrees[degree]['vertices'] = []
            degrees[degree]['vertices'].append(v)
        degrees_sorted = np.array(list(degrees_sorted), dtype='int')
        degrees_sorted = np.sort(degrees_sorted)

        l = len(degrees_sorted)
        for index, degree in enumerate(degrees_sorted):
            if (index > 0):
                degrees[degree]['before'] = degrees_sorted[index - 1]
            if (index < (l - 1)):
                degrees[degree]['after'] = degrees_sorted[index + 1]

        return degrees

    def _get_layer_rep(self, pair_distances):
        layer_distances = {}
        layer_adj = {}
        for v_pair, layer_dist in pair_distances.items():
            for layer, distance in layer_dist.items():
                vx = v_pair[0]
                vy = v_pair[1]

                layer_distances.setdefault(layer, {})
                layer_distances[layer][vx, vy] = distance

                layer_adj.setdefault(layer, {})
                layer_adj[layer].setdefault(vx, [])
                layer_adj[layer].setdefault(vy, [])
                layer_adj[layer][vx].append(vy)
                layer_adj[layer][vy].append(vx)

        return layer_adj, layer_distances

    def _get_transition_probs(self, layers_adj, layers_distances):
        layers_alias = {}
        layers_accept = {}

        for layer in layers_adj:

            neighbors = layers_adj[layer]
            layer_distances = layers_distances[layer]
            node_alias_dict = {}
            node_accept_dict = {}
            norm_weights = {}

            for v, neighbors in neighbors.items():
                e_list = []
                sum_w = 0.0

                for n in neighbors:
                    if (v, n) in layer_distances:
                        wd = layer_distances[v, n]
                    else:
                        wd = layer_distances[n, v]
                    w = np.exp(-float(wd))
                    e_list.append(w)
                    sum_w += w

                e_list = [x / sum_w for x in e_list]
                norm_weights[v] = e_list
                accept, alias = create_alias_table(e_list)
                node_alias_dict[v] = alias
                node_accept_dict[v] = accept

            pd.to_pickle(
                norm_weights, self.temp_path + 'norm_weights_distance-layer-' + str(layer)+'.pkl')

            layers_alias[layer] = node_alias_dict
            layers_accept[layer] = node_accept_dict

        return layers_accept, layers_alias
