import torch
import numpy as np
from collections import defaultdict
import warnings
import sys
try:
    from utils import operator_hadamard, custom_formatwarning
    from randomWalkEmbedding import RandomWalkEmbedding
except ModuleNotFoundError:
    from .utils import operator_hadamard, custom_formatwarning
    from .randomWalkEmbedding import RandomWalkEmbedding

class Node2vec(RandomWalkEmbedding):
    # Constructor
    def __init__(self, graph = None, walkLength = 0, embedDim = 0, numbOfWalksPerVertex = 0, \
                 windowSize = 0, lr = 0, p = 0, q = 0):
        if graph is None:
            warnings.warn("Provide a graph: {}".format(graph))
            sys.exit()
        super(Node2vec, self).__init__(graph, walkLength, embedDim, numbOfWalksPerVertex,  windowSize, lr)

        # validate arguments
        if p == 0:
            self.p = 0.5
            warnings.warn("Set p to default: {}".format(self.p))
        else:
            self.p = p
        if q == 0:
            self.q = 0.8
            warnings.warn("Set q to default: {}".format(self.q))
        else:
            self.q = q
        self.model = None

    # Calculating Probabilities for traversing a node
    def computeProbabilities(self, source_node):
        probs = defaultdict(dict)
        probs[source_node]['probabilities'] = dict()
        for current_node in self.graph.neighbors(source_node):
            probs_ = list()
            for destination in self.graph.neighbors(current_node):
                if source_node == destination:
                    prob_ = self.graph[current_node][destination].get('weight',1) * (1/self.p)
                elif destination in self.graph.neighbors(source_node):
                    prob_ = self.graph[current_node][destination].get('weight',1)
                else:
                    prob_ = self.graph[current_node][destination].get('weight',1) * (1/self.q)
                probs_.append(prob_)
            probs[source_node]['probabilities'][current_node] = probs_/np.sum(probs_)
        return probs

    # Walks generation
    def RandomWalk(self, startNode, walkLength):
        # walk contains encoded node labels
        walk = [int(self.nodeEncoder.transform([startNode]))]
        walkOptions = list(self.graph[startNode])
        if len(walkOptions)==0:
            return walk
        firstStep = np.random.choice(walkOptions)
        walk.append(int(self.nodeEncoder.transform([firstStep])))
        for k in range(walkLength-2):
            walkOptions = list(self.graph[(self.nodeEncoder.inverse_transform([walk[-1]])[0])])
            if len(walkOptions)==0:
                break
            probs = self.computeProbabilities((self.nodeEncoder.inverse_transform([walk[-2]])[0]))
            probabilities = probs[(self.nodeEncoder.inverse_transform([walk[-2]])[0])] \
                ['probabilities'][(self.nodeEncoder.inverse_transform([walk[-1]])[0])]
            # Traverse a vertex from a neighbors of node 'node"
            nextStep = np.random.choice(walkOptions, p=probabilities)
            walk.append(int(self.nodeEncoder.transform([nextStep])))
        return walk

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

    # Get node embedding for a specific node, i.e., "node"
    def getNodeEmbedding(self, node):
        return self.model.W1[int(self.nodeEncoder.transform([node]))].data

    # Training node embedding model
    def learnNodeEmbedding(self, model):
        self.model = model
        for startNode in list(self.graph.nodes):
            for i in range(self.numbOfWalksPerVertex):
                walkStartNode = self.RandomWalk(startNode, self.walkLength)
                self.model = self.learnEmbedding(self.model, walkStartNode)
        return self.model

    # Training edge embedding model
    def learnEdgeEmbedding(self, model):
        self.model = model
        for startNode in list(self.graph.nodes):
            for i in range(self.numbOfWalksPerVertex):
                walkStartNode = self.RandomWalk(startNode, self.walkLength)
                self.model = self.learnEmbedding(self.model, walkStartNode)
        return self.model

    # Get edge embedding for a specific edge having source node, i.e., "srcNode" and destination node, i.e., dstNode
    def getEdgeEmbedding(self, srcNode, dstNode):
        # Operator_hadamrd defined in Utils
        return operator_hadamard(self.getNodeEmbedding(srcNode), self.getNodeEmbedding(dstNode))



