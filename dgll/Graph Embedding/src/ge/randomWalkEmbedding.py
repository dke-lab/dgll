from abc import ABC, abstractmethod
from sklearn import preprocessing
import warnings
try:
    from utils import custom_formatwarning
except ModuleNotFoundError:
    from .utils import custom_formatwarning

class RandomWalkEmbedding:
    def __init__(self, graph, walkLength, embedDim, numbOfWalksPerVertex, windowSize, lr):
        self.graph = graph
        # validate arguments
        if walkLength == 0:
            self.walkLength = 3
            warnings.warn("Set Walk to default: {}".format(self.walkLength))
        else:
            self.walkLength = walkLength

        if embedDim == 0:
            self.embedDim = 2
            warnings.warn("Set Embedding Dimention to default: {}".format(self.embedDim))
        else:
            self.embedDim = embedDim

        if numbOfWalksPerVertex == 3:
            self.numbOfWalksPerVertex = 3
            warnings.warn("Setting Learning Rate to default: {}".format(self.numbOfWalksPerVertex))
        else:
            self.numbOfWalksPerVertex = numbOfWalksPerVertex

        if windowSize == 0:
            self.windowSize = 3
            warnings.warn("Set Context Window to default: {}".format(self.windowSize))
        else:
            self.windowSize = windowSize

        if lr == 0:
            self.lr = 0.25
            warnings.warn("Set Learning Rate to default: {}".format(self.lr))
        else:
            self.lr = lr
        self.adj_list, self.nodeEncoder = self.graph_to_adjList(graph)
        self.totalNodes = graph.number_of_nodes()




    def encoder(self, graph):
        nodeEncoder = preprocessing.LabelEncoder()
        return nodeEncoder.fit(list(graph.nodes()))

    def graph_to_adjList(self, graph):
        nodeEncoder = self.encoder(graph)
        adj_list1 = [None] * graph.number_of_nodes()
        for node, edges in list(graph.adjacency()):

            #     print(node, list(edges.keys()))
            adj_list1[nodeEncoder.transform([node])[0]] = list(nodeEncoder.transform(list(edges.keys())))
        return adj_list1, nodeEncoder

    def getAdjacencyList(self):
        return self.adj_list

    def getGraph(self):
        return self.graph


    @abstractmethod
    def generateWalk(self):
        pass

    @abstractmethod
    def learnEmbedding(self):
        pass

    @abstractmethod
    def learnNodeEmbedding(self):
        pass


    @abstractmethod
    def getNodeEmbedding(self):
        pass

    @abstractmethod
    def learnEdgeEmbedding(self):
        pass


    @abstractmethod
    def getEdgeEmbedding(self):
        pass