from dgll import backend as F
import random

class Base_sampler(object):
    """Base class for graph samplers.
    All graph samplers must subclass this class and override the ``sample``
    method.

        from dgll.sampling import Sampler
        class MySampler(Sampler):
            def __init__(self):
                super().__init__()

            def sample(self, g, indices):
                return sample
    """

    def sample(self, g, nodes):
        """Abstract sample method.

        Parameters
        ----------
        g : DGraph
            Input graph.
        nodes : tensor
            nodes for which sample is generated
        """
        raise NotImplementedError

    def _subgraph(self, nodes, neighbors_list):
        src_list = []
        dst_list = []

        for i, neighbors in enumerate(neighbors_list):
            dst_node = nodes[i]
            for src_node in neighbors:
                src_list.append(src_node)
                dst_list.append(dst_node.item())

        src_tensor = F.tensor(src_list)
        dst_tensor = F.tensor(dst_list)

        return sugbraph(src_tensor, dst_tensor)

    def sample_neighbours(self, g, nodes, fanout = None):
        neighbors_list = g.get_neighbors(nodes)
        random_neighbors = []
        for neighbors in neighbors_list:
            if len(neighbors) == 0:
                random_neighbors.append([])  # No neighbors available
            elif fanout is None:
                random_neighbors.append(neighbors)  # Return all neighbors if fanout is None
            elif len(neighbors) <= fanout:
                random_neighbors.append(neighbors)  # Return all neighbors if less than fanout
            else:
                random_neighbors.append(random.sample(neighbors, fanout))  # Randomly sample neighbors
        subg = self._subgraph(nodes, random_neighbors)
        return subg



