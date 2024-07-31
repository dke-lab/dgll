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

