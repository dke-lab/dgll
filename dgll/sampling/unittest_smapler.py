import unittest
import dgll
from dgll.sampling.dgllsampler import DGLLNeighborSampler
import pickle
from dgll import backend as F





class TestNeighborSampler(unittest.TestCase):
    def setUp(self):
        # Create a simple test graph
        self.g = pickle.load(open("../dataset/cora.graph", "rb"))
        self.seed_nodes = tseed_nodes = F.tensor([5, 4, 2, 9, 6, 7])  # Example seed nodes
        self.fanouts = [2, 3]
        self.sampler =DGLLNeighborSampler(self.fanouts)

    def test_sample_output_shapes(self):
        input_nodes, output_nodes, subgs = self.sampler.sample(self.g, self.seed_nodes)
        # Check that input_nodes and output_nodes have the correct shape
        self.assertEqual(input_nodes.shape[0], subgs[0].num_src_nodes())
        self.assertEqual(output_nodes.shape[0], self.seed_nodes.shape[0])

    def test_sample_subgraph(self):
        input_nodes, output_nodes, subgs = self.sampler.sample(self.g, self.seed_nodes)
        # Check that the number of subgraphs matches the number of fanouts
        self.assertEqual(len(subgs), len(self.fanouts))
        # Check that the subgraphs are indeed blocks (message flow graphs)
        for sg in subgs:
            self.assertIsInstance(sg, dgll.sampling.base_sampler.sugbraph)
        # Verify the input nodes are correct
        self.assertTrue(F.equal(input_nodes, subgs[0].src_data))

    def test_sample_with_empty_fanouts(self):
        empty_sampler = DGLLNeighborSampler([])
        input_nodes, output_nodes, subgs = empty_sampler.sample(self.g, self.seed_nodes)
        # With empty fanouts, there should be no subgraphs
        self.assertEqual(len(subgs), 0)
        # The input and output nodes should be the same as the seed nodes
        self.assertTrue(torch.equal(input_nodes, self.seed_nodes))
        self.assertTrue(torch.equal(output_nodes, self.seed_nodes))

    def test_sample_with_invalid_fanout(self):
        # Fanout greater than the number of nodes should not raise an error
        large_fanout_sampler = DGLLNeighborSampler([10])
        input_nodes, output_nodes, subgs = large_fanout_sampler.sample(self.g, self.seed_nodes)
        self.assertEqual(len(subgs), 1)  # Should still return one subgraph
        # The resulting subgraph should have at most 5 source nodes since the original graph has 5 nodes
        self.assertLessEqual(subgs[0].number_of_src_nodes(), 5)

if __name__ == '__main__':
    unittest.main()



