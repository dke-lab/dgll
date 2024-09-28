import unittest
from unittest.mock import patch, MagicMock
from dgll.data import RedditDataset, PubmedGraphDataset, CiteseerGraphDataset, CoraGraphDataset

# Import the functions to test
from load_data import load_reddit, load_pubmed, load_citeseer, load_arxiv, load_cora, load_products


class TestDatasetLoading(unittest.TestCase):

    @patch('load_data.RedditDataset')
    def test_load_reddit(self, MockRedditDataset):
        # Mock the RedditDataset object
        mock_data = MagicMock()
        MockRedditDataset.return_value = [mock_data]

        # Call the function with updated root directory
        g = load_reddit('/mnt/g/MQ-GNN-Seraph/dataset')

        # Check if the function returns the expected graph
        MockRedditDataset.assert_called_once_with(raw_dir='/mnt/g/MQ-GNN-Seraph/dataset', self_loop=True)
        self.assertEqual(g, mock_data)

    @patch('load_data.PubmedGraphDataset')
    def test_load_pubmed(self, MockPubmedGraphDataset):
        # Mock the PubmedGraphDataset object
        mock_data = MagicMock()
        MockPubmedGraphDataset.return_value = [mock_data]

        # Call the function with updated root directory
        g = load_pubmed('/mnt/g/MQ-GNN-Seraph/dataset')

        # Check if the function returns the expected graph
        MockPubmedGraphDataset.assert_called_once_with(raw_dir='/mnt/shared/dataset/', reverse_edge=True)
        self.assertEqual(g, mock_data)

    @patch('load_data.CiteseerGraphDataset')
    def test_load_citeseer(self, MockCiteseerGraphDataset):
        # Mock the CiteseerGraphDataset object
        mock_data = MagicMock()
        MockCiteseerGraphDataset.return_value = [mock_data]

        # Call the function with updated root directory
        g = load_citeseer('/mnt/g/MQ-GNN-Seraph/dataset')

        # Check if the function returns the expected graph
        MockCiteseerGraphDataset.assert_called_once_with(raw_dir='/mnt/g/MQ-GNN-Seraph/dataset', reverse_edge=True)
        self.assertEqual(g, mock_data)

    @patch('load_data.dgllNodePropPredDataset')
    def test_load_arxiv(self, MockdgllNodePropPredDataset):
        # Mock the dgllNodePropPredDataset object for 'ogbn-arxiv'
        mock_data = MagicMock()
        MockdgllNodePropPredDataset.return_value = mock_data

        # Call the function (root directory isn't used in this one)
        g = load_arxiv('/mnt/g/MQ-GNN-Seraph/dataset')

        # Check if the function returns the expected graph
        MockdgllNodePropPredDataset.assert_called_once_with('ogbn-arxiv')
        self.assertEqual(g, mock_data)

    @patch('load_data.CoraGraphDataset')
    def test_load_cora(self, MockCoraGraphDataset):
        # Mock the CoraGraphDataset object
        mock_data = MagicMock()
        MockCoraGraphDataset.return_value = mock_data

        # Call the function
        g = load_cora('/mnt/shared/dataset/')

        # Check if the function returns the expected graph
        MockCoraGraphDataset.assert_called_once()  # Check it was called
        self.assertEqual(g, mock_data)  # Check the returned graph

    @patch('load_data.dgllNodePropPredDataset')
    def test_load_products(self, MockdgllNodePropPredDataset):
        # Mock the dgllNodePropPredDataset object for 'ogbn-products'
        mock_data = MagicMock()
        MockdgllNodePropPredDataset.return_value = mock_data

        # Call the function (root directory isn't used in this one)
        g = load_products('/mnt/g/MQ-GNN-Seraph/dataset')

        # Check if the function returns the expected graph
        MockdgllNodePropPredDataset.assert_called_once_with('ogbn-products')
        self.assertEqual(g, mock_data)


if __name__ == '__main__':
    unittest.main()
